"""
RealtimeSession handles one WebSocket connection.

It manages the full voice pipeline: speech-to-text, LLM response
generation, and text-to-speech. Incoming audio is received from the
WebSocket, and generated audio is stored in `out_q` to be sent back
to the client.

Barge-in is handled using a `turn_id`. Each task gets a turn ID, and
when a new turn starts, older tasks detect that their ID is outdated
and stop processing.

No locks are held while STT, LLM, or TTS is running, so a new turn
can interrupt the current processing quickly, within about one audio
frame (30 ms).
"""

import logging
import queue
import struct
import threading

import numpy as np

from model import config
from model.llm import OllamaLLM
from model.vad import TurnDetector

logger = logging.getLogger("convo-ai.session")

# binary frame tag: [1 byte type][4 byte turn_id LE][float32 PCM]
AUDIO_MSG_TYPE = 1


def _drain(q: "queue.Queue"):
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


def pack_audio_frame(turn_id: int, audio: np.ndarray) -> bytes:
    header = struct.pack("<BI", AUDIO_MSG_TYPE, turn_id & 0xFFFFFFFF)
    return header + audio.astype(np.float32).tobytes()


class RealtimeSession:
    def __init__(self, session_id: str, stt, tts):
        self.session_id = session_id
        self.stt = stt
        self.tts = tts
        self.llm = OllamaLLM()

        self.detector = TurnDetector()
        self.utterance_q: "queue.Queue[np.ndarray]" = queue.Queue()
        self.tts_q: "queue.Queue[tuple[int, str]]" = queue.Queue()
        # Messages sent to the WebSocket layer to forward to the client.
        # `("event", dict)` is used for JSON control messages, while
        # `("audio", turn_id, np.ndarray)` is used for generated speech audio
        self.out_q: "queue.Queue[tuple]" = queue.Queue()

        self.interrupt_event = threading.Event()
        self._shutdown = threading.Event()
        self._turn_lock = threading.Lock()
        self._turn_id = 0

        self._brain_thread = threading.Thread(
            target=self._brain_loop, daemon=True)
        self._voice_thread = threading.Thread(
            target=self._voice_loop, daemon=True)

    # LIFECYCLE

    def start(self):
        self._brain_thread.start()
        self._voice_thread.start()
        self.out_q.put(("event", {"type": "ready"}))

    def stop(self):
        self._shutdown.set()
        self.interrupt_event.set()
        _drain(self.utterance_q)
        _drain(self.tts_q)

    # INBOUND AUDIO

    def feed_audio_frame(self, frame: bytes):
        """Call with one MIC_FRAME_MS int16 mono PCM frame from the client."""
        if self._shutdown.is_set():
            return
        for event, payload in self.detector.process_frame(frame):
            if event == "speech_start":
                self._handle_barge_in()
            elif event == "utterance":
                self.utterance_q.put(payload)

    def end_call(self):
        """User explicitly ended the call."""
        self.out_q.put(("event", {"type": "call_ended"}))
        self.stop()

    # TURN BOOKKEEPING

    def _new_turn(self) -> int:
        with self._turn_lock:
            self._turn_id += 1
            return self._turn_id

    def _current_turn(self) -> int:
        with self._turn_lock:
            return self._turn_id

    def _handle_barge_in(self):
        if not config.ALLOW_BARGE_IN:
            return
        prior_turn = self._current_turn()
        new_turn = self._new_turn()
        self.interrupt_event.set()
        _drain(self.tts_q)
        if prior_turn > 0:
            self.out_q.put(
                ("event", {"type": "interrupted", "turn_id": prior_turn}))
        self.out_q.put(
            ("event", {"type": "speech_start", "turn_id": new_turn}))

    # BRAIN: STT -> streaming LLM -> sentence chunks

    def _brain_loop(self):
        while not self._shutdown.is_set():
            try:
                audio = self.utterance_q.get(timeout=0.5)
            except queue.Empty:
                continue

            # speech_start already minted a turn id for this utterance
            turn_id = self._current_turn()
            try:
                text = self.stt.transcribe(audio)
            except Exception:
                logger.exception("STT failed for session %s", self.session_id)
                self.out_q.put(
                    ("event", {"type": "error", "message": "stt_failed"}))
                continue

            if self._shutdown.is_set() or turn_id != self._current_turn():
                continue
            if not text:
                continue

            self.out_q.put(
                ("event", {"type": "user_transcript", "turn_id": turn_id, "text": text}))
            self.interrupt_event.clear()

            try:
                interrupted = False
                for sentence_chunk in self.llm.stream_reply(text):
                    if self.interrupt_event.is_set() or turn_id != self._current_turn():
                        interrupted = True
                        break
                    self.out_q.put(
                        ("event", {"type": "assistant_partial",
                         "turn_id": turn_id, "text": sentence_chunk})
                    )
                    self.tts_q.put((turn_id, sentence_chunk))
                if not interrupted and turn_id == self._current_turn():
                    self.out_q.put(
                        ("event", {"type": "assistant_done", "turn_id": turn_id}))
            except Exception:
                logger.exception(
                    "LLM streaming failed for session %s", self.session_id)
                self.out_q.put(
                    ("event", {"type": "error", "message": "llm_failed"}))

    # VOICE: TTS per sentence chunk, streamed out as it's generated

    def _voice_loop(self):
        while not self._shutdown.is_set():
            try:
                item = self.tts_q.get(timeout=0.5)
            except queue.Empty:
                continue
            turn_id, text_chunk = item
            if turn_id != self._current_turn() or self.interrupt_event.is_set():
                continue
            try:
                for audio_chunk in self.tts.synthesize_chunk(text_chunk):
                    if (
                        self._shutdown.is_set()
                        or turn_id != self._current_turn()
                        or self.interrupt_event.is_set()
                    ):
                        break
                    self.out_q.put(("audio", turn_id, audio_chunk))
            except Exception:
                logger.exception("TTS failed for session %s", self.session_id)
                self.out_q.put(
                    ("event", {"type": "error", "message": "tts_failed"}))
