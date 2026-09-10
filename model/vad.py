import collections
from typing import Iterator, Tuple

import numpy as np
import webrtcvad

from . import config

# Event = (event_type, payload)
#   ("speech_start", None)
#   ("utterance", np.ndarray)   float32 mono PCM @ MIC_SAMPLE_RATE
Event = Tuple[str, object]


class TurnDetector:
    def __init__(self):
        self.vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)
        self.frame_samples = int(
            config.MIC_SAMPLE_RATE * config.MIC_FRAME_MS / 1000)
        self.frame_bytes = self.frame_samples * 2  # int16 mono

        self._padding_frames = max(
            1, int(config.VAD_PADDING_MS / config.MIC_FRAME_MS))
        self._silence_frames_needed = max(
            1, int(config.VAD_SILENCE_MS_END_OF_TURN / config.MIC_FRAME_MS)
        )
        self._min_speech_frames = max(
            1, int(config.VAD_MIN_UTTERANCE_MS / config.MIC_FRAME_MS))
        self._max_utterance_frames = max(
            1, int(config.MAX_UTTERANCE_MS / config.MIC_FRAME_MS))

        self._ring: "collections.deque[tuple[bytes, bool]]" = collections.deque(
            maxlen=self._padding_frames
        )
        self._triggered = False
        self._voiced_frames = []
        self._trailing_silence = 0
        self.enabled = True  # mirrors mute()/unmute() from mic_vad.py

    def mute(self):
        self.enabled = False

    def unmute(self):
        self.enabled = True

    def reset(self):
        self._ring.clear()
        self._triggered = False
        self._voiced_frames = []
        self._trailing_silence = 0

    def process_frame(self, frame: bytes) -> Iterator[Event]:
        if not self.enabled:
            return
        if len(frame) != self.frame_bytes:
            return

        is_speech = self.vad.is_speech(frame, config.MIC_SAMPLE_RATE)

        if not self._triggered:
            self._ring.append((frame, is_speech))
            voiced = sum(1 for _, s in self._ring if s)
            if len(self._ring) >= 3 and voiced / len(self._ring) >= config.VAD_START_VOICED_RATIO:
                self._triggered = True
                yield ("speech_start", None)
                self._voiced_frames = [f for f, _ in self._ring]
                self._ring.clear()
                self._trailing_silence = 0
        else:
            self._voiced_frames.append(frame)
            if is_speech:
                self._trailing_silence = 0
            else:
                self._trailing_silence += 1

            too_long = len(self._voiced_frames) >= self._max_utterance_frames
            if self._trailing_silence >= self._silence_frames_needed or too_long:
                self._triggered = False
                if len(self._voiced_frames) >= self._min_speech_frames:
                    pcm = b"".join(self._voiced_frames)
                    audio = np.frombuffer(pcm, dtype=np.int16).astype(
                        np.float32) / 32768.0
                    # Reject extremely quiet segments before Whisper can hallucinate
                    rms = float(np.sqrt(np.mean(audio * audio))
                                ) if audio.size else 0.0
                    if rms >= config.MIN_AUDIO_RMS:
                        yield ("utterance", audio)
                self._voiced_frames = []
                self._trailing_silence = 0
