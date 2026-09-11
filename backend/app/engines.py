import logging

from model.stt import FasterWhisperSTT
from model.tts import KokoroTTS

logger = logging.getLogger("convo-ai.engines")


class Engines:
    stt: FasterWhisperSTT = None
    tts: KokoroTTS = None

    @classmethod
    def load(cls):
        if cls.stt is None:
            logger.info("Loading Whisper STT engine ...")
            cls.stt = FasterWhisperSTT()
        if cls.tts is None:
            logger.info("Loading Kokoro TTS engine ...")
            cls.tts = KokoroTTS()
        logger.info("Model engines ready.")

    @classmethod
    def ready(cls):
        return cls.stt is not None and cls.tts is not None
