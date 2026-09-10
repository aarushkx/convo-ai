import numpy as np

from . import config


class KokoroTTS:
    def __init__(self):
        # Heavy import, so it is loaded only when the class is created
        from kokoro import KPipeline

        self.pipeline = KPipeline(lang_code=config.KOKORO_LANG_CODE)

    def synthesize_chunk(self, text: str):
        """
        text: one sentence/phrase-sized chunk
        yields: float32 numpy arrays of raw PCM audio @ TTS_SAMPLE_RATE,
                one per internal sub-chunk Kokoro produces
        """
        text = text.strip()
        if not text:
            return
        generator = self.pipeline(
            text,
            voice=config.KOKORO_VOICE,
            speed=config.KOKORO_SPEED,
        )
        for _graphemes, _phonemes, audio in generator:
            arr = np.asarray(audio, dtype=np.float32)
            if arr.size:
                yield arr
