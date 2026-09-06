from faster_whisper import WhisperModel

from . import config


class STT:
    def __init__(self):
        self.model = WhisperModel(
            config.MODEL_NAME,
            device="cpu",
            compute_type="int8"
        )

    def transcribe(self, audio):
        segments, _ = self.model.transcribe(
            audio,
            language=config.LANGUAGE,
        )
        return " ".join(segment.text for segment in segments)
