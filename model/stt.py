import numpy as np
from faster_whisper import WhisperModel

from . import config


class FasterWhisperSTT:
    def __init__(self):
        self.model = WhisperModel(
            config.WHISPER_MODEL_SIZE,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )

    def transcribe(self, audio: np.ndarray):
        if audio.size == 0:
            return ""
        rms = float(np.sqrt(np.mean(audio * audio)))
        if rms < config.MIN_AUDIO_RMS:
            return ""

        segments, info = self.model.transcribe(
            audio,
            language="en",
            beam_size=config.WHISPER_BEAM_SIZE,
            vad_filter=True,
            condition_on_previous_text=False,
            temperature=0.0,
            no_speech_threshold=config.WHISPER_NO_SPEECH_THRESHOLD,
            log_prob_threshold=config.WHISPER_LOGPROB_THRESHOLD,
            compression_ratio_threshold=config.WHISPER_COMPRESSION_RATIO_THRESHOLD,
        )

        parts = []
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            no_speech = float(getattr(seg, "no_speech_prob", 0.0))
            avg_logprob = float(getattr(seg, "avg_logprob", 0.0))
            compression = float(getattr(seg, "compression_ratio", 0.0))
            if no_speech >= config.WHISPER_NO_SPEECH_THRESHOLD:
                continue
            if avg_logprob < config.WHISPER_LOGPROB_THRESHOLD:
                continue
            if compression > config.WHISPER_COMPRESSION_RATIO_THRESHOLD:
                continue
            parts.append(text)

        if not parts:
            return ""
        return " ".join(parts).strip()


class OpenAIWhisperBackend:

    def __init__(self):
        import whisper

        self.model = whisper.load_model(
            config.WHISPER_MODEL_SIZE.replace(".en", "") + ".en")

    def transcribe(self, audio: np.ndarray) -> str:
        result = self.model.transcribe(audio, language="en", fp16=False)
        return result["text"].strip()
