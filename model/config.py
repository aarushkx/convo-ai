# These values can be overridden by environment variables for easier configuration of deployment

import os


def _env(name, default, cast=str):
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return cast(val)
    except (TypeError, ValueError):
        return default


# Audio I/O
MIC_SAMPLE_RATE = _env("MIC_SAMPLE_RATE", 16000, int)
MIC_FRAME_MS = _env("MIC_FRAME_MS", 30, int)
MIC_CHANNELS = _env("MIC_CHANNELS", 1, int)
TTS_SAMPLE_RATE = _env("TTS_SAMPLE_RATE", 24000, int)

# Voice Activity Detection / turn-taking
VAD_AGGRESSIVENESS = _env("VAD_AGGRESSIVENESS", 3, int)
VAD_SILENCE_MS_END_OF_TURN = _env("VAD_SILENCE_MS_END_OF_TURN", 900, int)
VAD_MIN_UTTERANCE_MS = _env("VAD_MIN_UTTERANCE_MS", 450, int)
VAD_PADDING_MS = _env("VAD_PADDING_MS", 300, int)
VAD_START_VOICED_RATIO = _env("VAD_START_VOICED_RATIO", 0.80, float)
MAX_UTTERANCE_MS = _env("MAX_UTTERANCE_MS", 15000, int)

# STT
WHISPER_MODEL_SIZE = _env("WHISPER_MODEL_SIZE", "tiny.en")
WHISPER_DEVICE = _env("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = _env("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_BEAM_SIZE = _env("WHISPER_BEAM_SIZE", 3, int)
WHISPER_NO_SPEECH_THRESHOLD = _env("WHISPER_NO_SPEECH_THRESHOLD", 0.55, float)
WHISPER_LOGPROB_THRESHOLD = _env("WHISPER_LOGPROB_THRESHOLD", -0.80, float)
WHISPER_COMPRESSION_RATIO_THRESHOLD = _env(
    "WHISPER_COMPRESSION_RATIO_THRESHOLD", 2.2, float)
MIN_AUDIO_RMS = _env("MIN_AUDIO_RMS", 0.008, float)

# LLM
OLLAMA_MODEL = _env("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_HOST = _env("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_NUM_CTX = _env("OLLAMA_NUM_CTX", 2048, int)
OLLAMA_TEMPERATURE = _env("OLLAMA_TEMPERATURE", 0.4, float)

SYSTEM_PROMPT = _env(
    "SYSTEM_PROMPT",
    (
        "You're a warm, easygoing person having a real spoken conversation out loud, "
        "not a chatbot writing text. Talk the way a thoughtful friend would: use "
        "natural sentence variety, ask follow-up questions when it fits, react "
        "genuinely, and give a fuller answer when the topic actually calls for one "
        "instead of always cutting yourself short. There's no length limit - say as "
        "much as the moment needs, the way a person would, but don't ramble just to "
        "fill space. Use only what's actually in the user's transcript below; never "
        "invent or guess words that aren't there. If the transcript is unclear or "
        "nonsensical, just ask them to say it again instead of pretending you "
        "understood. Since this is spoken aloud: no markdown, lists, emojis, or "
        "asterisks."
    ),
)

# Sentence chunker
CHUNK_SENTENCE_ENDERS = (".", "!", "?", "\n")
CHUNK_SOFT_ENDERS = (",", ";", ":", "—")
CHUNK_MAX_CHARS = _env("CHUNK_MAX_CHARS", 220, int)
CHUNK_MIN_CHARS = _env("CHUNK_MIN_CHARS", 8, int)
CHUNK_TARGET_SENTENCES = _env("CHUNK_TARGET_SENTENCES", 1, int)

# TTS
KOKORO_LANG_CODE = _env("KOKORO_LANG_CODE", "a")
KOKORO_VOICE = _env("KOKORO_VOICE", "af_heart")
KOKORO_SPEED = _env("KOKORO_SPEED", 1.0, float)

# Barge-in
ALLOW_BARGE_IN = _env("ALLOW_BARGE_IN", "true").lower() in ("1", "true", "yes")


class Config:

    def __getattr__(self, item):
        return globals()[item]


config = Config()
