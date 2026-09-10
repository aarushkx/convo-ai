from typing import List, Optional
import re
import ollama

from . import config


class SentenceChunker:
    def __init__(self):
        self._buf = ""
        self._sentence_count = 0

    def feed(self, token: str):
        self._buf += token
        out = []
        while True:
            cut = self._find_boundary(self._buf, config.CHUNK_SENTENCE_ENDERS)
            if cut is None:
                if len(self._buf) >= config.CHUNK_MAX_CHARS:
                    cut = self._find_boundary(
                        self._buf, config.CHUNK_SOFT_ENDERS)
                    if cut is None:
                        cut = self._buf.rfind(" ", 0, config.CHUNK_MAX_CHARS)
                        if cut > config.CHUNK_MIN_CHARS:
                            chunk = self._emit(cut)
                            if chunk:
                                out.append(chunk)
                            continue
                break

            candidate = self._buf[:cut].strip()
            sentences = len(re.findall(r"[.!?]+(?=\s|$)", candidate))
            if sentences >= config.CHUNK_TARGET_SENTENCES or len(candidate) >= config.CHUNK_MAX_CHARS:
                chunk = self._emit(cut)
                if chunk:
                    out.append(chunk)
                self._sentence_count = 0
            else:
                # Keep incomplete text buffered until a sentence boundary arrives
                self._sentence_count = sentences
                break
        return out

    def finish(self):
        tail = self._buf.strip()
        self._buf = ""
        self._sentence_count = 0
        return [tail] if tail else []

    @staticmethod
    def _find_boundary(buf, enders):
        best = None
        for e in enders:
            idx = buf.find(e)
            if idx != -1 and (best is None or idx < best):
                best = idx
        return None if best is None else best + 1

    def _emit(self, cut_idx):
        chunk = self._buf[:cut_idx].strip()
        self._buf = self._buf[cut_idx:].lstrip()
        return chunk if len(chunk) >= config.CHUNK_MIN_CHARS else None


class OllamaLLM:
    def __init__(self, history: Optional[List[dict]] = None):
        self.client = ollama.Client(host=config.OLLAMA_HOST)
        self.history = history if history is not None else [
            {"role": "system", "content": config.SYSTEM_PROMPT}
        ]

    def stream_reply(self, user_text: str):
        self.history.append({"role": "user", "content": user_text})
        chunker = SentenceChunker()
        full_reply = []
        stream = self.client.chat(
            model=config.OLLAMA_MODEL,
            messages=self.history,
            stream=True,
            options={
                "num_ctx": config.OLLAMA_NUM_CTX,
                "temperature": config.OLLAMA_TEMPERATURE,
            },
        )
        for part in stream:
            token = part["message"].get("content", "")
            if not token:
                continue
            full_reply.append(token)
            for chunk in chunker.feed(token):
                yield chunk
        for chunk in chunker.finish():
            yield chunk
        self.history.append(
            {"role": "assistant", "content": "".join(full_reply)})
