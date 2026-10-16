"""Offline stand-ins for Ollama models, used by the test-suite (no GPU / no network needed)."""
import hashlib
import re
from typing import Any, List

import numpy as np
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.types import CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms import CustomLLM
from llama_index.core.llms.callbacks import llm_completion_callback


class HashEmbedding(BaseEmbedding):
    """Bag-of-words hashing embedding: similar words -> similar vectors, so retrieval tests are meaningful."""
    dim: int = 256

    def _vec(self, text: str) -> List[float]:
        v = np.zeros(self.dim, dtype=np.float32)
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def _get_query_embedding(self, query: str): return self._vec(query)
    async def _aget_query_embedding(self, query: str): return self._vec(query)
    def _get_text_embedding(self, text: str): return self._vec(text)


class ScriptedLLM(CustomLLM):
    """LLM that answers from a list of (substring, reply) rules; first rule whose substring is in the prompt wins."""
    rules: List[Any] = []
    default: str = "OK"
    calls: List[str] = []

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(context_window=16000, num_output=512, model_name="scripted")

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kw) -> CompletionResponse:
        self.calls.append(prompt)
        for sub, reply in self.rules:
            if sub in prompt:
                return CompletionResponse(text=reply(prompt) if callable(reply) else reply)
        return CompletionResponse(text=self.default)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kw) -> CompletionResponseGen:
        yield self.complete(prompt)
