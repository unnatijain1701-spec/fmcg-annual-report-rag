"""Ollama model wiring. Settings.llm / Settings.embed_model are set once; SQL model is called directly."""
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

import time

import config
import timing


class TimedOllama(Ollama):
    """Ollama LLM that prints the duration of every call (see timing.py)."""

    def chat(self, messages, **kwargs):
        t = time.time()
        r = super().chat(messages, **kwargs)
        timing.log("LLM chat", time.time() - t, f"prompt {sum(len(m.content or '') for m in messages)} chars, "
                                                  f"answer {len(r.message.content or '')} chars")
        return r

    def complete(self, prompt, formatted=False, **kwargs):
        t = time.time()
        r = super().complete(prompt, formatted=formatted, **kwargs)
        timing.log("LLM complete", time.time() - t, f"prompt {len(prompt)} chars, answer {len(r.text or '')} chars")
        return r


def make_llm():
    # Qwen3 and Nemotron reason before answering, which is slow and fills the answer with a trace: switch that off.
    thinking = False if any(k in config.LLM_MODEL.lower() for k in ("qwen3", "nemotron")) else None
    return TimedOllama(model=config.LLM_MODEL, base_url=config.LLM_URL, temperature=0.0,
                       request_timeout=600.0, context_window=config.LLM_CONTEXT, thinking=thinking,
                       additional_kwargs={"num_ctx": config.LLM_CONTEXT})


def make_embedder():
    # bge-m3 supports 8k tokens; make sure Ollama does not truncate at its 2k default
    return OllamaEmbedding(model_name=config.EMBED_MODEL, base_url=config.OLLAMA_URL, embed_batch_size=32,
                           ollama_additional_kwargs={"num_ctx": 8192}, keep_alive="30m")


def setup_models():
    Settings.llm = make_llm()
    Settings.embed_model = make_embedder()
    return Settings.llm, Settings.embed_model