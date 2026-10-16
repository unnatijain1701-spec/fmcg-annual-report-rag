"""Pre-flight check: is every model and file in place?   python -m scripts.check_env"""
import sys
import time

import config


def check(name, fn):
    t = time.time()
    try:
        detail = fn()
        print(f"  PASS  {name:34} {detail or ''} ({time.time() - t:.1f}s)")
        return True
    except Exception as e:
        print(f"  FAIL  {name:34} {type(e).__name__}: {str(e)[:160]}")
        return False


def main():
    import ollama
    from models import make_embedder, make_llm
    client = ollama.Client(host=config.OLLAMA_URL, timeout=120)
    ok = []
    ok.append(check("Ollama server reachable", lambda: f"{len(client.list()['models'])} local models"))
    ok.append(check(f"embeddings: {config.EMBED_MODEL}", lambda: f"dim={len(make_embedder().get_text_embedding('revenue from operations'))}"))
    ok.append(check(f"answer LLM: {config.LLM_MODEL}", lambda: make_llm().complete("Reply with the single word OK.").text.strip()[:40]))
    ok.append(check(f"SQL model: {config.SQL_MODEL}", lambda: client.chat(model=config.SQL_MODEL, messages=[{"role": "user", "content": "SELECT 1; reply OK"}])["message"]["content"][:30]))
    if config.RERANK_MODEL.lower() not in ("none", ""):
        def rr():
            from llama_index.core.postprocessor import SentenceTransformerRerank
            m = SentenceTransformerRerank(model=config.RERANK_MODEL, top_n=2)
            return "loaded"
        ok.append(check(f"reranker: {config.RERANK_MODEL}", rr))
    ok.append(check("Docling installed", lambda: __import__("docling") and "ok"))
    ok.append(check("parsed markdown files", lambda: f"{len(list(config.PROCESSED_DIR.rglob('*.md')))} found in data/processed"))
    print("\nAll good." if all(ok) else "\nFix the FAIL lines above before building.")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
