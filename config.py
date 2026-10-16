"""Central configuration. Everything tunable lives here (or in a .env file)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Chainlit removes the app folder from sys.path after loading app.py, which breaks imports done later (first chat).
# A second, permanent entry keeps `import models` / `import stores` / `import parsing` working at any time.
sys.path.append(str(ROOT))
try:  # optional .env support
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

def env(key, default):
    return os.environ.get(key, default)

# ---- paths -----------------------------------------------------------------
PROCESSED_DIR = ROOT / "data" / "processed"      # Docling markdown (one .md per report)
UPLOAD_DIR = ROOT / "data" / "uploads"           # PDFs uploaded through the chat UI
NODE_DIR = ROOT / "data" / "nodes"
RETRIEVAL_NODES = NODE_DIR / "retrieval_nodes.jsonl"   # narrative chunks + table summaries (embedded + BM25)
TABLE_NODES = NODE_DIR / "table_nodes.jsonl"           # full table markdown, looked up after reranking
FIN_DIR = ROOT / "data" / "financials"
EXTRACTIONS = FIN_DIR / "extractions.jsonl"
DUCKDB_PATH = FIN_DIR / "financials.duckdb"
QDRANT_PATH = ROOT / "data" / "qdrant"           # embedded Qdrant (no Docker needed)
QDRANT_URL = env("QDRANT_URL", "")               # set to http://localhost:6333 to use a Qdrant server instead
COLLECTION = env("QDRANT_COLLECTION", "fmcg_reports")

# ---- models ----------------------------------------------------------------
OLLAMA_URL = env("OLLAMA_URL", "http://localhost:11434")     # embeddings (bge-m3) and text-to-SQL
LLM_URL = env("LLM_URL", OLLAMA_URL)                         # answering model; set to this PC's own Ollama for cloud models
EMBED_MODEL = env("EMBED_MODEL", "bge-m3")
EMBED_DIM = 1024
LLM_MODEL = env("LLM_MODEL", "gemma4:31b-cloud")        # answers, sub-question routing, faithfulness, revision
SQL_MODEL = env("SQL_MODEL", "qwen2.5-coder:7b")         # text-to-SQL (called directly via ollama)
EXTRACT_MODEL = env("EXTRACT_MODEL", LLM_MODEL)          # table -> structured facts (numbers layer build)
EXTRACT_URL = env("EXTRACT_URL", LLM_URL)
RERANK_MODEL = env("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
LLM_CONTEXT = int(env("LLM_CONTEXT", "16384"))

# ---- chunking --------------------------------------------------------------
# Sections whose narrative text is split with SemanticSplitterNodeParser (costs one embedding per sentence).
SEMANTIC_SECTIONS = set(env("SEMANTIC_SECTIONS", "mdna,risk").split(","))
SENTENCE_CHUNK_TOKENS = 512
SENTENCE_CHUNK_OVERLAP = 64
SEMANTIC_BREAKPOINT_PERCENTILE = 95
SEMANTIC_BUFFER_SIZE = 1
MAX_CHUNK_CHARS = 4500          # semantic chunks above this are re-split
MIN_CHUNK_CHARS = 120           # drop junk chunks (page numbers, stray headings)
TABLE_SUMMARY_MODE = env("TABLE_SUMMARY_MODE", "rules")   # "rules" (fast, no LLM) or "llm" (one LLM call per table)

# ---- retrieval -------------------------------------------------------------
DENSE_TOP_K = 30
BM25_TOP_K = 30
FUSION_TOP_K = 30
RERANK_TOP_N = int(env("RERANK_TOP_N", "10"))   # passages given to the LLM after reranking
RRF_WEIGHTS = (0.5, 0.5)         # dense, sparse
MAX_TABLE_CHARS = 6000           # full-table text handed to the LLM is capped to this

# ---- verification / memory -------------------------------------------------
VERIFY = env("VERIFY", "1") == "1"
VERIFY_MAX_CONTEXTS = 10
VERIFY_CONTEXT_CHARS = 2500
MEMORY_TOKEN_LIMIT = 3000