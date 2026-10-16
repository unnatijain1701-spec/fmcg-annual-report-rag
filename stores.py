"""Storage layer: Qdrant (dense vectors), BM25 registry (sparse), table store, report catalogue."""
import json
import threading
from collections import defaultdict
from typing import Callable, Dict, List, Optional

from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

import config
from chunking import read_jsonl

_lock = threading.RLock()
_client: Optional[QdrantClient] = None


# ---- catalogue of indexed reports ----------------------------------------------------------
REPORTS_JSON = config.NODE_DIR / "reports.json"


def load_reports() -> Dict[str, dict]:
    if REPORTS_JSON.exists():
        return json.loads(REPORTS_JSON.read_text(encoding="utf-8"))
    return {}


def save_report(meta: dict, n_nodes: int, n_tables: int):
    with _lock:
        reps = load_reports()
        reps[meta["report_id"]] = {**meta, "n_nodes": n_nodes, "n_tables": n_tables}
        config.NODE_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_JSON.write_text(json.dumps(reps, indent=2, ensure_ascii=False), encoding="utf-8")


def companies() -> List[str]:
    return sorted({r["company"] for r in load_reports().values()})


# ---- Qdrant ---------------------------------------------------------------------------------
def qdrant_client() -> QdrantClient:
    global _client
    with _lock:
        if _client is None:
            if config.QDRANT_URL:
                _client = QdrantClient(url=config.QDRANT_URL)
            else:
                config.QDRANT_PATH.mkdir(parents=True, exist_ok=True)
                _client = QdrantClient(path=str(config.QDRANT_PATH))
        return _client


def set_qdrant_client(client: QdrantClient):   # used by tests (in-memory client)
    global _client
    _client = client


def vector_store() -> QdrantVectorStore:
    client = qdrant_client()
    if not client.collection_exists(config.COLLECTION):
        client.create_collection(config.COLLECTION,
                                 vectors_config=models.VectorParams(size=config.EMBED_DIM, distance=models.Distance.COSINE))
        if config.QDRANT_URL:   # payload indexes only matter (and only exist) on a Qdrant server
            for key in ("company", "fy", "section", "report_id", "node_kind"):
                try:
                    client.create_payload_index(config.COLLECTION, key, models.PayloadSchemaType.KEYWORD)
                except Exception:
                    pass
    return QdrantVectorStore(client=client, collection_name=config.COLLECTION)


def delete_report_vectors(report_id: str):
    client = qdrant_client()
    if client.collection_exists(config.COLLECTION):
        client.delete(config.COLLECTION, points_selector=models.FilterSelector(filter=models.Filter(
            must=[models.FieldCondition(key="report_id", match=models.MatchValue(value=report_id))])))


def embed_and_store(nodes: List[TextNode], embed_model, batch: int = 64, progress: Optional[Callable] = None):
    vs = vector_store()
    for i in range(0, len(nodes), batch):
        chunk = nodes[i:i + batch]
        vecs = embed_model.get_text_embedding_batch([n.get_content(metadata_mode="none") for n in chunk])
        for n, v in zip(chunk, vecs):
            n.embedding = v
        vs.add(chunk)
        for n in chunk:
            n.embedding = None
        if progress:
            progress(f"embedded {min(i + batch, len(nodes))}/{len(nodes)} chunks")


# ---- BM25 registry --------------------------------------------------------------------------
class BM25Registry:
    """One BM25 index over everything plus one per company (so a company scope is exact and fast)."""

    def __init__(self):
        self.global_: Optional[BM25Retriever] = None
        self.by_company: Dict[str, BM25Retriever] = {}

    def rebuild(self, nodes: List[TextNode]):
        by_c = defaultdict(list)
        for n in nodes:
            by_c[n.metadata["company"]].append(n)
        new_by_company = {c: BM25Retriever.from_defaults(nodes=ns, similarity_top_k=config.BM25_TOP_K)
                          for c, ns in by_c.items()}
        new_global = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=config.BM25_TOP_K) if nodes else None
        self.by_company, self.global_ = new_by_company, new_global      # swap after building: readers never see a half-built index

    def get(self, company: Optional[str]) -> Optional[BM25Retriever]:
        return self.by_company.get(company) if company else self.global_


BM25 = BM25Registry()


def rebuild_bm25():
    BM25.rebuild(read_jsonl(config.RETRIEVAL_NODES))


# ---- table store (full tables, looked up after reranking) -----------------------------------
class TableStore:
    def __init__(self):
        self.tables: Dict[str, TextNode] = {}

    def reload(self):
        self.tables = {n.metadata["table_id"]: n for n in read_jsonl(config.TABLE_NODES)}

    def get(self, table_id: str) -> Optional[TextNode]:
        return self.tables.get(table_id)


TABLES = TableStore()
