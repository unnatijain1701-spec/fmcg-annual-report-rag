import traceback, time
import config, stores
from models import setup_models

print("OLLAMA_URL:", config.OLLAMA_URL, "| QDRANT_URL:", repr(config.QDRANT_URL), "| path exists:", config.QDRANT_PATH.exists())
llm, embed = setup_models()
c = stores.qdrant_client()
print("1) points in collection:", c.count(config.COLLECTION).count)
t = time.time()
v = embed.get_text_embedding("ITC key risks climate cyber security")
print("2) embedding OK, dimension", len(v), f"({time.time()-t:.1f}s)")
for p in c.query_points(config.COLLECTION, query=v, limit=3).points:
    print("   hit score", round(p.score, 3), "payload keys:", list(p.payload.keys())[:6])
print("3) loading reranker ...")
import retrieval
try:
    rr = retrieval.get_reranker()
    print("   reranker OK:", type(rr).__name__)
except Exception:
    traceback.print_exc()
print("4) full text engine ...")
try:
    eng = retrieval.build_text_engine(embed, llm)
    r = eng.query("What are the key risks ITC reports in its latest annual report?")
    print("   sources used:", len(r.source_nodes))
    print(str(r)[:600])
except Exception:
    traceback.print_exc()