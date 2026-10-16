"""Where does the right passage get lost? Run:  PYTHONPATH=. python probe.py
For each question: search (before rerank) -> rerank -> what the LLM would see. No LLM answer is generated."""
import time
from collections import Counter

import config, models, retrieval

TESTS = [
    ("What are the key risks ITC reports in its latest annual report?",
     ["tax", "regulat", "raw material", "competition", "price"]),
    ("What are the key risks ITC reports in its FY2025-26 annual report?",
     ["tax", "regulat", "raw material", "competition", "price"]),
    ("What reasons does ITC's FY2024-25 annual report give for changes in its EBITDA margin?",
     ["margin", "ebitda", "cost", "price", "mix"]),
]


def head(n):
    t = n.node.get_content(metadata_mode="none")
    return t.split("\n", 1)[0][:60]


def hits(nodes, terms):
    out = 0
    for n in nodes:
        t = n.node.get_content(metadata_mode="none").lower()
        out += any(w in t for w in terms)
    return out


llm, embed = models.setup_models()
fusion = retrieval.build_fusion(embed, llm)
rr = retrieval.get_reranker()
rr.top_n = 30                                  # keep the full reranked order so we can see where things land

for q, terms in TESTS:
    print("\n" + "=" * 100 + "\nQ:", q)
    print("scope:", retrieval.parse_scope(q))
    t0 = time.time()
    nodes = fusion.retrieve(q)
    t1 = time.time()
    ranked = rr.postprocess_nodes(nodes, query_str=q)
    t2 = time.time()
    print(f"search {t1 - t0:.1f}s ({len(nodes)} passages) | rerank {t2 - t1:.1f}s")
    print("before rerank, by (year, section):",
          dict(Counter((n.node.metadata.get("fy"), n.node.metadata.get("section")) for n in nodes)))
    print(f"passages containing any of {terms}:  before rerank {hits(nodes, terms)}/{len(nodes)} | "
          f"top 5 after rerank {hits(ranked[:5], terms)}/5 | top 10 {hits(ranked[:10], terms)}/10")
    print("top 10 after rerank:")
    for i, n in enumerate(ranked[:10]):
        print(f"  {i + 1:2d}. score {n.score:.3f}  {head(n)}")