"""Step 1: parsed markdown -> chunks -> embeddings (Qdrant) + node files (BM25, table store).

    python -m scripts.build_corpus                 # all .md under data/processed (resumes; skips finished reports)
    python -m scripts.build_corpus --only ITC      # one company
    python -m scripts.build_corpus --rebuild       # wipe and start over
Stop the Chainlit app first when using embedded Qdrant (one process may open the database at a time).
"""
import argparse
import shutil
import time

import config
import chunking
import stores
from corpus import infer_meta
from models import setup_models


def process_report(md_path, meta, embed_model, parsers, resplitter, log=print):
    t0 = time.time()
    md = md_path.read_text(encoding="utf-8", errors="ignore")
    retrieval, tables = chunking.chunk_report(md, meta, parsers, resplitter, progress=log)
    log(f"{meta['report_id']}: {len(retrieval)} chunks ({len(tables)} tables) in {time.time() - t0:.0f}s, embedding...")
    # replace any earlier copy of this report, then store
    stores.delete_report_vectors(meta["report_id"])
    chunking.drop_report(config.RETRIEVAL_NODES, meta["report_id"])
    chunking.drop_report(config.TABLE_NODES, meta["report_id"])
    stores.embed_and_store(retrieval, embed_model, progress=log)
    chunking.append_jsonl(config.RETRIEVAL_NODES, retrieval)
    chunking.append_jsonl(config.TABLE_NODES, tables)
    stores.save_report(meta, len(retrieval), len(tables))
    log(f"{meta['report_id']}: done in {time.time() - t0:.0f}s")
    return retrieval, tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="company name filter, e.g. ITC")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    if a.rebuild:
        for p in (config.QDRANT_PATH, config.NODE_DIR):
            shutil.rmtree(p, ignore_errors=True)
        config.NODE_DIR.mkdir(parents=True, exist_ok=True)
    llm, embed = setup_models()
    parsers, resplitter = chunking.build_parsers(embed)
    done = set(stores.load_reports())
    files = sorted(config.PROCESSED_DIR.rglob("*.md"))
    print(f"{len(files)} markdown files found, {len(done)} already indexed")
    for f in files:
        meta = infer_meta(f)
        if a.only and a.only.lower() not in meta["company"].lower():
            continue
        if meta["report_id"] in done:
            print(f"skip {meta['report_id']} (already indexed)")
            continue
        process_report(f, meta, embed, parsers, resplitter)
    stores.rebuild_bm25()
    print("Index complete:", len(stores.load_reports()), "reports")


if __name__ == "__main__":
    main()
