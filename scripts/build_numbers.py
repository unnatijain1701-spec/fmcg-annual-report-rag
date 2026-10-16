"""Step 2: build the numbers layer from the indexed table nodes.

    python -m scripts.build_numbers                    # extract (resumable) + build DuckDB
    python -m scripts.build_numbers --max-per-report 40 --workers 3
    python -m scripts.build_numbers --db-only          # rebuild DuckDB from existing extractions (seconds)
"""
import argparse

import chunking
import config
import financials
from models import setup_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-report", type=int, default=40)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="only process N tables (quick test)")
    ap.add_argument("--db-only", action="store_true")
    ap.add_argument("--fill-gaps", action="store_true",
                    help="find key metrics a company is missing (e.g. dividend per share) and extract just those tables")
    a = ap.parse_args()
    if a.fill_gaps:
        setup_models()
        gaps = financials.coverage_gaps()
        print("coverage gaps:", gaps or "none")
        tables = chunking.read_jsonl(config.TABLE_NODES)
        for company, metrics in gaps.items():
            chosen = financials.select_gap_tables([n for n in tables if n.metadata["company"] == company], metrics)
            print(f"{company}: {len(chosen)} candidate tables for {', '.join(metrics)}")
            financials.extract_tables(chosen, workers=a.workers)
    elif not a.db_only:
        setup_models()
        tables = chunking.read_jsonl(config.TABLE_NODES)
        chosen = financials.select_tables(tables, max_per_report=a.max_per_report)
        print(f"{len(tables)} tables in the corpus, {len(chosen)} selected as financial statements/ratios/segments")
        financials.extract_tables(chosen, workers=a.workers, limit=a.limit)
    financials.build_db()


if __name__ == "__main__":
    main()