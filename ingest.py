"""Add a new annual report (PDF or already-parsed markdown) to the live system, without blocking the chat.

Steps: parse (Docling) -> chunk -> embed + index (text search live) -> extract tables -> rebuild DuckDB (numbers live).
One job at a time. Progress is exposed through Job.log so the UI can poll it.
"""
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from llama_index.core import Settings

import chunking
import config
import financials
import parsing
import stores
from corpus import normalize_fy

STEPS = ["parse", "chunk", "index", "numbers"]


class Cancelled(Exception):
    pass


@dataclass
class Job:
    company: str
    fy: str
    source: Path
    status: str = "queued"
    step: str = ""
    log: List[str] = field(default_factory=list)
    error: str = ""
    cancel: bool = False
    started: float = field(default_factory=time.time)

    @property
    def done(self) -> bool:
        return self.status in ("done", "failed", "cancelled")

    def say(self, msg: str):
        if self.cancel:
            raise Cancelled()
        self.log.append(msg)


_current: Optional[Job] = None
_lock = threading.Lock()


def current_job() -> Optional[Job]:
    return _current


def start_job(source: Path, company: str, fy_text: str, parse_fn: Optional[Callable] = None, background: bool = True) -> Job:
    global _current
    with _lock:
        if _current is not None and not _current.done:
            raise RuntimeError(f"An upload is already running ({_current.company} {_current.fy}). Wait for it to finish.")
        label, end, months = normalize_fy(fy_text)
        job = Job(company=company.strip(), fy=label, source=Path(source))
        _current = job
    if background:
        threading.Thread(target=_run, args=(job, end, months, parse_fn), daemon=True).start()
    else:
        _run(job, end, months, parse_fn)
    return job


def _run(job: Job, period_end: int, months: int, parse_fn):
    try:
        meta = {"company": job.company, "fy": job.fy, "period_end": period_end, "months": months,
                "report_id": f"{job.company.replace(' ', '')}_{job.fy}", "source_file": job.source.name}
        # 1. parse
        job.status, job.step = "running", "parse"
        if job.source.suffix.lower() == ".pdf":
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", job.source.stem)
            md_path = config.PROCESSED_DIR / job.company.replace(" ", "_") / safe / f"{safe}.md"
            if parse_fn is None:
                parse_fn = parsing.pdf_to_markdown
            parse_fn(job.source, md_path, job.say)
        else:
            md_path = job.source
        md = md_path.read_text(encoding="utf-8", errors="ignore")
        # 2. chunk
        job.step = "chunk"
        job.say("Splitting into chunks (tables + narrative)...")
        parsers, resplitter = chunking.build_parsers(Settings.embed_model)
        retrieval_nodes, tables = chunking.chunk_report(md, meta, parsers, resplitter, progress=job.say)
        job.say(f"{len(retrieval_nodes)} chunks, {len(tables)} tables")
        # 3. index (text search becomes live here)
        job.step = "index"
        stores.delete_report_vectors(meta["report_id"])
        chunking.drop_report(config.RETRIEVAL_NODES, meta["report_id"])
        chunking.drop_report(config.TABLE_NODES, meta["report_id"])
        stores.embed_and_store(retrieval_nodes, Settings.embed_model, progress=job.say)
        chunking.append_jsonl(config.RETRIEVAL_NODES, retrieval_nodes)
        chunking.append_jsonl(config.TABLE_NODES, tables)
        stores.save_report(meta, len(retrieval_nodes), len(tables))
        stores.rebuild_bm25()
        stores.TABLES.reload()
        job.say("Text search is ready for this report.")
        # 4. numbers
        job.step = "numbers"
        chosen = financials.select_tables(tables, max_per_report=40)
        job.say(f"Extracting figures from {len(chosen)} financial tables...")
        ok, failed = financials.extract_tables(chosen, workers=3, progress=job.say)
        financials.build_db(progress=job.say)
        job.say(f"Financial figures ready ({ok} tables read, {failed} failed).")
        job.status = "done"
    except Cancelled:
        job.status = "cancelled"
    except Exception as e:           # surface the error to the UI instead of dying silently in a thread
        job.status, job.error = "failed", f"{type(e).__name__}: {e}"