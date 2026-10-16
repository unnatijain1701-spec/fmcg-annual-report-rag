"""Numbers layer: financial tables -> structured facts (JSONL) -> DuckDB -> text-to-SQL tool.

    extract_tables()  picks statement/ratio/segment tables from the table nodes and asks an LLM for validated JSON
    build_db()        loads the JSONL into DuckDB: `facts` (all values) and `facts_latest` (newest report wins)
    SQLEngine         question -> DuckDB SQL (qwen2.5-coder) -> result table -> LlamaIndex Response
"""
import json
import os
import re
import threading
import time

import timing
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import duckdb
import ollama
from llama_index.core.base.response.schema import Response
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.schema import NodeWithScore, TextNode
from pydantic import BaseModel, field_validator

import config
from chunking import read_jsonl
from corpus import DEFAULT_UNITS, normalize_fy
import stores

METRICS = {
    # money (reported in the table's unit)
    "revenue_from_operations": "Revenue from operations / net sales / turnover (net of GST where reported)",
    "total_income": "Total income", "other_income": "Other income",
    "total_expenses": "Total expenses", "cost_of_materials": "Cost of materials consumed / raw material cost",
    "employee_cost": "Employee benefits expense", "finance_cost": "Finance costs", "depreciation": "Depreciation and amortisation",
    "ebitda": "EBITDA / operating profit before depreciation", "ebit": "EBIT / PBIT",
    "profit_before_tax": "Profit before tax (before exceptional items if the table says so, else PBT)",
    "tax_expense": "Total tax expense", "net_profit": "Profit for the year / net profit / PAT (attributable to owners if split)",
    "total_assets": "Total assets", "total_equity": "Total equity / net worth", "total_borrowings": "Borrowings (total debt)",
    "cash_and_equivalents": "Cash and cash equivalents", "inventories": "Inventories", "trade_receivables": "Trade receivables",
    "trade_payables": "Trade payables", "cash_from_operations": "Net cash from operating activities",
    "cash_from_investing": "Net cash from investing activities", "cash_from_financing": "Net cash from financing activities",
    "capex": "Capital expenditure / purchase of PPE", "dividend_total": "Total dividend paid/proposed",
    # per share
    "eps_basic": "Basic earnings per share", "eps_diluted": "Diluted earnings per share",
    "dividend_per_share": "Dividend per share", "book_value_per_share": "Book value per share",
    # percent (only when the table itself reports the percentage)
    "ebitda_margin_pct": "EBITDA margin %", "net_profit_margin_pct": "Net profit margin / PAT %", "roce_pct": "ROCE %",
    "roe_pct": "Return on net worth / equity %", "gross_margin_pct": "Gross margin %", "volume_growth_pct": "Volume growth %",
    "revenue_growth_pct": "Revenue / sales growth %",
}
PER_SHARE = {"eps_basic", "eps_diluted", "dividend_per_share", "book_value_per_share"}
PERCENT = {m for m in METRICS if m.endswith("_pct")}
CRORE_FACTOR = {"crore": 1.0, "million": 0.1, "lakh": 0.01, "billion": 100.0, "thousand": 1e-4}

# ---- which tables to extract ----------------------------------------------------------------
# A table is a financial statement/ratio/segment table when several of its ROW LABELS (first column) name real metrics.
# (Matching anywhere in the text also picked up contents pages, director tables and BRSR tables.)
_LABEL_PATTERNS = [re.compile(p, re.I) for p in [
    r"revenue from operations|net sales|turnover|gross revenue|income from operations|sales value",
    r"total income", r"other income", r"profit before (exceptional|tax)|\bpbt\b",
    r"tax expense|income tax|current tax", r"profit (for the (year|period)|after tax)|\bpat\b|net profit",
    r"total assets", r"total equity|net worth|shareholders.? (funds|equity)|total liabilities",
    r"borrowings", r"cash and cash equivalents", r"inventories", r"trade receivables", r"trade payables",
    r"net cash (generated|from|used|flow)|cash flows? from", r"earnings per share|\beps\b",
    r"dividend", r"ebitda|operating profit|\bebit\b", r"depreciation", r"employee (benefits?|cost)",
    r"finance costs?|interest expense", r"return on (capital|net worth|equity|assets)|\broce\b|\bronw\b",
    r"cost of materials|materials consumed|cost of goods", r"segment (result|revenue|assets|liabilities)",
    r"capital expenditure|purchase of (property|fixed)", r"gross margin|ebitda margin|volume growth"]]
EXCLUDE_SECTIONS = {"brsr", "governance", "notice", "auditor"}


def row_labels(text: str):
    out = []
    for line in text.split("\n"):
        if line.startswith("|"):
            first = line.strip().strip("|").split("|")[0].strip()
            if first and not set(first) <= set("-: "):
                out.append(first)
    return out


def numeric_density(text: str) -> float:
    cells = [c.strip() for line in text.split("\n") if line.startswith("|") for c in line.strip("|").split("|")]
    cells = [c for c in cells if c and not set(c) <= set("-: ")]
    if not cells:
        return 0.0
    num = sum(bool(re.fullmatch(r"[\(\-–]?[\d,]+(\.\d+)?\)?%?", c.replace(" ", ""))) for c in cells)
    return num / len(cells)


def table_score(text: str) -> int:
    labels = row_labels(text)
    return sum(any(p.search(l) for l in labels) for p in _LABEL_PATTERNS)


def select_tables(table_nodes, max_per_report: int = 40, min_score: int = 3):
    by_report: Dict[str, list] = {}
    for n in table_nodes:
        if n.metadata.get("section") in EXCLUDE_SECTIONS:
            continue
        t = n.text[:6000]
        if n.text.count("\n|") < 5 or numeric_density(t) < 0.3:
            continue
        s = table_score(t)
        if s >= min_score:
            by_report.setdefault(n.metadata["report_id"], []).append((s, len(t), n))
    chosen = []
    for rid, items in by_report.items():
        items.sort(key=lambda x: (-x[0], -x[1]))
        chosen += [n for _, _, n in items[:max_per_report]]
    return chosen


# ---- gap filling: key metrics missing for a company ------------------------------------------------
GAP_PATTERNS = {
    "dividend_per_share": re.compile(r"dividends? per (equity )?share|dividend \(.{0,30}per share|\bdps\b", re.I),
    "eps_basic": re.compile(r"basic (earnings|eps)|earnings per (equity )?share|\beps\b", re.I),
    "ebitda": re.compile(r"\bebitda\b|operating profit before", re.I),
    "total_assets": re.compile(r"total assets", re.I),
    "total_equity": re.compile(r"total equity|net worth|shareholders.? (funds|equity)", re.I),
    "profit_before_tax": re.compile(r"profit before (exceptional|tax)|\bpbt\b", re.I),
}


def coverage_gaps():
    """{company: [metrics]} where the company-total series covers fewer fiscal years than it has indexed reports."""
    reports = stores.load_reports()
    n_reports = {}
    for r in reports.values():
        n_reports[r["company"]] = n_reports.get(r["company"], 0) + 1
    gaps = {}
    if not config.DUCKDB_PATH.exists():
        return gaps
    con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
    try:
        for company, n in n_reports.items():
            have = dict(con.execute("SELECT metric, count(DISTINCT period) FROM facts_latest WHERE company=? AND segment='' "
                                    "AND default_basis GROUP BY 1", [company]).fetchall())
            miss = [m for m in GAP_PATTERNS if have.get(m, 0) < n]
            if miss:
                gaps[company] = miss
    finally:
        con.close()
    return gaps


def select_gap_tables(table_nodes, metrics, per_report: int = 4):
    """Small tables are skipped by select_tables (they rarely hold three metrics). For a missing metric, pick the tables
    whose ROW LABELS name it, a few per report, preferring numeric and compact tables."""
    by_report: Dict[str, list] = {}
    for n in table_nodes:
        if n.metadata.get("section") in EXCLUDE_SECTIONS or n.text.count("\n|") < 3:
            continue
        t = n.text[:6000]
        dens = numeric_density(t)
        if dens < 0.2:
            continue
        labels = row_labels(t)
        for m in metrics:
            pat = GAP_PATTERNS[m]
            if any(pat.search(l) and len(l) <= 60 for l in labels):
                by_report.setdefault((n.metadata["report_id"], m), []).append((dens, len(t), n))
    chosen, seen = [], set()
    for key, items in by_report.items():
        items.sort(key=lambda x: (-x[0], x[1]))
        for _, _, n in items[:per_report]:
            if n.node_id not in seen:
                seen.add(n.node_id)
                chosen.append(n)
    return chosen


# ---- standalone / consolidated: read it from the report text before the table -------------------
_BASIS_LINE = re.compile(r"\b(consolidated|standalone)\b", re.I)
_STATEMENT_WORDS = re.compile(r"financial statements?|balance sheet|profit and loss|statement of|cash flows?|"
                              r"notes? (to|forming)|segment|results|highlights|financial (performance|summary)|ratios?", re.I)


def _table_sig(lines):
    """Signature of a markdown table: its first data cells (works for the raw Docling table and our rebuilt copy)."""
    rows = []
    for l in lines:
        if not l.startswith("|"):
            continue
        cells = [c.strip() for c in l.strip().strip("|").split("|")]
        if cells and all(set(c) <= set("-: ") for c in cells):
            continue
        rows.append(cells)
    sig = "".join("".join(r) for r in rows[1:5])
    return re.sub(r"\s+", "", sig).lower()[:300]


def basis_hints(nodes) -> Dict[str, str]:
    """table_id -> 'consolidated' | 'standalone' for tables whose position in the source markdown follows a heading or
    caption naming one of them (reports repeat such running headers on every page). Missing = no hint."""
    from collections import defaultdict
    by_report = defaultdict(list)
    for n in nodes:
        by_report[n.metadata["report_id"]].append(n)
    hints: Dict[str, str] = {}
    for rid, ns in by_report.items():
        src = ns[0].metadata.get("source_file", "")
        found = next(iter(config.PROCESSED_DIR.rglob(src)), None) if src else None
        if not found:
            continue
        lines = found.read_text(encoding="utf-8", errors="ignore").replace("&amp;", "&").split("\n")
        state, i, blocks = None, 0, {}
        while i < len(lines):
            l = lines[i]
            if l.startswith("|"):
                j = i
                while j < len(lines) and lines[j].startswith("|"):
                    j += 1
                blocks.setdefault(_table_sig(lines[i:j]), []).append(state)
                i = j
                continue
            if len(l) <= 220 and _BASIS_LINE.search(l) and _STATEMENT_WORDS.search(l):
                m = _BASIS_LINE.search(l)
                state = m.group(1).lower()
            i += 1
        for n in ns:
            sig = _table_sig([x for x in n.text.split("\n") if x.startswith("|")])
            states = blocks.get(sig)
            if states and len(set(states)) == 1 and states[0]:
                hints[n.metadata["table_id"]] = states[0]
    return hints


# ---- extraction -----------------------------------------------------------------------------
class Value(BaseModel):
    period: str
    value: float

    @field_validator("value", mode="before")
    @classmethod
    def _num(cls, v):
        if isinstance(v, str):
            s = v.strip().replace(",", "").replace("₹", "")
            neg = s.startswith("(") and s.endswith(")")
            s = s.strip("()%")
            v = float(s) * (-1 if neg else 1)
        return v


class Row(BaseModel):
    metric: str
    segment: str = ""
    label: str = ""
    values: List[Value]


class Extraction(BaseModel):
    basis: str = "unknown"
    unit: str = "unknown"
    rows: List[Row]


JSON_SCHEMA = Extraction.model_json_schema()
SYSTEM = ("You extract financial figures from one table of an Indian company annual report into JSON. "
          "Output only what the table states; never estimate or compute. Ignore rows that match no metric.")


def extract_prompt(table_text: str, meta: dict, hint: str = "") -> str:
    hint_line = (f"Context: in the report text this table comes after a heading about the {hint.upper()} financial statements, "
                 f"so use basis=\"{hint}\" unless the table itself says otherwise.\n") if hint else ""
    metric_lines = "\n".join(f"- {k}: {v}" for k, v in METRICS.items())
    return f"""Company: {meta['company']}. Report: {meta['fy']}.
{hint_line}Allowed metrics (use these exact keys; skip other rows):
{metric_lines}

Fields:
- basis: "standalone" or "consolidated" if the table or its heading says so, else "unknown".
- unit: the unit of the monetary figures: "crore", "million", "lakh", "billion" or "unknown". Look for "Rs. in crore", "₹ crore", "₹ million" etc.
- rows: one entry per matching table row. metric = key above. segment = "" for company totals; for segment tables put the
  segment name (e.g. "FMCG - Cigarettes"). label = the row label as printed.
- values: one entry per column period, period written like "FY2024-25" (year ended 31 March 2025) or "CY2022" (calendar year). Use plain numbers;
  brackets mean negative. Percent rows keep the percent number (12.5, not 0.125). Skip columns that are not a financial year.

TABLE:
{table_text}
"""


def detect_unit(text: str, company: str):
    t = text.lower()
    for key, rx in (("crore", r"crores?\b"), ("million", r"\bmillions?\b|\bmn\b"), ("lakh", r"lakhs?\b|\blacs?\b"), ("billion", r"billions?\b")):
        if re.search(rx, t):
            return key, False
    return DEFAULT_UNITS.get(company, "unknown"), True


_write_lock = threading.Lock()


def _to_float(v):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        t = v.strip().replace(",", "").replace("₹", "").replace("%", "")
        neg = t.startswith("(") and t.endswith(")")
        t = t.strip("()").strip()
        try:
            return float(t) * (-1 if neg else 1)
        except ValueError:
            return None
    return None


def _clean_extraction(d: dict):
    """Normalise the shapes models actually return (dict-style values, 'Rs crore', null cells)."""
    basis = str(d.get("basis", "unknown")).lower()
    basis = "consolidated" if "consol" in basis else "standalone" if "stand" in basis else "unknown"
    unit = str(d.get("unit", "unknown")).lower()
    unit = next((u for u in CRORE_FACTOR if u in unit), "unknown")
    rows = []
    for r in d.get("rows") or []:
        if not isinstance(r, dict) or r.get("metric") not in METRICS:
            continue
        vals = r.get("values") or []
        if isinstance(vals, dict):
            vals = [{"period": k, "value": v} for k, v in vals.items()]
        good = []
        for v in vals:
            if isinstance(v, dict) and v.get("period") is not None and _to_float(v.get("value")) is not None:
                good.append({"period": str(v["period"]), "value": _to_float(v["value"])})
        if good:
            rows.append({"metric": r["metric"], "segment": str(r.get("segment") or ""),
                         "label": str(r.get("label") or ""), "values": good})
    return Extraction(basis=basis, unit=unit, rows=rows)


def parse_extractions(content: str) -> List[Extraction]:
    """Accepts bare JSON, ```json fenced JSON, an object or a list of objects, with trailing text."""
    import json as _json
    text = (content or "").strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text).strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON found in model reply")
    obj, _ = _json.JSONDecoder().raw_decode(text[min(starts):])
    items = obj if isinstance(obj, list) else [obj]
    out = [_clean_extraction(i) for i in items if isinstance(i, dict) and ("rows" in i or "basis" in i)]
    if not out:
        raise ValueError("JSON had no 'rows'")
    return out


def _call_llm(prompt: str, retries: int = 3) -> List[Extraction]:
    client = ollama.Client(host=config.EXTRACT_URL, timeout=600)
    last = None
    for attempt in range(retries):
        try:
            r = client.chat(model=config.EXTRACT_MODEL, format=JSON_SCHEMA, options={"temperature": 0, "num_ctx": 16384},
                            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
            return parse_extractions(r["message"]["content"])
        except Exception as e:     # bad JSON, timeouts, rate limits: back off and retry
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"extraction failed: {last}")


def extract_tables(table_nodes, workers: int = 3, limit: Optional[int] = None, progress=print, llm_fn=_call_llm):
    """Resumable: tables already present in extractions.jsonl are skipped; failed ones are retried next run."""
    config.FIN_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if config.EXTRACTIONS.exists():
        done = {json.loads(l)["table_id"] for l in open(config.EXTRACTIONS, encoding="utf-8") if l.strip()}
    todo = [n for n in table_nodes if n.metadata["table_id"] not in done]
    if limit:
        todo = todo[:limit]
    progress(f"extracting {len(todo)} tables ({len(done)} already done)")
    ok = failed = 0
    hints = basis_hints(todo)
    progress(f"basis (standalone/consolidated) found from the report text for {len(hints)} of {len(todo)} tables")

    def work(n):
        meta = n.metadata
        text = n.text[:7000]
        hint = hints.get(meta["table_id"], "")
        res = llm_fn(extract_prompt(text, meta, hint))
        exts = res if isinstance(res, list) else [res]
        unit_found, inferred0 = detect_unit(text, meta["company"])
        recs = []
        for ext in exts:
            known = ext.unit in CRORE_FACTOR
            basis = ext.basis if ext.basis != "unknown" else (hint or "unknown")
            recs.append({"table_id": meta["table_id"], "report_id": meta["report_id"], "company": meta["company"],
                         "report_fy": meta["fy"], "section": meta.get("section", ""), "basis": basis,
                         "unit": ext.unit if known else unit_found, "unit_inferred": False if known else inferred0,
                         "model": config.EXTRACT_MODEL, "rows": [r.model_dump() for r in ext.rows]})
        return recs

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, n): n for n in todo}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                recs = f.result()
                with _write_lock, open(config.EXTRACTIONS, "a", encoding="utf-8") as out:
                    for rec in recs:
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                ok += 1
            except Exception as e:
                failed += 1
                progress(f"  failed {futs[f].metadata['table_id'][:8]}: {e}")
            if i % 10 == 0:
                progress(f"  {i}/{len(todo)} tables processed")
    progress(f"extraction done: ok={ok}, failed={failed}")
    return ok, failed


# ---- DuckDB ---------------------------------------------------------------------------------
def _to_cr(value, unit, metric):
    if metric in PER_SHARE or metric in PERCENT:
        return None
    f = CRORE_FACTOR.get(unit)
    return None if f is None else value * f


def _pick_latest(df):
    """One value per (company, basis, metric, segment, period).
    Newest report wins (restatements). Inside that report, a company TOTAL is the largest money figure (sub-tables such as
    subsidiaries, related parties or components print smaller numbers under the same row label); for ratios and per-share
    figures the value that appears most often wins. Ties fall back to the table id so the result never changes between builds."""
    import pandas as pd
    key = ["company", "basis", "metric", "segment", "period"]
    d = df.copy()
    d["value_cr"] = pd.to_numeric(d["value_cr"], errors="coerce")
    d["_agree"] = d.groupby(key + ["report_end", "value"])["value"].transform("size")
    d["_size"] = d["value_cr"].abs().where(d["value_cr"].notna(), d["_agree"].astype(float))
    d = d.sort_values(["report_end", "_size", "table_id"], ascending=[False, False, True])
    return d.drop_duplicates(key, keep="first").drop(columns=["_agree", "_size"])


def _validate_bases(df):
    """A company counts as having consolidated accounts only if its consolidated revenue is not far below its other
    revenue figures (Nestle India publishes standalone accounts only; stray tables must not create a 'consolidated' basis).
    Companies without valid consolidated data: drop consolidated rows and read 'unknown' as standalone."""
    if df.empty:
        return df, []
    rev = df[(df.metric == "revenue_from_operations") & (df.segment == "") & df.value_cr.notna()]
    standalone_only = []
    for company, g in rev.groupby("company"):
        ratios = []
        for period, p in g.groupby("period"):
            c = p[p.basis == "consolidated"].value_cr.max()
            n = p[p.basis != "consolidated"].value_cr.max()
            if c == c and n == n and n > 0:
                ratios.append(c / n)
        if ratios and sorted(ratios)[len(ratios) // 2] < 0.8:
            standalone_only.append(company)
    if standalone_only:
        mask = df.company.isin(standalone_only)
        df = df[~(mask & (df.basis == "consolidated"))].copy()
        df.loc[df.company.isin(standalone_only) & (df.basis == "unknown"), "basis"] = "standalone"
    return df, standalone_only


def _add_default_basis(d, group):
    """default_basis = TRUE on the best basis available for each (company, metric, segment, period[, ...]):
    consolidated if the company has it for that figure, else standalone, else unknown. The SQL model filters on this
    instead of hard-coding a basis, so standalone-only companies and basis-less metrics (dividend per share) just work."""
    rank = {"consolidated": 0, "standalone": 1, "unknown": 2}
    d = d.copy()
    d["_r"] = d["basis"].map(rank).fillna(3)
    d["default_basis"] = d["_r"] == d.groupby(group)["_r"].transform("min")
    return d.drop(columns="_r")


def _growth_pairs(latest):
    """Every (start year, end year) pair per money metric with growth % and CAGR %, so questions like 'highest revenue CAGR
    from FY2022-23 to FY2025-26' become a plain SELECT instead of a hand-written calculation. Partial periods are skipped."""
    import pandas as pd
    cols = ["company", "basis", "metric", "segment", "start_period", "end_period", "years", "start_value_cr", "end_value_cr",
            "change_pct", "cagr_pct", "default_basis"]
    m = latest[latest.value_cr.notna() & ~latest.partial_period.astype(bool) & ~latest.metric.str.endswith("_pct")]
    if m.empty:
        return pd.DataFrame(columns=cols)
    a = m[["company", "basis", "metric", "segment", "period", "period_end", "value_cr"]]
    j = a.merge(a, on=["company", "basis", "metric", "segment"], suffixes=("_s", "_e"))
    j = j[(j.period_end_e > j.period_end_s) & (j.value_cr_s > 0) & (j.value_cr_e > 0)].copy()
    j["years"] = j.period_end_e - j.period_end_s
    j["change_pct"] = (100 * (j.value_cr_e / j.value_cr_s - 1)).round(2)
    j["cagr_pct"] = (100 * ((j.value_cr_e / j.value_cr_s) ** (1.0 / j.years) - 1)).round(2)
    j = j.rename(columns={"period_s": "start_period", "period_e": "end_period",
                          "value_cr_s": "start_value_cr", "value_cr_e": "end_value_cr"})
    j = _add_default_basis(j, ["company", "metric", "segment", "start_period", "end_period"])
    return j[cols].reset_index(drop=True)


def build_db(progress=print):
    import pandas as pd
    reports = stores.load_reports()
    rows = []
    if config.EXTRACTIONS.exists():
        for line in open(config.EXTRACTIONS, encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            for r in rec["rows"]:
                if r["metric"] not in METRICS:
                    continue
                for v in r["values"]:
                    try:
                        label, pend, months = normalize_fy(v["period"])
                    except ValueError:
                        continue
                    unit = "percent" if r["metric"] in PERCENT else "rupees_per_share" if r["metric"] in PER_SHARE else rec["unit"]
                    # period length: a 15-month period is flagged so growth is never computed across it
                    rmeta = reports.get(f"{rec['company'].replace(' ', '')}_{label}")
                    months = rmeta["months"] if rmeta else 12
                    rows.append({
                        "company": rec["company"], "basis": rec["basis"], "metric": r["metric"],
                        "segment": (r.get("segment") or "").strip(), "period": label, "period_end": pend,
                        "partial_period": months != 12, "value": v["value"], "unit": unit,
                        "unit_inferred": bool(rec["unit_inferred"]) and unit not in ("percent", "rupees_per_share"),
                        "value_cr": _to_cr(v["value"], rec["unit"], r["metric"]), "label": r.get("label", ""),
                        "section": rec["section"], "report_fy": rec["report_fy"],
                        "report_end": normalize_fy(rec["report_fy"])[1], "table_id": rec["table_id"], "model": rec["model"]})
    df = pd.DataFrame(rows)
    if df.empty:
        progress("no facts to load yet")
        df = pd.DataFrame(columns=["company", "basis", "metric", "segment", "period", "period_end", "partial_period", "value",
                                   "unit", "unit_inferred", "value_cr", "label", "section", "report_fy", "report_end", "table_id", "model"])
    df["value_cr"] = pd.to_numeric(df["value_cr"], errors="coerce")
    df, standalone_only = _validate_bases(df)
    if standalone_only:
        progress(f"standalone-only companies (no valid consolidated data): {', '.join(standalone_only)}")
    df = _derived(df)
    config.FIN_DIR.mkdir(parents=True, exist_ok=True)
    tmp = str(config.DUCKDB_PATH) + ".tmp"
    for leftover in (tmp, tmp + ".wal"):          # a failed earlier build can leave a stale write-ahead file
        if os.path.exists(leftover):
            os.remove(leftover)
    con = duckdb.connect(tmp)
    con.register("df_in", df)
    con.execute("CREATE TABLE facts AS SELECT * FROM df_in")
    latest = _add_default_basis(_pick_latest(df), ["company", "metric", "segment", "period"])
    pairs = _growth_pairs(latest)
    con.register("pairs_in", pairs)
    con.execute("CREATE TABLE growth_pairs AS SELECT * FROM pairs_in")
    con.register("latest_in", latest)
    con.execute("CREATE TABLE facts_latest_t AS SELECT * FROM latest_in")
    con.execute("CREATE VIEW facts_latest AS SELECT * FROM facts_latest_t")
    n_all = con.execute("SELECT count(*) FROM facts").fetchone()[0]
    n_latest = con.execute("SELECT count(*) FROM facts_latest").fetchone()[0]
    con.close()
    os.replace(tmp, config.DUCKDB_PATH)       # atomic swap: running queries are never disturbed
    progress(f"DuckDB built: facts={n_all}, facts_latest={n_latest}")
    return n_all, n_latest


def _derived(df):
    """Margins and YoY growth computed once here, so the SQL model only has to select them."""
    import pandas as pd
    if df.empty:
        return df
    key = ["company", "basis", "segment", "period"]
    base = df[df.metric.isin(["revenue_from_operations", "net_profit", "ebitda", "profit_before_tax"]) & df.value_cr.notna()]
    base = _pick_latest(base)
    piv = base.pivot_table(index=key + ["period_end", "partial_period"], columns="metric", values="value_cr", aggfunc="last").reset_index()
    out = []

    def add(r, metric, val):
        out.append({"company": r["company"], "basis": r["basis"], "metric": metric, "segment": r["segment"], "period": r["period"],
                    "period_end": r["period_end"], "partial_period": r["partial_period"], "value": round(float(val), 4),
                    "unit": "percent", "unit_inferred": False, "value_cr": None, "label": "derived (computed from stored values)",
                    "section": "derived", "report_fy": r["period"], "report_end": r["period_end"], "table_id": "derived", "model": "derived"})
    if "revenue_from_operations" in piv:
        for _, r in piv.iterrows():
            rev = r.get("revenue_from_operations")
            if pd.notna(rev) and rev:
                for src, name in (("net_profit", "net_profit_margin_pct"), ("ebitda", "ebitda_margin_pct"), ("profit_before_tax", "pbt_margin_pct")):
                    if src in piv and pd.notna(r.get(src)):
                        add(r, name, 100 * r[src] / rev)
        for metric, name in (("revenue_from_operations", "revenue_growth_yoy_pct"), ("net_profit", "net_profit_growth_yoy_pct")):
            if metric not in piv:
                continue
            idx = {(r["company"], r["basis"], r["segment"], r["period_end"]): r for _, r in piv.iterrows()}
            for k, r in idx.items():
                prev = idx.get((k[0], k[1], k[2], k[3] - 1))
                if prev is not None and pd.notna(r.get(metric)) and pd.notna(prev.get(metric)) and prev[metric] \
                        and not r["partial_period"] and not prev["partial_period"]:
                    add(r, name, 100 * (r[metric] - prev[metric]) / abs(prev[metric]))
    return pd.concat([df, pd.DataFrame(out)], ignore_index=True) if out else df


# ---- text-to-SQL ----------------------------------------------------------------------------
FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|create|alter|attach|copy|pragma|install|load|export|call)\b", re.I)

SQL_RULES = """Write ONE DuckDB SQL SELECT query that answers the question.
Rules:
- Query the view facts_latest (and the table growth_pairs for growth/CAGR). SELECT only.
- Money amounts: use value_cr (Rs crore). Percentages (metrics ending in _pct) and per-share figures: use value.
  value_cr is NULL for percentages and per-share figures, so never select value_cr for them.
- To match several companies or periods write company IN ('A', 'B') or period IN (...). Never chain them with OR: AND binds tighter than OR,
  so an OR chain drops the metric and period filters.
- If the question does not say standalone or consolidated, filter default_basis = TRUE and do NOT hard-code a basis (it selects
  consolidated figures where a company has them and standalone otherwise). If the question names a basis, filter basis = that value.
- Comparisons and rankings ('highest', 'which company', 'compare', 'rank'): return ONE ROW PER COMPANY ordered by the value. Never use LIMIT
  unless the question asks for a top-N.
- Company totals: segment = ''. Segment questions: filter segment with ILIKE.
- Always return company, period and basis columns next to the numbers, ordered sensibly.
- Derived metrics already exist as rows: net_profit_margin_pct, ebitda_margin_pct, pbt_margin_pct,
  revenue_growth_yoy_pct, net_profit_growth_yoy_pct (use these instead of recomputing when they fit).
- For CAGR, growth between two chosen periods, or 'fastest growing / highest growth' questions, SELECT from the table growth_pairs
  (it can be queried together with facts_latest). Do not compute CAGR by hand.
- Other differences and rankings are computed in SQL. CAGR % = (POWER(end/start, 1.0/years) - 1) * 100.
- Periods marked partial_period = true are not 12 months long; mention them, never compare them as full years.
- Round results to 2 decimals.

Examples (follow their shape):
Q: Which company had the highest revenue CAGR from FY2022-23 to FY2025-26?
SELECT company, basis, start_period, end_period, cagr_pct FROM growth_pairs
WHERE metric = 'revenue_from_operations' AND segment = '' AND default_basis
  AND start_period = 'FY2022-23' AND end_period = 'FY2025-26' ORDER BY cagr_pct DESC;
Q: Compare the net profit margin of HUL and Dabur in FY2024-25.
SELECT company, period, basis, value AS net_profit_margin_pct FROM facts_latest
WHERE metric = 'net_profit_margin_pct' AND segment = '' AND default_basis AND period = 'FY2024-25'
  AND company IN ('HUL','Dabur') ORDER BY value DESC;
Q: Which company paid the highest dividend per share in FY2024-25?
SELECT company, period, basis, value AS dividend_per_share_rs FROM facts_latest
WHERE metric = 'dividend_per_share' AND segment = '' AND default_basis AND period = 'FY2024-25' ORDER BY value DESC;
Return only the SQL, no explanation."""


def schema_context(con) -> str:
    def distinct(col, where=""):
        return [r[0] for r in con.execute(f"SELECT DISTINCT {col} FROM facts_latest {where} ORDER BY 1").fetchall()]
    segs = distinct("company || ': ' || segment", "WHERE segment <> ''")[:60]
    all_c = set(distinct("company"))
    has_cons = {r[0] for r in con.execute("SELECT DISTINCT company FROM facts_latest WHERE basis='consolidated'").fetchall()}
    only_sa = sorted(all_c - has_cons)
    return f"""View facts_latest (one row per company, basis, metric, segment, period):
  company TEXT   -- one of: {', '.join(distinct('company'))}
  basis TEXT     -- 'standalone', 'consolidated' or 'unknown'
  metric TEXT    -- one of: {', '.join(distinct('metric'))}
  segment TEXT   -- '' for company totals, else a segment name
  period TEXT    -- e.g. {', '.join(distinct('period')[-8:])}
  period_end INTEGER   -- year the period ends (2025 for FY2024-25); use for ordering and CAGR
  partial_period BOOLEAN
  value DOUBLE, unit TEXT   -- as reported (crore, million, lakh, percent, rupees_per_share)
  value_cr DOUBLE    -- converted to Rs crore (NULL for %, per-share)
  default_basis BOOLEAN   -- TRUE on the preferred basis for that figure (consolidated, else standalone)
  label TEXT, section TEXT, report_fy TEXT, table_id TEXT   -- provenance
Table growth_pairs (precomputed growth between any two periods; money metrics only):
  company, basis, metric, segment, start_period, end_period, years INTEGER, start_value_cr, end_value_cr, change_pct, cagr_pct, default_basis BOOLEAN
Companies with standalone figures only (default_basis already selects standalone for them): {', '.join(only_sa) or 'none'}\nKnown segments: {'; '.join(segs) if segs else 'none'}"""


def clean_sql(text: str) -> str:
    text = re.sub(r"^```(?:sql)?\s*|\s*```$", "", text.strip(), flags=re.I)
    m = re.search(r"\b(with|select)\b", text, re.I)
    text = text[m.start():] if m else text
    return text.split(";")[0].strip()


_PCT_ALIAS = re.compile(r"\bvalue_cr(\s+AS\s+\w*(?:_pct|per_share|_rs|_eps)\w*)", re.I)


_OR_CHAIN = re.compile(r"\b(company|period|metric|basis|segment)\s*=\s*'([^']*)'((?:\s+OR\s+\1\s*=\s*'[^']*')+)", re.I)
_OR_VALUE = re.compile(r"OR\s+\w+\s*=\s*'([^']*)'", re.I)


def _or_to_in(m) -> str:
    vals = [m.group(2)] + _OR_VALUE.findall(m.group(3))
    return f"{m.group(1)} IN ({', '.join(repr(v) for v in vals)})"


# Rs amounts. `value` holds them as reported (Nestle India reports in Rs million, the rest in Rs crore); value_cr is always Rs crore.
_MONEY_METRICS = {"capex", "cash_and_equivalents", "cash_from_financing", "cash_from_investing", "cash_from_operations",
                  "cost_of_materials", "depreciation", "dividend_total", "ebit", "ebitda", "employee_cost", "finance_cost",
                  "inventories", "net_profit", "other_income", "profit_before_tax", "revenue_from_operations", "tax_expense",
                  "total_assets", "total_borrowings", "total_equity", "total_expenses", "total_income", "trade_payables",
                  "trade_receivables"}
_OTHER_METRICS = {"book_value_per_share", "dividend_per_share", "ebitda_margin_pct", "eps_basic", "eps_diluted",
                  "net_profit_growth_yoy_pct", "net_profit_margin_pct", "pbt_margin_pct", "revenue_growth_yoy_pct", "roce_pct", "roe_pct"}
_BARE_VALUE = re.compile(r"(?<![\w.'])value(?![\w'])")


def money_in_crore(sql: str) -> str:
    """A query that only asks for Rs amounts must use value_cr (Rs crore), not value: Nestle India's `value` is in Rs million, so a
    ranking by revenue would put it ten times too high."""
    named = set(re.findall(r"'([a-z_]+)'", sql)) & (_MONEY_METRICS | _OTHER_METRICS)
    if not named or not named <= _MONEY_METRICS:
        return sql
    return _BARE_VALUE.sub("COALESCE(value_cr, value)", sql)


def fix_sql(sql: str) -> str:
    """Repairs two typical small-model mistakes.
    1) value_cr (NULL for percentages and per-share figures) selected under a percentage alias -> use value.
    2) `... AND company = 'A' OR company = 'B'`: AND binds tighter than OR, so every branch after the first ignores the
       metric and period filters and returns the whole table. A chain of ORs on one column becomes `column IN (...)`."""
    sql = _PCT_ALIAS.sub(r"value\1", sql)
    return money_in_crore(_OR_CHAIN.sub(_or_to_in, sql))


def null_columns(df) -> list:
    """Columns that are empty in every row (the typical sign of selecting the wrong value column)."""
    return [str(c) for c in df.columns if df[c].dtype.kind == "f" and df[c].isna().all()]


def _fmt_cell(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:,.2f}"              # 81612.78 -> 81,612.78 ; 17.3641 -> 17.36
    return str(v)


def df_to_markdown(df, max_rows: int = 40) -> str:
    if df.empty:
        return "(no rows)"
    d = df.head(max_rows)
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for rec in d.to_dict("records"):
        lines.append("| " + " | ".join(_fmt_cell(v) for v in rec.values()) + " |")
    if len(df) > max_rows:
        lines.append(f"... {len(df) - max_rows} more rows")
    return "\n".join(lines)


def source_tables(df, query_str: str, limit: int = 4):
    """Report tables the figures in a database result came from: [{id, company, period}] (the id is also the passage id of that table)."""
    try:
        cols = {str(c).lower(): c for c in df.columns}
        ccol, pcol = cols.get("company"), cols.get("period") or cols.get("fy")
        qp = re.findall(r"(?:FY|CY)\s?\d{4}(?:-\d{2})?", query_str, flags=re.I)
        qperiod = re.sub(r"\s", "", qp[0]).upper() if qp else None
        if ccol is None:
            return []
        out, seen = [], set()
        con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
        try:
            for _, r in df.head(30).iterrows():
                period = str(r[pcol]) if pcol and str(r[pcol]).upper().startswith(("FY", "CY")) else qperiod
                nums = [float(v) for v in r.values if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v]
                if not period or not nums:
                    continue
                cond = " OR ".join(["ABS(value - ?) < 0.006 OR ABS(COALESCE(value_cr, -1e18) - ?) < 0.006"] * len(nums))
                args = [str(r[ccol]), period] + [x for v in nums for x in (v, v)]
                for tid, c, p in con.execute(
                        f"SELECT DISTINCT table_id, company, period FROM facts_latest WHERE company = ? AND period = ? AND ({cond}) LIMIT 3", args).fetchall():
                    if tid and tid not in seen:
                        seen.add(tid)
                        out.append({"id": tid, "company": c, "period": p})
        finally:
            con.close()
        return out[:limit]
    except Exception:
        return []


class SQLEngine(CustomQueryEngine):
    """Question -> SQL (qwen2.5-coder, direct Ollama call) -> DuckDB -> Response whose source node holds the SQL and rows."""
    max_fix: int = 3

    def _ask_sql(self, messages) -> str:
        t = time.time()
        r = ollama.Client(host=config.OLLAMA_URL, timeout=300).chat(
            model=config.SQL_MODEL, messages=messages, options={"temperature": 0, "num_ctx": 8192})
        timing.log("SQL model", time.time() - t)
        return r["message"]["content"]

    def run(self, question: str):
        if not config.DUCKDB_PATH.exists():
            return None, None, "The numbers database has not been built yet."
        con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
        try:
            if con.execute("SELECT count(*) FROM facts").fetchone()[0] == 0:
                return None, None, "The numbers database is empty."
            messages = [{"role": "system", "content": SQL_RULES + "\n\n" + schema_context(con)},
                        {"role": "user", "content": question}]
            err = None
            for attempt in range(self.max_fix + 1):
                sql = fix_sql(clean_sql(self._ask_sql(messages)))
                if not sql.lower().startswith(("select", "with")) or FORBIDDEN.search(sql):
                    err = "Only a single SELECT query on facts_latest is allowed."
                else:
                    try:
                        df = con.execute(sql).fetchdf()
                        if len(df):
                            empty = null_columns(df)
                            if empty and re.search(r"\bvalue_cr\b", sql):      # wrong value column: repair once, silently
                                sql2 = re.sub(r"\bvalue_cr\b", "value", sql)
                                df2 = con.execute(sql2).fetchdf()
                                if len(df2) and not null_columns(df2):
                                    return sql2, df2, None
                            if empty:
                                err = (f"Column(s) {', '.join(empty)} are NULL in every row. value_cr is NULL for percentages and "
                                       "per-share figures: select value for those, value_cr only for Rs crore amounts.")
                            else:
                                return sql, df, None
                        else:
                            err = "The query ran but returned no rows. Check the filters (company, basis, metric, period values)."
                    except Exception as e:
                        err = str(e)
                messages += [{"role": "assistant", "content": sql},
                             {"role": "user", "content": f"That failed: {err}\nReturn a corrected SQL query only."}]
            return sql, None, err
        finally:
            con.close()

    @staticmethod
    def resolve_basis(question: str, df):
        """If the question names neither standalone nor consolidated and the result holds both for one company/period,
        keep consolidated (or standalone for companies without consolidated accounts), exactly as the SQL rules say."""
        q = question.lower()
        if "basis" not in df.columns or "standalone" in q or "consolidated" in q or df["basis"].nunique() < 2:
            return df
        rank = {"consolidated": 0, "standalone": 1, "unknown": 2}
        keys = [c for c in df.columns if c in ("company", "period", "metric", "segment", "start_period", "end_period")]
        d = df.assign(_r=df["basis"].map(rank).fillna(3))
        d = d.sort_values("_r", kind="stable").drop_duplicates(keys, keep="first") if keys else d
        return d.drop(columns="_r").sort_index()

    @staticmethod
    def missing_companies_note(question: str, df) -> str:
        """Say which companies have no row in this result, so the answer can explain a gap (for example Nestle India has no
        FY2022-23 period) instead of silently leaving it out. If the question names companies, only those are checked."""
        if "company" not in df.columns:
            return ""
        try:
            from retrieval import parse_scope
            named = parse_scope(question)["companies"]
            con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
            known = [r[0] for r in con.execute("SELECT DISTINCT company FROM facts ORDER BY 1").fetchall()]
            con.close()
        except Exception:
            return ""
        present = set(df["company"].astype(str))
        missing = [c for c in (named or known) if c in known and c not in present]
        if not missing:
            return ""
        return ("NOTE: the database has no matching figures for " + ", ".join(missing) +
                " (the requested metric or periods are not available for them in the indexed reports).")

    @staticmethod
    def basis_note(question: str, sql: str, df) -> str:
        """The preferred basis is shown in the result. Also report (a) the other basis where the database has it, (b) companies that
        only have one basis, (c) a warning when the companies in one result are on different bases."""
        q = question.lower()
        if "basis" not in df.columns or "company" not in df.columns or "standalone" in q or "consolidated" in q:
            return ""
        lines = []
        try:
            keys = [c for c in ("company", "period", "start_period", "end_period", "metric", "segment") if c in df.columns]
            alt_sql = re.sub(r"\s+AND\s+default_basis\b", "", sql, flags=re.I)
            others = None
            if alt_sql != sql:
                con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
                try:
                    alt = con.execute(alt_sql).fetchdf()
                finally:
                    con.close()
                if len(alt) and "basis" in alt.columns and all(k in alt.columns for k in keys):
                    m = alt.merge(df[keys + ["basis"]], on=keys, how="inner", suffixes=("", "_main"))
                    others = m[(m["basis"] != m["basis_main"]) & (m["basis"] != "unknown")].drop(columns="basis_main").drop_duplicates()
            if others is not None and len(others):
                lines.append("OTHER BASIS ALSO IN THE DATABASE (same figure, not used above; mention it next to the main value):\n"
                             + df_to_markdown(others.head(40)))
                only_one = sorted(set(df["company"].astype(str)) - set(others["company"].astype(str)))
            else:
                only_one = sorted(set(df["company"].astype(str)))
            if only_one and others is not None:
                lines.append("Only one basis exists in the database for: " + ", ".join(only_one) + ".")
        except Exception:
            pass
        if df["basis"].nunique() > 1:
            by = df.groupby("basis")["company"].apply(lambda x: ", ".join(sorted(set(map(str, x))))).to_dict()
            lines.append("WARNING: the companies above are on different bases (" + "; ".join(f"{b}: {c}" for b, c in by.items()) +
                         "). State this and do not present the gap between them as like-for-like.")
        return "\n".join(lines)

    def custom_query(self, query_str: str) -> Response:
        timing.step("Querying the financial database", query_str[:160])
        sql, df, err = self.run(query_str)
        if df is not None:
            df = self.resolve_basis(query_str, df)
        if df is None:
            known = ""
            try:
                con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
                known = " Companies in the numbers database: " + ", ".join(r[0] for r in con.execute("SELECT DISTINCT company FROM facts ORDER BY 1").fetchall()) + "."
                con.close()
            except Exception:
                pass
            return Response(response=f"No figures could be retrieved from the numbers database ({err}).{known}",
                            source_nodes=[], metadata={"sql": sql, "error": err})
        md = df_to_markdown(df)
        missing_note = self.missing_companies_note(query_str, df)
        if missing_note:
            md = md + "\n" + missing_note
        bnote = self.basis_note(query_str, sql, df)
        if bnote:
            md = md + "\n" + bnote
        evidence = TextNode(text=f"Structured figures from DuckDB (amounts in Rs crore unless a unit column says otherwise).\n"
                                 f"SQL: {sql}\nResult:\n{md}",
                            metadata={"node_kind": "sql_result", "sql": sql,
                                      "chart_table": df.head(120).to_json(orient="split", date_format="iso"),
                                      "table_ids": source_tables(df, query_str)})
        evidence.excluded_llm_metadata_keys = ["node_kind", "sql", "chart_table", "table_ids"]
        evidence.excluded_embed_metadata_keys = ["node_kind", "sql", "chart_table", "table_ids"]
        return Response(response=f"Result of the database query (columns ending in _cr are Rs crore; *_pct are percent):\n{md}",
                        source_nodes=[NodeWithScore(node=evidence, score=1.0)], metadata={"sql": sql})