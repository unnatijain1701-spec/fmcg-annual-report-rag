"""Ranking questions ("which company had the highest X", "rank the companies by X") answered straight from the numbers database.
No language model writes the SQL or the numbers, so the order and values are exactly what the database holds."""
import re
from typing import Optional

import duckdb
import pandas as pd

import config
from corpus import COMPANY_ALIASES, normalize_fy

# phrase -> (metric, readable name, unit text); longest phrases first, matched spans are removed so "revenue growth" is not also "revenue"
_METRICS = [
    (r"net profit growth|profit growth|growth in (?:net )?profit|pat growth", "net_profit_growth_yoy_pct", "net profit growth", "%"),
    (r"revenue growth|sales growth|growth in (?:revenue|sales)|top[- ]line growth", "revenue_growth_yoy_pct", "revenue growth", "%"),
    (r"ebitda margin|operating margin", "ebitda_margin_pct", "EBITDA margin", "%"),
    (r"net profit margin|net margin|pat margin|profit margin", "net_profit_margin_pct", "net profit margin", "%"),
    (r"pbt margin", "pbt_margin_pct", "PBT margin", "%"),
    (r"roce|return on capital employed", "roce_pct", "ROCE", "%"),
    (r"\broe\b|return on equity", "roe_pct", "ROE", "%"),
    (r"dividend per share|\bdps\b", "dividend_per_share", "dividend per share", "Rs"),
    (r"earnings per share|\beps\b", "eps_basic", "earnings per share", "Rs"),
    (r"revenue from operations|revenue|turnover|net sales|\bsales\b", "revenue_from_operations", "revenue from operations", "Rs crore"),
    (r"net profit|profit after tax|\bpat\b", "net_profit", "net profit", "Rs crore"),
    (r"\bebitda\b", "ebitda", "EBITDA", "Rs crore"),
]
_RANK = re.compile(r"\b(rank|ranking|which (?:company|firm|one)|which of|highest|lowest|best|worst|top|largest|biggest|smallest|"
                   r"fastest|slowest|most|least|higher|lower)\b", re.I)
_LOW = re.compile(r"\b(lowest|worst|smallest|slowest|least|lower)\b", re.I)
_SKIP = re.compile(r"\b(why|reason|reasons|explain|how|cagr|average|trend|over the years|each year|year[- ]by[- ]year|between|from fy)\b", re.I)


def _companies_named(q: str):
    out = []
    for name, aliases in COMPANY_ALIASES.items():
        if any(re.search(rf"(?<![a-z]){re.escape(a)}(?![a-z])", q) for a in aliases):
            out.append(name)
    return out


def _metric_of(q: str):
    found, rest = [], q
    for pat, metric, name, unit in _METRICS:
        if re.search(pat, rest):
            found.append((metric, name, unit))
            rest = re.sub(pat, " ", rest)
    return found[0] if len(found) == 1 else None            # two different measures in one question: leave it to the general route


def is_ranking(question: str) -> bool:
    q = question.lower()
    return bool(_RANK.search(q)) and not _SKIP.search(q) and _metric_of(q) is not None


def rank_response(question: str):
    """Returns a llama_index Response, or None when the question is not a plain single-measure ranking."""
    from llama_index.core.base.response.schema import Response
    from llama_index.core.schema import NodeWithScore, TextNode
    q = question.lower()
    if not is_ranking(question):
        return None
    metric, name, unit = _metric_of(q)
    named = _companies_named(q)
    fys = []
    for m in re.finditer(r"(?:fy\s*'?\d{2,4}(?:\s*[-–/]\s*\d{2,4})?|20\d\d\s*[-–/]\s*\d{2,4}|cy\s*20\d\d)", q):
        try:
            fys.append(normalize_fy(m.group(0))[0])
        except ValueError:
            pass
    con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
    try:
        period = fys[0] if fys else None
        if period is None:                                    # the newest year in which at least two companies have the measure
            r = con.execute("SELECT period FROM facts_latest WHERE metric = ? AND segment = '' AND default_basis AND period LIKE 'FY%' "
                            "GROUP BY period, period_end HAVING COUNT(DISTINCT company) >= 2 ORDER BY period_end DESC LIMIT 1", [metric]).fetchone()
            if not r:
                return None
            period = r[0]
        df = con.execute("SELECT company, basis, period, COALESCE(value_cr, value) AS value, table_id FROM facts_latest "
                         "WHERE metric = ? AND segment = '' AND default_basis AND period = ?", [metric, period]).fetchdf()
        everyone = [r[0] for r in con.execute("SELECT DISTINCT company FROM facts ORDER BY 1").fetchall()]
    finally:
        con.close()
    if named:
        if len(named) < 2:
            return None                                       # one company named: that is a single-figure question
        df = df[df.company.isin(named)]
        everyone = [c for c in everyone if c in named]
    df = df.dropna(subset=["value"]).drop_duplicates("company")
    if df.empty:
        return None
    low = bool(_LOW.search(q))
    df = df.sort_values("value", ascending=low).reset_index(drop=True)
    fmt = (lambda v: f"{v:,.2f}")
    u = {"%": "%", "Rs": " Rs", "Rs crore": " Rs crore"}[unit]
    first = df.iloc[0]
    word = "lowest" if low else "highest"
    head = f"{first.company} had the {word} {name} in {period}: {fmt(first.value)}{u} ({first.basis})."
    if "rank" in q or "all" in q.split():
        head = f"Ranking by {name}, {period}, {'lowest' if low else 'highest'} first."
    lines = [head, "", f"| Rank | Company | {name} ({unit}) | Basis |", "|---|---|---|---|"]
    for i, r in df.iterrows():
        lines.append(f"| {i + 1} | {r.company} | {fmt(r.value)} | {r.basis} |")
    absent = [c for c in everyone if c not in set(df.company)]
    if absent:
        lines += ["", f"Not available in the database for {period}: {', '.join(absent)}."]
    lines += ["", "[Financial database]"]
    text = "\n".join(lines)
    tids = [{"id": r.table_id, "company": r.company, "period": r.period} for r in df.itertuples() if r.table_id and r.table_id != "derived"][:4]
    shown = df[["company", "basis", "period", "value"]].rename(columns={"value": name})
    node = TextNode(text=f"Ranking computed from the financial database ({metric}, {period}); order is exact.\n{text}",
                    metadata={"node_kind": "sql_result", "sql": f"ranking of {metric} for {period}",
                              "chart_table": shown.to_json(orient="split", date_format="iso"), "table_ids": tids})
    for attr in ("excluded_llm_metadata_keys", "excluded_embed_metadata_keys"):
        setattr(node, attr, ["node_kind", "sql", "chart_table", "table_ids"])
    return Response(response=text, source_nodes=[NodeWithScore(node=node, score=1.0)], metadata={})


# ---- statistics: average / median / range / standard deviation, worked out in code from the database ----
_STAT = re.compile(r"\b(average|mean|median|standard deviation|std\.? ?dev\w*|range|spread)\b", re.I)


def _stat_kind(q: str):
    m = _STAT.search(q)
    if not m:
        return None
    w = m.group(1).lower()
    if w in ("average", "mean"):
        return "mean"
    if w == "median":
        return "median"
    if w in ("range", "spread"):
        return "range"
    return "stdev"


def is_stat(question: str) -> bool:
    q = question.lower()
    return _stat_kind(q) is not None and _metric_of(q) is not None and "cagr" not in q


def stat_response(question: str):
    """Average / median / range / standard deviation of one measure, across companies in one year or across years for one company."""
    import statistics
    from llama_index.core.base.response.schema import Response
    from llama_index.core.schema import NodeWithScore, TextNode
    q = question.lower()
    if not is_stat(question):
        return None
    kind = _stat_kind(q)
    metric, name, unit = _metric_of(q)
    named = _companies_named(q)
    fys = []
    for m in re.finditer(r"(?:fy\s*'?\d{2,4}(?:\s*[-–/]\s*\d{2,4})?|20\d\d\s*[-–/]\s*\d{2,4}|cy\s*20\d\d)", q):
        try:
            fy = normalize_fy(m.group(0))[0]
            if fy not in fys:
                fys.append(fy)
        except ValueError:
            pass
    con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
    try:
        base = "metric = ? AND segment = '' AND default_basis"
        if len(named) == 1 and len(fys) >= 2:                        # one company across years
            lo, hi = [con.execute("SELECT MIN(period_end) FROM facts_latest WHERE period = ?", [f]).fetchone()[0] for f in (fys[0], fys[-1])]
            lo, hi = min(lo, hi), max(lo, hi)
            df = con.execute(f"SELECT company AS label, basis, period, COALESCE(value_cr, value) AS value FROM facts_latest WHERE {base} AND company = ? "
                             "AND period_end BETWEEN ? AND ? AND period LIKE 'FY%' ORDER BY period_end", [metric, named[0], lo, hi]).fetchdf()
            df["label"] = df["period"]
            scope = f"{fys[0]} to {fys[-1]}"
            who = f"{named[0]}'s"
        else:                                                         # several companies in one year
            period = fys[0] if fys else None
            if period is None:
                r = con.execute("SELECT period FROM facts_latest WHERE metric = ? AND segment = '' AND default_basis AND period LIKE 'FY%' "
                                "GROUP BY period, period_end HAVING COUNT(DISTINCT company) >= 2 ORDER BY period_end DESC LIMIT 1", [metric]).fetchone()
                if not r:
                    return None
                period = r[0]
            df = con.execute(f"SELECT company AS label, company, basis, period, COALESCE(value_cr, value) AS value FROM facts_latest WHERE {base} AND period = ?",
                             [metric, period]).fetchdf().drop_duplicates("company")
            if len(named) >= 2:
                df = df[df.company.isin(named)]
            scope = period
            who = "the companies'" if len(named) >= 2 else "the five companies'"
    finally:
        con.close()
    df = df.dropna(subset=["value"]).reset_index(drop=True)
    if len(df) < 2 or (kind == "stdev" and len(df) < 3):
        return None
    vals = list(df["value"])
    f = lambda v: f"{v:,.2f}"
    u = {"%": "%", "Rs": " Rs", "Rs crore": " Rs crore"}[unit]
    if kind == "mean":
        res, label = statistics.mean(vals), "average"
        detail = f"mean of {len(vals)} values"
    elif kind == "median":
        res, label = statistics.median(vals), "median"
        detail = f"median of {len(vals)} values"
    elif kind == "range":
        hi_i, lo_i = int(df["value"].idxmax()), int(df["value"].idxmin())
        res, label = max(vals) - min(vals), "range"
        detail = f"highest {df.label[hi_i]} {f(max(vals))}, lowest {df.label[lo_i]} {f(min(vals))}"
    else:
        res, label = statistics.stdev(vals), "standard deviation"
        detail = f"sample standard deviation of {len(vals)} values"
    ru = " percentage points" if (unit == "%" and kind in ("range", "stdev")) else u
    head = f"The {label} of {who} {name} for {scope} is {f(res)}{ru} ({detail}); calculated by the system from the financial database."
    lines = [head, "", f"| {'Year' if 'period' in df.columns and len(named) == 1 and len(fys) >= 2 else 'Company'} | {name} ({unit}) | Basis |", "|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r.label} | {f(r.value)} | {r.basis} |")
    if len(named) != 1 or len(fys) < 2:
        everyone = []
        try:
            con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
            everyone = [r[0] for r in con.execute("SELECT DISTINCT company FROM facts ORDER BY 1").fetchall()]
            con.close()
        except Exception:
            pass
        absent = [c for c in everyone if c not in set(df["company"])] if "company" in df.columns and not named else []
        if absent:
            lines += ["", f"Not available in the database for {scope}: {', '.join(absent)} (left out of the calculation)."]
    lines += ["", "[Financial database]"]
    text = "\n".join(lines)
    shown = df[[c for c in ("company", "basis", "period", "value") if c in df.columns]].rename(columns={"value": name})
    node = TextNode(text=f"Statistic computed from the financial database ({metric}); values used:\n{text}",
                    metadata={"node_kind": "sql_result", "sql": f"{kind} of {metric} for {scope}",
                              "chart_table": shown.to_json(orient="split", date_format="iso"), "table_ids": []})
    for attr in ("excluded_llm_metadata_keys", "excluded_embed_metadata_keys"):
        setattr(node, attr, ["node_kind", "sql", "chart_table", "table_ids"])
    return Response(response=text, source_nodes=[NodeWithScore(node=node, score=1.0)], metadata={})