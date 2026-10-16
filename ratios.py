"""Ratio and brief tables computed straight from the numbers database (no language model, so no guessing).

    dupont(companies, period)   -> DataFrame: net margin x asset turnover x equity multiplier = ROE, with reported ROE/ROCE beside it
    brief_table(company)        -> markdown table of the key figures for the latest year and the year before
"""
import duckdb
import pandas as pd

import config

_AMOUNTS = ("revenue_from_operations", "net_profit", "total_assets", "total_equity", "ebitda", "profit_before_tax")
_OTHERS = ("roe_pct", "roce_pct", "ebitda_margin_pct", "net_profit_margin_pct", "revenue_growth_yoy_pct",
           "eps_basic", "dividend_per_share")


def _fetch(companies, periods, metrics) -> pd.DataFrame:
    con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
    try:
        cs = ", ".join(f"'{c}'" for c in companies)
        ps = ", ".join(f"'{p}'" for p in periods)
        ms = ", ".join(f"'{m}'" for m in metrics)
        return con.execute(
            f"SELECT company, period, basis, metric, COALESCE(value_cr, value) AS v FROM facts_latest "
            f"WHERE segment = '' AND default_basis AND company IN ({cs}) AND period IN ({ps}) AND metric IN ({ms})").fetchdf()
    finally:
        con.close()


def all_companies():
    con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
    try:
        return [r[0] for r in con.execute("SELECT DISTINCT company FROM facts ORDER BY 1").fetchall()]
    finally:
        con.close()


def latest_period(company: str) -> str:
    con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
    try:
        r = con.execute("SELECT period FROM facts_latest WHERE company = ? AND metric = 'revenue_from_operations' AND segment = '' "
                        "AND period LIKE 'FY%' AND NOT partial_period ORDER BY period_end DESC LIMIT 1", [company]).fetchone()
        return r[0] if r else ""
    finally:
        con.close()


def dupont(companies, period: str) -> pd.DataFrame:
    df = _fetch(companies, [period], _AMOUNTS + _OTHERS)
    rows = []
    for c in companies:
        d = df[df.company == c]
        if d.empty:
            rows.append({"company": c, "basis": "", "note": f"no figures for {period} in the database"})
            continue
        v = {m: float(x) for m, x in zip(d.metric, d.v)}
        basis = ", ".join(sorted(set(d.basis)))
        row = {"company": c, "basis": basis}
        need = ("revenue_from_operations", "net_profit", "total_assets", "total_equity")
        if not all(m in v for m in need):
            row["note"] = "missing: " + ", ".join(m for m in need if m not in v)
        elif v["total_assets"] < v["total_equity"]:
            row["note"] = "total assets are lower than equity in the database, so the split is not shown (data to be checked)"
        else:
            nm = v["net_profit"] / v["revenue_from_operations"] * 100
            at = v["revenue_from_operations"] / v["total_assets"]
            em = v["total_assets"] / v["total_equity"]
            row.update({"net_margin_pct": round(nm, 2), "asset_turnover_x": round(at, 2), "equity_multiplier_x": round(em, 2),
                        "roe_dupont_pct": round(nm * at * em, 2)})
            row["note"] = "year-end balances"
        if "roe_pct" in v:
            row["roe_reported_pct"] = round(v["roe_pct"], 2)
        if "roce_pct" in v:
            row["roce_reported_pct"] = round(v["roce_pct"], 2)
        rows.append(row)
    return pd.DataFrame(rows)


_LABELS = [("revenue_from_operations", "Revenue from operations (Rs crore)"), ("revenue_growth_yoy_pct", "Revenue growth (%)"),
           ("ebitda_margin_pct", "EBITDA margin (%)"), ("net_profit", "Net profit (Rs crore)"),
           ("net_profit_margin_pct", "Net profit margin (%)"), ("roe_pct", "ROE (%)"), ("roce_pct", "ROCE (%)"),
           ("eps_basic", "EPS (Rs)"), ("dividend_per_share", "Dividend per share (Rs)")]


def brief_table(company: str):
    """(markdown table, latest period) for the latest full year and the year before, each figure on the basis the database prefers."""
    cur = latest_period(company)
    if not cur:
        return "", ""
    prev = f"FY{int(cur[2:6]) - 1}-{str(int(cur[2:6]))[2:]}"
    df = _fetch([company], [cur, prev], [m for m, _ in _LABELS])
    if df.empty:
        return "", cur
    bases = set()
    lines = [f"| Figure | {prev} | {cur} | Basis |", "|---|---|---|---|"]
    for m, label in _LABELS:
        d = df[df.metric == m]
        if d.empty:
            continue
        get = lambda p: (f"{float(d[d.period == p].v.iloc[0]):,.2f}" if (d.period == p).any() else "")
        basis = ", ".join(sorted(set(d.basis)))
        bases |= set(d.basis)
        lines.append(f"| {label} | {get(prev)} | {get(cur)} | {basis} |")
    out = "\n".join(lines)
    if len({b for b in bases if b in ("standalone", "consolidated")}) > 1:
        rows = {b: [lab for m, lab in _LABELS if ((df.metric == m) & (df.basis == b)).any()] for b in ("standalone", "consolidated")}
        out += ("\n\n_Not like-for-like: the rows are on different bases (" +
                "; ".join(f"{b}: {', '.join(l.split(' (')[0] for l in v)}" for b, v in rows.items() if v) + ")._")
    return out, cur