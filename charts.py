"""Charts for database answers (Plotly).

`chart_spec` turns a result table into a description of the chart (type, series, axis titles with units);
`make_figure` draws it. Both are plain functions so they can be tested without the UI.

Rules
  - fewer than 2 values, or several different metrics in one table  -> no chart
  - several fiscal years                                              -> line chart, x = fiscal year, one line per company/segment
  - one period, several companies (or one company, several segments)  -> bar chart, sorted high to low
  - y axis title = metric + unit (Rs crore, %, Rs per share); x axis title = Company / Fiscal year / Segment
  - subtitle states the basis (consolidated / standalone) so mixed bases are visible
"""
import json
import re
import traceback
from typing import List, Optional

import pandas as pd

try:
    from corpus import normalize_fy
except Exception:           # pragma: no cover
    normalize_fy = None

UNIT_LABEL = {"percent": "%", "rupees_per_share": "Rs per share", "crore": "Rs crore", "million": "Rs million",
              "lakh": "Rs lakh", "billion": "Rs billion", "thousand": "Rs thousand"}
TEXT_COLS = {"company", "basis", "metric", "segment", "period", "start_period", "end_period", "label", "unit", "section",
             "report_fy", "table_id", "model"}
NOT_VALUE = {"period_end", "years", "partial_period", "default_basis", "report_end", "unit_inferred"}
PREFERRED = ["cagr_pct", "change_pct", "value_cr", "value", "end_value_cr"]
MAX_POINTS = 60


def humanize(name: str) -> str:
    s = re.sub(r"_(pct|cr|rs)$", "", str(name))
    s = s.replace("_", " ").strip()
    s = re.sub(r"\bebitda\b", "EBITDA", s, flags=re.I)
    s = re.sub(r"\bebit\b", "EBIT", s, flags=re.I)
    s = re.sub(r"\beps\b", "EPS", s, flags=re.I)
    s = re.sub(r"\bpbt\b", "PBT", s, flags=re.I)
    s = re.sub(r"\byoy\b", "YoY", s, flags=re.I)
    return s[:1].upper() + s[1:] if s else s


def _period_key(p: str):
    if normalize_fy:
        try:
            return normalize_fy(p)[1]
        except Exception:
            pass
    return str(p)


def _value_column(df: pd.DataFrame) -> Optional[str]:
    nums = []
    for c in df.columns:
        if c in TEXT_COLS or c in NOT_VALUE:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            df[c] = s
            nums.append(c)
    for c in PREFERRED:
        if c in nums:
            return c
    for c in nums:
        if c.endswith("_pct"):
            return c
    for c in nums:
        if c.endswith(("_cr", "_rs")):
            return c
    return nums[0] if nums else None


def _unit_for(col: str, df: pd.DataFrame) -> str:
    if col.endswith("_pct"):
        return "%"
    if col.endswith("_cr"):
        return "Rs crore"
    if col.endswith("_rs") or "per_share" in col:
        return "Rs per share"
    if "unit" in df.columns and df["unit"].dropna().nunique() == 1:
        return UNIT_LABEL.get(str(df["unit"].dropna().iloc[0]), "")
    return ""


def _metric_label(df: pd.DataFrame, sql: str) -> Optional[str]:
    if "metric" in df.columns and df["metric"].dropna().nunique() == 1:
        return humanize(df["metric"].dropna().iloc[0])
    found = set(re.findall(r"metric\s*=\s*'([a-z_]+)'", sql or "", flags=re.I))
    if len(found) == 1:
        return humanize(found.pop())
    return None


def _basis_note(df: pd.DataFrame) -> str:
    if "basis" not in df.columns or "company" not in df.columns:
        return ""
    by = df.dropna(subset=["basis"]).groupby("basis")["company"].unique()
    if len(by) == 0:
        return ""
    if len(by) == 1:
        return f"Basis: {by.index[0]}"
    return "Basis: " + "; ".join(f"{b} - {', '.join(sorted(map(str, cs)))}" for b, cs in by.items())


def chart_spec(df: pd.DataFrame, sql: str = "") -> Optional[dict]:
    df = df.head(MAX_POINTS * 2).copy()
    if len(df) < 2:
        return None
    if "metric" in df.columns and df["metric"].dropna().nunique() > 1:
        return None                                   # different metrics have different units: never share one axis
    col = _value_column(df)
    if col is None:
        return None
    df = df[df[col].notna()]
    if len(df) < 2:
        return None

    unit = _unit_for(col, df)
    metric = _metric_label(df, sql)
    if metric:
        core = metric
    elif col in ("value", "value_cr"):
        core = "Amount"
    else:
        core = humanize(col)
    if col == "cagr_pct":
        core = f"{metric or ''} CAGR".strip()
    elif col == "change_pct":
        core = f"{metric or ''} change".strip()
    fmt = "pct" if unit == "%" else "money" if unit.startswith("Rs") and "share" not in unit else "pershare" if "share" in unit else "num"

    has_company = "company" in df.columns and df["company"].notna().any()
    seg_ok = "segment" in df.columns and (df["segment"].fillna("") != "").any()
    one_company = (not has_company) or df["company"].nunique() <= 1
    if seg_ok and one_company:
        entity = "segment"
    elif has_company:
        entity = "company"
    else:
        return None

    time_col = next((c for c in ("period", "end_period") if c in df.columns and df[c].nunique() >= 2), None)
    if "start_period" in df.columns and "end_period" in df.columns and df["start_period"].nunique() == 1 and df["end_period"].nunique() == 1:
        time_col = None
        period_text = f"{df['start_period'].iloc[0]} to {df['end_period'].iloc[0]}"
    elif time_col:
        periods = sorted(df[time_col].dropna().unique(), key=_period_key)
        period_text = f"{periods[0]} to {periods[-1]}"
    elif "period" in df.columns and df["period"].nunique() == 1:
        period_text = str(df["period"].iloc[0])
    else:
        period_text = ""

    def label(row):
        name = str(row[entity]) if entity == "segment" else str(row["company"])
        if "basis" in df.columns and entity == "company":
            n_basis = df[df["company"] == row["company"]]["basis"].nunique()
            if n_basis > 1:
                name += f" ({row['basis']})"
        return name

    df["_label"] = df.apply(label, axis=1)
    series = []
    if time_col:
        periods = sorted(df[time_col].dropna().unique(), key=_period_key)
        for name, g in df.groupby("_label", sort=False):
            g = g.drop_duplicates(time_col).set_index(time_col).reindex(periods)
            series.append({"name": name, "x": [str(p) for p in periods], "y": [None if pd.isna(v) else float(v) for v in g[col]]})
        kind, x_title = "line", "Fiscal year"
    else:
        g = df.drop_duplicates("_label").sort_values(col, ascending=False)
        series.append({"name": core or "value", "x": g["_label"].tolist(), "y": [float(v) for v in g[col]]})
        kind, x_title = "bar", "Company" if entity == "company" else "Segment"

    title = f"{core}, {period_text}" if period_text else core
    y_title = f"{core} ({unit})" if unit else core
    if not core:
        return None
    return {"kind": kind, "title": title, "subtitle": _basis_note(df), "x_title": x_title, "y_title": y_title,
            "unit": unit, "fmt": fmt, "series": series}


def make_figure(spec: dict):
    import plotly.graph_objects as go
    fmt = spec["fmt"]
    tick = {"pct": ".1f", "money": ",.0f", "pershare": ",.2f", "num": ",.2f"}[fmt]
    suffix = "%" if fmt == "pct" else ""
    fig = go.Figure()
    for s in spec["series"]:
        hover = f"%{{x}}<br>{spec['y_title']}: %{{y:{tick}}}{suffix}<extra>{s['name']}</extra>"
        if spec["kind"] == "bar":
            fig.add_trace(go.Bar(x=s["x"], y=s["y"], name=s["name"], text=s["y"], texttemplate=f"%{{y:{tick}}}{suffix}",
                                 textposition="outside", cliponaxis=False, hovertemplate=hover, marker_color="#1F6F8B"))
        else:
            fig.add_trace(go.Scatter(x=s["x"], y=s["y"], name=s["name"], mode="lines+markers", connectgaps=True, hovertemplate=hover))
    title = spec["title"] + (f"<br><sup>{spec['subtitle']}</sup>" if spec["subtitle"] else "")
    fig.update_layout(template="plotly_white", title=dict(text=title, x=0.01), height=430, margin=dict(t=80, r=20, b=60, l=70),
                      showlegend=spec["kind"] == "line" and len(spec["series"]) > 1,
                      xaxis=dict(title=dict(text=spec["x_title"]), type="category",
                                 categoryorder="array", categoryarray=spec["series"][0]["x"]),
                      yaxis=dict(title=dict(text=spec["y_title"]), tickformat=tick, ticksuffix=suffix, zeroline=True, rangemode="tozero" if spec["kind"] == "bar" else "normal"))
    return fig


def table_from_node(node):
    meta = node.metadata or {}
    raw = meta.get("chart_table")
    if not raw:
        return None, ""
    obj = json.loads(raw) if isinstance(raw, str) else raw
    return pd.DataFrame(obj["data"], columns=obj["columns"]), meta.get("sql", "")


def build_charts(source_nodes, max_charts: int = 2) -> List:
    """Figures for the database results behind an answer. Never raises: a chart problem must not hide the answer."""
    figs, seen = [], set()
    for n in source_nodes or []:
        try:
            node = n.node
            if node.node_id in seen or node.metadata.get("node_kind") != "sql_result":
                continue
            seen.add(node.node_id)
            df, sql = table_from_node(node)
            if df is None:
                continue
            spec = chart_spec(df, sql)
            if spec:
                figs.append(make_figure(spec))
            if len(figs) >= max_charts:
                break
        except Exception:
            traceback.print_exc()
            continue
    return figs