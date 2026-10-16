"""Company / fiscal-year metadata for every report (original corpus and new uploads)."""
import re
from pathlib import Path

COMPANY_ALIASES = {
    "ITC": ["itc"],
    "HUL": ["hul", "hindustan unilever", "hindustan unilever limited"],
    "Dabur": ["dabur", "dabur india"],
    "Nestle India": ["nestle", "nestlé", "nestle india", "nestlé india"],
    "Tata Consumer": ["tata consumer", "tata consumer products", "tcpl", "tata tea"],
}
DEFAULT_UNITS = {   # fallback only; the extractor tries to read the unit from the table first
    "ITC": "crore", "HUL": "crore", "Dabur": "crore", "Nestle India": "million", "Tata Consumer": "crore",
}


def normalize_fy(text: str):
    """'2024-25', 'FY25', 'FY 2024-25', 'March 2025', 'CY2022' -> ('FY2024-25', 2025, 12)."""
    t = str(text).strip()
    m = re.search(r"(?:CY|calendar year)\s*(20\d\d)", t, re.I)
    if m:
        y = int(m.group(1)); return f"CY{y}", y, 12
    m = re.search(r"(20\d\d)\s*[-–/]\s*(\d{2,4})", t)
    if m:
        a = int(m.group(1)); b = m.group(2)[-2:]
        return f"FY{a}-{b}", a + 1, 12
    m = re.search(r"FY\s*'?(\d{2})\b", t, re.I)
    if m:
        e = 2000 + int(m.group(1)); return f"FY{e-1}-{str(e)[-2:]}", e, 12
    m = re.search(r"(20\d\d)", t)
    if m:   # a bare year is read as the year the financial year ends (March year-end)
        e = int(m.group(1)); return f"FY{e-1}-{str(e)[-2:]}", e, 12
    raise ValueError(f"Cannot read a fiscal year from {text!r} (try e.g. 'FY2024-25')")


def guess_company(text: str):
    t = text.lower()
    for name, aliases in COMPANY_ALIASES.items():
        if any(re.search(rf"(?<![a-z]){re.escape(a)}(?![a-z])", t) for a in aliases):
            return name
    return None


def infer_meta(md_path: Path) -> dict:
    """Metadata for a parsed report, from its folder/file name. Raises if it can't tell."""
    p = Path(md_path)
    hay = re.sub(r"[_\-]+", " ", f"{p.parent.name} {p.stem}".replace("#U00e9", "e"))
    company = guess_company(hay) or guess_company(re.sub(r"[_\-]+", " ", p.parent.parent.name))
    if company is None:
        raise ValueError(f"Unknown company for {p}")
    stem = p.stem.lower().replace("_", "-")
    months = 12
    if company == "Nestle India" and "annual-report-2022" in stem:
        fy, end = "CY2022", 2022               # Jan-Dec 2022 (Nestle changed its year-end afterwards)
    elif company == "Nestle India" and "2023-24" in stem:
        fy, end, months = "FY2023-24", 2024, 15  # 1 Jan 2023 - 31 Mar 2024 (15-month period)
    elif re.search(r"20\d\d\s*[-–]\s*\d\d", stem):
        fy, end, months = normalize_fy(stem)
    else:   # ITC / Tata style: '...-2025' = year ended 31 March 2025
        fy, end, months = normalize_fy(stem)
    return {"company": company, "fy": fy, "period_end": end, "months": months,
            "report_id": f"{company.replace(' ', '')}_{fy}", "source_file": p.name}
