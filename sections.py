"""Heuristic section tagging for Docling markdown.

Annual reports repeat a running page header ("Report on Corporate Governance", ...) on every page, and Docling turns
those into markdown headings. We walk the headings, keep the "current section", and cut the document into contiguous
segments that each carry one section label. Labels decide how a segment is chunked (semantic vs. sentence splitting).
"""
import re
from dataclasses import dataclass

# (label, regex on heading text). First match wins. 'soft' labels never override a narrative (mdna/risk) section.
RULES = [
    ("risk", r"risks? (and|&) (concerns|opportunities|mitigation|uncertaint)|key risks|principal risks|material risks|"
             r"risk factors|risk management (framework|and internal|policy|process)|enterprise risk|risks? and (their )?mitigation"),
    ("mdna", r"management discussion|\bmd&a\b|md&amp;a|business review|segment review|operating (and|&|&amp;) financial review|"
             r"economic (and|&|&amp;) industry|industry (structure|overview|review)|opportunities and threats|"
             r"financial (performance|review)|performance review|operational review|business performance|outlook"),
    ("brsr", r"business responsibility|brsr|sustainability report"),
    ("governance", r"corporate governance|report on governance"),
    ("auditor", r"independent auditor|auditor'?s report|secretarial audit"),
    ("financials", r"balance sheet|statement of (profit|cash|changes)|notes to (the )?(standalone|consolidated|financial)|"
                   r"standalone financial|consolidated financial|financial statements"),
    ("notice", r"notice (of|is hereby)|annual general meeting"),
    ("board_report", r"board'?s report|report of the board|directors'? report|report of the directors"),
    ("strategy", r"chairman|managing director|letter to (the )?(shareholders|stakeholders)|our strategy|business model|value creation"),
]
SOFT = {"board_report", "strategy", "notice"}
NARRATIVE = {"mdna", "risk"}
_compiled = [(lab, re.compile(rx, re.I)) for lab, rx in RULES]
_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def classify_heading(text: str):
    for lab, rx in _compiled:
        if rx.search(text):
            return lab
    return None


@dataclass
class Segment:
    section: str
    heading: str       # most recent heading text at the start of the segment
    text: str


def segment_markdown(md: str, min_chars: int = 200):
    """Split markdown into contiguous segments with a section label each."""
    cur_label, cur_head = "other", ""
    buf, segs = [], []

    def flush():
        txt = "\n".join(buf).strip()
        if txt:
            segs.append(Segment(cur_label, cur_head, txt))
        buf.clear()

    for line in md.split("\n"):
        m = _HEADING.match(line)
        if m:
            head = m.group(2)
            lab = classify_heading(head)
            if lab and lab != cur_label:
                soft_blocked = lab in SOFT and cur_label in NARRATIVE
                if not soft_blocked:
                    flush()
                    cur_label = lab
            cur_head = head
        buf.append(line)
    flush()
    # merge tiny segments into the previous one (keeps chunk context together)
    merged = []
    for s in segs:
        if merged and len(s.text) < min_chars:
            merged[-1].text += "\n" + s.text
        else:
            merged.append(s)
    return merged
