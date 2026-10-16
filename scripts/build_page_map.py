"""Finds the PDF page of every passage by matching its text against the text of each PDF page (no re-parsing, no re-embedding).
Each PDF must be named after its report ID, e.g. Dabur_FY2024-25.pdf, NestleIndia_CY2022.pdf (any sub-folder is fine).

    python -m scripts.build_page_map /path/to/pdf/folder

Writes data/nodes/page_map.json  {node_id: {"pdf": relative path, "p1": first page, "p2": last page}}
and eval/page_sample.md (30 random passages to check by eye against the PDF).
"""
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import config

NODES = [config.RETRIEVAL_NODES, config.TABLE_NODES]
OUT = Path("data/nodes/page_map.json")
SAMPLE = Path("eval/page_sample.md")
N = 5                                   # words per shingle
_WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def words(text: str):
    return _WORD.findall(text.lower())


def shingles(ws):
    return {" ".join(ws[i:i + N]) for i in range(max(len(ws) - N + 1, 0))}


def page_texts(pdf: Path):
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(pdf))
    out = []
    for i in range(len(doc)):
        page = doc[i]
        tp = page.get_textpage()
        out.append(tp.get_text_range())
        tp.close()
        page.close()
    doc.close()
    return out


def clean_node_text(text: str) -> str:
    text = re.sub(r"^\[[^\]]*\]\s*", "", text)                # the [Company | FY | Section] header is not in the PDF
    return text.replace("|", " ").replace("---", " ")


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python -m scripts.build_page_map /path/to/pdf/folder")
    root = Path(sys.argv[1]).expanduser()
    pdfs = {}
    for p in root.rglob("*.pdf"):
        pdfs.setdefault(p.stem.lower(), p)
    print(f"{len(pdfs)} PDFs found under {root}")
    nodes = defaultdict(list)                                  # source stem -> [(id, text, kind)]
    for f in NODES:
        if not f.exists():
            continue
        for line in open(f, encoding="utf-8"):
            d = json.loads(line)
            stem = d["metadata"].get("report_id", Path(d["metadata"]["source_file"]).stem).lower()
            nodes[stem].append((d["id"], d["text"], d["metadata"].get("node_kind", "")))
    page_map, missing = {}, []
    for stem, items in sorted(nodes.items()):
        pdf = pdfs.get(stem)
        if pdf is None:
            missing.append(stem)
            continue
        print(f"{pdf.name}: reading pages …", end=" ", flush=True)
        pages = page_texts(pdf)
        npg = max(len(pages), 1)
        tok_pages, tri_pages = defaultdict(set), defaultdict(set)     # word / 3-word run -> pages that contain it
        for pno, t in enumerate(pages, 1):
            ws = words(t)
            for w in set(ws):
                tok_pages[w].add(pno)
            for i in range(len(ws) - 2):
                tri_pages[(ws[i], ws[i + 1], ws[i + 2])].add(pno)
        done = 0
        for nid, text, kind in items:
            ws = words(clean_node_text(text))
            score, hits = Counter(), Counter()
            for w in set(ws):                                  # rare words and numbers: independent of the column order of the page
                ps = tok_pages.get(w)
                if ps and len(ps) <= 12:
                    wt = math.log(npg / len(ps))
                    for p in ps:
                        score[p] += wt
                        hits[p] += 1
            for i in range(len(ws) - 2):                       # short phrases that survive in the page text count extra
                ps = tri_pages.get((ws[i], ws[i + 1], ws[i + 2]))
                if ps and len(ps) <= 4:
                    for p in ps:
                        score[p] += 1.5 / len(ps)
            if not score:
                continue
            best, bs = score.most_common(1)[0]
            if bs < 12 or hits[best] < 3:                      # too little evidence: no link rather than a wrong one
                continue
            near = [p for p, sc in score.items() if sc >= 0.7 * bs and abs(p - best) <= 2]
            page_map[nid] = {"pdf": str(pdf.relative_to(root)), "p1": min(near), "p2": max(near)}
            done += 1
        print(f"{done}/{len(items)} passages placed")
    OUT.write_text(json.dumps({"root": str(root), "map": page_map}), encoding="utf-8")
    total = sum(len(v) for v in nodes.values())
    ranges = sum(1 for v in page_map.values() if v["p1"] != v["p2"])
    print(f"\nplaced {len(page_map)} of {total} passages ({100 * len(page_map) / max(total, 1):.1f}%); {ranges} shown as a page range")
    if missing:
        print("No PDF found for:", ", ".join(missing))
    texts = {nid: t for items in nodes.values() for nid, t, _ in items}
    meta = {}
    for f in NODES:
        if f.exists():
            for line in open(f, encoding="utf-8"):
                d = json.loads(line)
                meta[d["id"]] = d["metadata"]
    random.seed(1)
    pick = random.sample(sorted(page_map), min(30, len(page_map)))
    SAMPLE.parent.mkdir(exist_ok=True)
    lines = ["# Page check: open each PDF at the page and see whether the passage is there\n"]
    for i, nid in enumerate(pick, 1):
        m, pm = meta[nid], page_map[nid]
        pg = pm["p1"] if pm["p1"] == pm["p2"] else f"{pm['p1']}-{pm['p2']}"
        lines.append(f"{i}. **{m['company']} {m['fy']}**, `{pm['pdf']}`, page **{pg}**\n   > {re.sub(chr(92) + 's+', ' ', clean_node_text(texts[nid]))[:160]}…\n")
    SAMPLE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Sample to check by eye: {SAMPLE}")


if __name__ == "__main__":
    main()