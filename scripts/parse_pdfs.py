"""Optional: convert PDFs with Docling into data/processed/<company>/<report>/<report>.md (same layout as the supplied files).

    python -m scripts.parse_pdfs path/to/pdf_folder
"""
import sys
from pathlib import Path

import config
from corpus import guess_company
from parsing import pdf_to_markdown

for pdf in sorted(Path(sys.argv[1]).rglob("*.pdf")):
    company = (guess_company(f"{pdf.parent.name} {pdf.stem}".replace("_", " ").replace("-", " ")) or "unknown").replace(" ", "_")
    out = config.PROCESSED_DIR / company / pdf.stem / f"{pdf.stem}.md"
    if out.exists():
        print("skip", out)
        continue
    pdf_to_markdown(pdf, out)
