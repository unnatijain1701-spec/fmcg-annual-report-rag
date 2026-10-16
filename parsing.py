"""PDF -> markdown with Docling. The same function serves the original corpus and every upload."""
from pathlib import Path

_converter = None


def _get_converter():
    global _converter
    if _converter is None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        opts = PdfPipelineOptions()
        opts.do_ocr = False                 # annual reports are digital PDFs; OCR only slows things down
        opts.do_table_structure = True      # keep tables as markdown tables
        _converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    return _converter


def pdf_to_markdown(pdf_path, out_md_path, progress=print) -> Path:
    pdf_path, out_md_path = Path(pdf_path), Path(out_md_path)
    progress(f"Docling: converting {pdf_path.name} (a 300-page report takes several minutes)")
    result = _get_converter().convert(str(pdf_path))
    md = result.document.export_to_markdown()          # images become <!-- image --> placeholders
    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    out_md_path.write_text(md, encoding="utf-8")
    progress(f"Docling: wrote {out_md_path.name} ({len(md) / 1e6:.1f} MB)")
    return out_md_path
