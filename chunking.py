"""Chunking + metadata.

* Tables      -> MarkdownElementNodeParser  (table summary node for retrieval + full table node kept aside)
* Narrative   -> SemanticSplitterNodeParser (MD&A, risk sections) / SentenceSplitter (everything else)

MarkdownElementNodeParser accepts a `nested_node_parser` for the text between tables, so the same parser instance
does both jobs: tables become summary+table node pairs, narrative goes to the nested splitter.
"""
import html
import json
import re
import uuid
from typing import List, Tuple

from llama_index.core import Document
from llama_index.core.node_parser import (MarkdownElementNodeParser, SemanticSplitterNodeParser,
                                          SentenceSplitter)
from llama_index.core.node_parser.relational.base_element import TableColumnOutput, TableOutput
from llama_index.core.schema import IndexNode, TextNode

import config
from sections import segment_markdown

SECTION_NAMES = {"mdna": "MD&A", "risk": "Risk", "brsr": "BRSR", "governance": "Governance",
                 "auditor": "Auditor report", "financials": "Financial statements", "notice": "AGM notice",
                 "board_report": "Board report", "strategy": "Strategy", "other": "Other"}


def clean_md(md: str) -> str:
    md = html.unescape(md)
    md = md.replace("<!-- image -->", "")
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md


def det_id(report_id: str, kind: str, i: int) -> str:
    """Deterministic UUID so re-running the pipeline overwrites instead of duplicating."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{report_id}|{kind}|{i}"))


class RuleSummaryElementParser(MarkdownElementNodeParser):
    """MarkdownElementNodeParser whose table summaries are built from the table itself (no LLM call per table).

    7,000+ tables per corpus would otherwise mean 7,000+ LLM calls. The summary names the section heading, the
    column headers (periods/units) and the row labels, which is what dense and BM25 retrieval need to find a table.
    Set TABLE_SUMMARY_MODE=llm to use the stock LLM summariser instead (slow, optional).
    """

    @classmethod
    def class_name(cls) -> str:
        return "RuleSummaryElementParser"

    @staticmethod
    def _title_before(elements, idx: int) -> str:
        for j in range(idx - 1, max(idx - 3, -1), -1):
            if elements[j].type == "text":
                lines = [l.strip() for l in str(elements[j].element).split("\n") if l.strip()]
                for l in reversed(lines[-6:]):
                    if len(l) <= 160:
                        return l.lstrip("# ").strip()
        return ""

    def extract_table_summaries(self, elements) -> None:
        for idx, el in enumerate(elements):
            if el.type not in ("table", "table_text"):
                continue
            title = self._title_before(elements, idx)
            cols, labels = [], []
            if el.type == "table" and getattr(el, "table", None) is not None:
                df = el.table
                cols = [str(c).strip() for c in df.columns if str(c).strip() and not str(c).startswith("Unnamed")]
                if len(df.columns):
                    seen = set()
                    for v in df.iloc[:, 0].tolist():
                        v = str(v).strip()
                        if v and v.lower() != "nan" and v not in seen:
                            seen.add(v); labels.append(v[:70])
            summary = "Table"
            if title:
                summary += f" under '{title}'"
            if labels:
                summary += ". Rows: " + "; ".join(labels[:14])
            el.table_output = TableOutput(
                summary=summary, table_title=title or None,
                columns=[TableColumnOutput(col_name=c[:80], col_type="text", summary="") for c in cols[:10]])

    async def aextract_table_summaries(self, elements) -> None:
        self.extract_table_summaries(elements)


def build_parsers(embed_model):
    semantic = SemanticSplitterNodeParser(
        buffer_size=config.SEMANTIC_BUFFER_SIZE,
        breakpoint_percentile_threshold=config.SEMANTIC_BREAKPOINT_PERCENTILE,
        embed_model=embed_model)
    sentence = SentenceSplitter(chunk_size=config.SENTENCE_CHUNK_TOKENS, chunk_overlap=config.SENTENCE_CHUNK_OVERLAP)
    cls = RuleSummaryElementParser if config.TABLE_SUMMARY_MODE == "rules" else MarkdownElementNodeParser
    return {"semantic": cls(nested_node_parser=semantic, show_progress=False),
            "sentence": cls(nested_node_parser=sentence, show_progress=False)}, sentence


def _header(meta: dict, section: str) -> str:
    return f"[{meta['company']} | {meta['fy']} | {SECTION_NAMES.get(section, section)}]\n"


def chunk_report(md_text: str, meta: dict, parsers: dict, resplitter: SentenceSplitter,
                 progress=None) -> Tuple[List[TextNode], List[TextNode]]:
    """Return (retrieval_nodes, table_nodes) for one report.

    retrieval_nodes : narrative TextNodes + table-summary IndexNodes (these get embedded and BM25-indexed)
    table_nodes     : full table TextNodes, keyed by the IndexNode's index_id (looked up after reranking)
    """
    segments = segment_markdown(clean_md(md_text))
    retrieval, tables = [], []
    n_nar = n_tab = 0
    for si, seg in enumerate(segments):
        kind = "semantic" if seg.section in config.SEMANTIC_SECTIONS else "sentence"
        base_meta = {"company": meta["company"], "fy": meta["fy"], "period_end": meta["period_end"],
                     "months": meta["months"], "report_id": meta["report_id"], "section": seg.section,
                     "source_file": meta["source_file"]}
        doc = Document(text=seg.text, metadata=dict(base_meta))
        parser = parsers[kind]
        nodes = parser.get_nodes_from_documents([doc])
        base_nodes, mappings = parser.get_base_nodes_and_mappings(nodes)
        hdr = _header(meta, seg.section)
        for node in base_nodes:
            if isinstance(node, IndexNode):
                full = mappings.get(node.index_id)
                if full is None:
                    continue
                n_tab += 1
                tid = det_id(meta["report_id"], "tbl", n_tab)
                summ = TextNode(id_=det_id(meta["report_id"], "tsum", n_tab), text=hdr + node.text,
                                metadata={**base_meta, "node_kind": "table_summary", "table_id": tid})
                full_md = full.text
                tables.append(TextNode(id_=tid, text=hdr + full_md,
                                       metadata={**base_meta, "node_kind": "table", "table_id": tid,
                                                 "heading": seg.heading}))
                retrieval.append(summ)
            else:
                text = node.get_content(metadata_mode="none").strip()
                pieces = [text]
                if len(text) > config.MAX_CHUNK_CHARS:
                    pieces = [n.text for n in resplitter.get_nodes_from_documents([Document(text=text)])]
                for p in pieces:
                    p = p.strip()
                    if len(p) < config.MIN_CHUNK_CHARS:
                        continue
                    n_nar += 1
                    retrieval.append(TextNode(id_=det_id(meta["report_id"], "txt", n_nar), text=hdr + p,
                                              metadata={**base_meta, "node_kind": "narrative"}))
        if progress and (si % 25 == 0):
            progress(f"{meta['report_id']}: segment {si + 1}/{len(segments)}")
    for n in retrieval + tables:        # the [Company | FY | Section] header is already in the text
        n.excluded_embed_metadata_keys = list(n.metadata.keys())
        n.excluded_llm_metadata_keys = list(n.metadata.keys())
    return retrieval, tables


# ---- persistence -------------------------------------------------------------------------------
def node_to_row(n: TextNode) -> dict:
    return {"id": n.node_id, "text": n.text, "metadata": n.metadata}


def row_to_node(r: dict) -> TextNode:
    n = TextNode(id_=r["id"], text=r["text"], metadata=r["metadata"])
    n.excluded_embed_metadata_keys = list(n.metadata.keys())
    n.excluded_llm_metadata_keys = list(n.metadata.keys())
    return n


def append_jsonl(path, nodes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for n in nodes:
            f.write(json.dumps(node_to_row(n), ensure_ascii=False) + "\n")


def read_jsonl(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [row_to_node(json.loads(l)) for l in f if l.strip()]


def drop_report(path, report_id: str):
    """Remove all nodes of one report from a jsonl file (used when a report is re-ingested)."""
    if not path.exists():
        return
    keep = []
    with open(path, encoding="utf-8") as f:
        for l in f:
            if l.strip() and json.loads(l)["metadata"].get("report_id") != report_id:
                keep.append(l)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(keep)
