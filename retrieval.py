"""Text retrieval: dense (Qdrant) + sparse (BM25) -> QueryFusionRetriever (RRF) -> bge-reranker -> table expansion."""
import re
import time
from typing import List, Optional

from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters

import config
import stores
import timing
from corpus import COMPANY_ALIASES, normalize_fy


# ---- query scope: which companies / fiscal years does the question mention? -------------------
def _alias_map():
    amap = {c: list(a) for c, a in COMPANY_ALIASES.items()}
    for c in stores.companies():                # companies added through uploads
        amap.setdefault(c, [])
        if c.lower() not in amap[c]:
            amap[c].append(c.lower())
    return amap


_LATEST = re.compile(r"\b(latest|most recent|newest|recent annual report|last annual report)\b", re.I)
_ESG = re.compile(r"sustainab|emission|climate|brsr|\besg\b|water|biodivers|\bcsr\b|diversity|workforce|employee|human rights", re.I)
_OTHER_SECTION = re.compile(r"director|remuneration|governance|auditor|audit report|notes to|balance sheet|cash flow|annual return|"
                            r"shareholding|dividend", re.I)
_ANALYSIS = re.compile(r"\brisks?\b|outlook|strateg|reasons?|\bwhy\b|driver|challeng|opportunit|headwind|tailwind|"
                       r"competit|management (?:says|discuss|commentary)|what (?:does|did) .* (?:say|attribute)|attribut|explain", re.I)


def section_intent(query: str) -> List[str]:
    """Questions about risks, reasons, outlook or strategy live in the management-discussion, board-report and strategy
    sections. Searching the whole report lets BRSR tables and financial-statement notes crowd them out."""
    if _ESG.search(query) or _OTHER_SECTION.search(query) or not _ANALYSIS.search(query):
        return []
    return ["risk", "mdna", "board_report", "strategy"]


def _latest_fys(companies: List[str]) -> List[str]:
    best = {}
    for r in stores.load_reports().values():
        if companies and r["company"] not in companies:
            continue
        cur = best.get(r["company"])
        if cur is None or (r.get("period_end", 0), r.get("months", 0)) > (cur.get("period_end", 0), cur.get("months", 0)):
            best[r["company"]] = r
    return sorted({r["fy"] for r in best.values()})


def parse_scope(query: str) -> dict:
    q = query.lower()
    companies = []
    for name, aliases in _alias_map().items():
        if any(re.search(rf"(?<![a-z]){re.escape(a)}(?![a-z])", q) for a in aliases):
            companies.append(name)
    known_fy = {r["fy"] for r in stores.load_reports().values()}
    fys = []
    for m in re.finditer(r"(?:fy\s*'?\d{2,4}(?:\s*[-–/]\s*\d{2,4})?|20\d\d\s*[-–/]\s*\d{2,4}|cy\s*20\d\d)", q):
        try:
            fy = normalize_fy(m.group(0))[0]
        except ValueError:
            continue
        if fy in known_fy and fy not in fys:
            fys.append(fy)
    if not fys and _LATEST.search(q):          # "latest annual report" -> the newest year of each named company
        fys = _latest_fys(companies)
    return {"companies": companies, "fys": fys, "sections": section_intent(query)}


def _filters(company: Optional[str], fys: List[str], sections: Optional[List[str]] = None) -> Optional[MetadataFilters]:
    fl = []
    if company:
        fl.append(MetadataFilter(key="company", value=company, operator=FilterOperator.EQ))
    if fys:
        fl.append(MetadataFilter(key="fy", value=fys, operator=FilterOperator.IN))
    if sections:
        fl.append(MetadataFilter(key="section", value=sections, operator=FilterOperator.IN))
    return MetadataFilters(filters=fl) if fl else None


class ScopedDenseRetriever(BaseRetriever):
    """Dense retrieval from Qdrant. If the question names companies/years the search is restricted to them,
    with an equal share of the budget per company so one company cannot crowd out the others."""

    def __init__(self, embed_model, top_k: int = config.DENSE_TOP_K):
        super().__init__()
        self._embed, self._top_k = embed_model, top_k

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        timing.step("Searching the reports", query_bundle.query_str[:160])
        scope = parse_scope(query_bundle.query_str)
        if query_bundle.embedding is None:
            query_bundle.embedding = self._embed.get_query_embedding(query_bundle.query_str)
        index = VectorStoreIndex.from_vector_store(stores.vector_store(), embed_model=self._embed)
        groups = scope["companies"] or [None]
        k = max(self._top_k // len(groups), 8)
        out = []
        for c in groups:
            hits = []
            if scope["sections"]:
                hits = index.as_retriever(similarity_top_k=k, filters=_filters(c, scope["fys"], scope["sections"])).retrieve(query_bundle)
            if not hits:                                   # no passages in those sections: search the whole report
                hits = index.as_retriever(similarity_top_k=k, filters=_filters(c, scope["fys"])).retrieve(query_bundle)
            out += hits
        return out


class ScopedBM25Retriever(BaseRetriever):
    """Keyword retrieval (BM25). Same scoping rules as the dense side."""

    def __init__(self, top_k: int = config.BM25_TOP_K):
        super().__init__()
        self._top_k = top_k

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        scope = parse_scope(query_bundle.query_str)
        groups = scope["companies"] or [None]
        k = max(self._top_k // len(groups), 8)
        out = []
        for c in groups:
            bm = stores.BM25.get(c)
            if bm is None:
                continue
            bm.similarity_top_k = k * (8 if (scope["fys"] or scope["sections"]) else 1)
            hits = bm.retrieve(query_bundle)
            if scope["fys"]:
                hits = [h for h in hits if h.node.metadata.get("fy") in scope["fys"]]
            if scope["sections"]:
                in_sec = [h for h in hits if h.node.metadata.get("section") in scope["sections"]]
                hits = in_sec or hits                      # fall back to the whole report if none match
            out += hits[:k]
        return out


def build_fusion(embed_model, llm=None) -> QueryFusionRetriever:
    return QueryFusionRetriever(
        [ScopedDenseRetriever(embed_model), ScopedBM25Retriever()],
        llm=llm,
        mode="reciprocal_rerank",          # Reciprocal Rank Fusion
        similarity_top_k=config.FUSION_TOP_K,
        num_queries=1,                     # 1 = no LLM query rewriting; the chat engine already rewrites follow-ups
        use_async=False,
        retriever_weights=list(config.RRF_WEIGHTS),
    )


# ---- post-processing ------------------------------------------------------------------------
class TableExpander(BaseNodePostprocessor):
    """Table summaries are what gets embedded; once ranked we swap in the full table markdown for the LLM."""

    @classmethod
    def class_name(cls) -> str:
        return "TableExpander"

    def _postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None):
        out, seen = [], set()
        for n in nodes:
            node = n.node
            if node.metadata.get("node_kind") == "table_summary":
                full = stores.TABLES.get(node.metadata.get("table_id"))
                if full is not None:
                    node = TextNode(id_=full.node_id, text=full.text[:config.MAX_TABLE_CHARS], metadata=full.metadata)
                    node.excluded_llm_metadata_keys = list(node.metadata.keys())
            if node.node_id in seen:
                continue
            seen.add(node.node_id)
            out.append(NodeWithScore(node=node, score=n.score))
        return out


def get_reranker():
    if config.RERANK_MODEL.lower() in ("none", ""):
        return None
    try:
        from llama_index.core.postprocessor import SentenceTransformerRerank
        return SentenceTransformerRerank(model=config.RERANK_MODEL, top_n=config.RERANK_TOP_N)
    except ImportError as e:
        raise RuntimeError("bge-reranker needs `pip install sentence-transformers torch` "
                           "(or set RERANK_MODEL=none to skip reranking)") from e


QA_PROMPT = PromptTemplate(
    "You are an equity-research assistant answering from company annual reports.\n"
    "Context passages (each starts with [Company | Fiscal year | Section]):\n"
    "---------------------\n{context_str}\n---------------------\n"
    "Rules:\n"
    "- Use ONLY the context. If it does not contain the answer, say 'not found in the retrieved passages'; never guess or use outside knowledge.\n"
    "- Cover every distinct point the passages make (list all risks, drivers or reasons found), not just the first.\n"
    "- Cite every factual claim with its source tag in square brackets, copied character for character from the "
    "header of the passage the claim came from (the text in square brackets at its start). Never invent or alter a tag.\n"
    "- When a passage holds a table, report every row that bears on the question (all years shown, totals and intensity "
    "figures, not only the single number asked for). When asked how many or how often, also give the dates or names listed.\n"
    "- Copy numbers exactly as reported, with their unit (e.g. Rs crore) and period. If the periods differ in length "
    "(for example a 15-month year), say so.\n"
    "- Keep the answer focused and short.\n\n"
    "Question: {query_str}\nAnswer: ")


class TimedPostprocessor(BaseNodePostprocessor):
    """Wraps a post-processor (the reranker) and prints how long it takes."""
    inner: object
    label: str = "reranker"

    @classmethod
    def class_name(cls) -> str:
        return "TimedPostprocessor"

    def _postprocess_nodes(self, nodes, query_bundle=None):
        t = time.time()
        timing.step("Ranking the best passages")
        out = self.inner.postprocess_nodes(nodes, query_bundle=query_bundle)
        timing.log(self.label, time.time() - t, f"{len(nodes)} passages in, {len(out)} kept")
        timing.step("Writing the answer")
        return out


def build_text_engine(embed_model, llm, retriever=None, reranker="auto") -> RetrieverQueryEngine:
    post = []
    rr = get_reranker() if reranker == "auto" else reranker
    if rr is not None:
        post.append(TimedPostprocessor(inner=rr) if timing.ENABLED else rr)
    post.append(TableExpander())
    synth = get_response_synthesizer(llm=llm, response_mode=ResponseMode.COMPACT, text_qa_template=QA_PROMPT)
    return RetrieverQueryEngine(retriever=retriever or build_fusion(embed_model, llm),
                                node_postprocessors=post, response_synthesizer=synth)