"""Assembles the full pipeline:

   chat history -> CondenseQuestionChatEngine (rewrites follow-ups, ChatMemoryBuffer)
                -> VerifiedQueryEngine (FaithfulnessEvaluator + revise prompt)
                   -> SubQuestionQueryEngine (decompose, route each sub-question)
                        |- annual_report_text : hybrid retrieval -> RRF -> bge-reranker -> table expansion -> gemma
                        '- financials_sql      : qwen2.5-coder -> DuckDB
"""
import re
from typing import Any, List, Optional, Sequence

from llama_index.core import get_response_synthesizer
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import Response
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import CustomQueryEngine, SubQuestionQueryEngine
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core.question_gen.types import SubQuestion
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.schema import QueryBundle
from llama_index.core.tools import QueryEngineTool, ToolMetadata

import config
import financials
import ranking
import timing
import retrieval
import stores
from models import setup_models

TEXT_TOOL = "annual_report_text"
SQL_TOOL = "financials_sql"

TEXT_DESC = ("Qualitative and narrative content of the annual reports: strategy, management discussion, outlook, risks, "
             "brands, segments commentary, governance, sustainability, reasons behind performance, and tables read as text. "
             "Ask about ONE company (and ideally one fiscal year) per sub-question.")
SQL_DESC = ("Structured financial figures in a database: revenue, profit, EBITDA, margins, EPS, dividends, assets, debt, cash flow, "
            "growth rates, CAGR, rankings and comparisons across companies and years. Use for any question that needs a number, "
            "calculation or statistic. Companies and fiscal years can be compared in one sub-question. "
            "Margins, ratios, growth rates and CAGR are stored as ready-made figures: ask for the margin or ratio ITSELF, for all "
            "companies and years in ONE sub-question. Never ask for two raw amounts (for example EBITDA and revenue) to divide.")

FINAL_PROMPT = PromptTemplate(
    "You are an equity-research assistant. Below are answers to sub-questions, each drawn from company annual reports "
    "or a financial database.\n---------------------\n{context_str}\n---------------------\n"
    "Write the final answer to the original question using ONLY these sub-answers. "
    "Do not add facts, numbers or causes that are not in them.\n\n"
    "FORMAT:\n"
    "- Start with ONE sentence that directly answers the question. If the measures asked about move in different directions "
    "(for example gross and net, or two segments), say so in that sentence instead of one overall direction.\n"
    "- Then give the supporting detail: at most 4 bullets (one per fiscal year if the question is year by year). "
    "Each bullet starts with a bold label of 2-4 words, then ONE short sentence of at most 25 words with the key number or fact.\n"
    "- Use a small markdown table instead of bullets when comparing companies or years on the same measure "
    "(columns such as Company, Basis, and the value).\n"
    "- No introduction and no closing summary paragraph.\n\n"
    "SOURCE TAGS:\n"
    "- Put ONE source tag at the end of each bullet or table row, copied character for character from the sub-answers; never alter a tag. "
    "Never put a tag in the middle of a sentence and never repeat the same tag within a bullet.\n"
    "- For figures from the database the tag is exactly [Financial database]. "
    "Never write tags such as [source: ...] or [Structured figures from DuckDB].\n\n"
    "NUMBERS:\n"
    "- Never calculate a margin, ratio or growth rate yourself. Use the figure the database returned for that measure. "
    "If a sub-answer holds only raw amounts, write that the ratio is not available instead of dividing figures; "
    "never divide figures from different companies, years or bases.\n"
    "- For a ranking, list the companies in the exact order of the database result (it is already sorted) and label them "
    "1st, 2nd, 3rd...; every value must be lower than the one above it.\n"
    "- For a comparison, the first sentence must say which is higher or lower and the basis of each figure. When the companies' "
    "main figures are on different bases (one standalone, one consolidated), add one line: 'Not like-for-like: ...'.\n"
    "- When you use a table, do not repeat the same figures in bullets or in a second sentence.\n"
    "- If a database result has an 'OTHER BASIS ALSO IN THE DATABASE' list, show those figures next to the main value "
    "(for example 'HUL 21.94% consolidated, 21.50% standalone'). If it says only one basis exists for a company, say so. "
    "If it warns that companies are on different bases, say so in one line and do not call the gap like-for-like.\n"
    "- If a figure is for a different period than the one asked, say which period it is for; never present it as the asked period.\n"
    "- State the unit and the period of every figure (Rs crore, %, Rs per share). Use thousands separators (81,613) "
    "and show percentages with a % sign, rounded to at most 2 decimals.\n"
    "- If a figure has both a standalone and a consolidated value, label each; never list two values for one company "
    "without saying why. If companies are reported on different bases, say so in one line.\n\n"
    "MISSING INFORMATION:\n"
    "- Where a text sub-answer lacks something, write 'not found in the retrieved passages' (never 'the company does not "
    "mention it'): the passages are only part of the report.\n"
    "- Where a database result has a NOTE saying a company has no figures, repeat that note "
    "(say 'no figures available in the database for <company>', not 'not found in the retrieved passages').\n"
    "- Put everything that is missing in ONE final line, not as separate bullets.\n\n"
    "Question: {query_str}\nAnswer: ")


# ---- tidy-up of the model's final text ----------------------------------------------------------
_DB_TAG = re.compile(r"\[(?:source:\s*)?(?:structured figures from duckdb|financial database|duckdb|database)[^\]]*\]", re.I)
_SRC_TAG = re.compile(r"\[source:\s*([^\]]+)\]", re.I)
_BASIS_TAG = re.compile(r"\[[^\[\]|]+\|\s*(?:FY|CY)[\w\-]*\s*\|\s*(standalone|consolidated)(?:\s*\|[^\[\]]*)?\]", re.I)
_REPEAT_TAG = re.compile(r"(\[[^\]]+\])(?:\s*(?:,|;|and)?\s*\1)+")
_LEAD_IN = re.compile(r"^\s*(?:based on|according to|from) the (?:provided |retrieved |available )?"
                      r"(?:financial )?(?:data(?:base)?|sub-answers?|information|passages|context)[^,:\n]*[,:]\s*", re.I)


def clean_answer(text: str) -> str:
    """Make small-model output consistent: one tag style, no repeated tags, no filler lead-in."""
    t = (text or "").strip()
    t = re.sub(r"^\s*answer:\s*", "", t, flags=re.I)
    t = _DB_TAG.sub("[Financial database]", t)
    t = _BASIS_TAG.sub(lambda m: f"[Financial database, {m.group(1).lower()}]", t)   # a tag carrying a basis is a database figure
    t = _SRC_TAG.sub(r"[\1]", t)
    t = _REPEAT_TAG.sub(r"\1", t)
    t = _LEAD_IN.sub("", t, count=1)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return (t[:1].upper() + t[1:]) if t else t


CONDENSE_PROMPT = PromptTemplate(
    "Given the conversation below and a follow-up message, rewrite the follow-up as ONE standalone question. "
    "Replace pronouns with the company, metric and fiscal year they refer to, and keep company names and years explicit. "
    "If the follow-up is already standalone, return it unchanged.\n\n"
    "<Chat History>\n{chat_history}\n\n<Follow Up Message>\n{question}\n\n<Standalone question>\n")

REVISE_PROMPT = PromptTemplate(
    "You are fact-checking a draft answer against the evidence it was written from.\n\n"
    "EVIDENCE:\n{evidence}\n\nQUESTION: {query}\n\nDRAFT ANSWER:\n{answer}\n\n"
    "Task: remove or rewrite every statement in the draft that the evidence does not support (figures, causes, comparisons, "
    "names, years). Keep supported statements and their [source tags] exactly as they are. If something the question asks "
    "for cannot be supported, say 'Not found in the retrieved passages.' Do not add any new fact. "
    "Return only the corrected answer.")


class SafeQuestionGenerator(LLMQuestionGenerator):
    """Default LLM question generator, with a safe fallback if the model returns malformed JSON."""

    @staticmethod
    def _fallback(query: QueryBundle) -> List[SubQuestion]:
        numeric = any(w in query.query_str.lower() for w in
                      ("revenue", "profit", "margin", "growth", "cagr", "ebitda", "eps", "dividend", "ratio", "how much", "highest", "lowest", "compare"))
        return [SubQuestion(sub_question=query.query_str, tool_name=SQL_TOOL if numeric else TEXT_TOOL)]

    @staticmethod
    def _merge_text(subs: List[SubQuestion]) -> List[SubQuestion]:
        """Text sub-questions about the same company and year share ONE search (each search costs a full rerank, ~15 s).
        Different years / companies stay separate, so every year is still searched with its own filter."""
        out, groups = [], {}
        for sq in subs:
            if sq.tool_name != TEXT_TOOL:
                out.append(sq)
                continue
            try:
                sc = retrieval.parse_scope(sq.sub_question)
                key = (tuple(sc["companies"]), tuple(sc["fys"]))
            except Exception:
                key = (sq.sub_question,)
            if key in groups:
                groups[key].sub_question += " Also: " + sq.sub_question
            else:
                groups[key] = SubQuestion(sub_question=sq.sub_question, tool_name=sq.tool_name)
                out.append(groups[key])
        return out

    _MULTI_YEAR = re.compile(r"previous|prior|last year|year[- ]on[- ]year|yoy|growth|cagr|trend|chang|"
                             r"between|since|each year|across|histor|declin|increas|decreas|earlier|before|trajectory", re.I)

    @classmethod
    def _pin_year(cls, subs: List[SubQuestion], query: QueryBundle) -> List[SubQuestion]:
        """If the user asked about ONE year (and nothing that needs another), every sub-question must use that year.
        The splitting model sometimes drops it or invents another (HUL margin was fetched for FY2022-23)."""
        try:
            q = query.query_str
            fys = retrieval.parse_scope(q)["fys"]
            if len(fys) != 1 or cls._MULTI_YEAR.search(q):
                return subs
            fy = fys[0]
            for sq in subs:
                sc = retrieval.parse_scope(sq.sub_question)["fys"]
                if sc == [fy]:
                    continue
                if sc:
                    sq.sub_question = re.sub(r"(?:fy\s*'?\d{2,4}(?:\s*[-\u2013/]\s*\d{2,4})?|20\d\d\s*[-\u2013/]\s*\d{2,4})",
                                             fy, sq.sub_question, flags=re.I)
                else:
                    sq.sub_question = f"{sq.sub_question.rstrip(' ?.')} for {fy}"
        except Exception:
            pass
        return subs

    _ALL = re.compile(r"\b(all|every|each)\b[^.?]{0,25}\b(compan|firm|player)|\bwhich\s+(?:company|firm|companies)\b", re.I)
    _PLACEHOLDER = re.compile(r"\bcompany\s+[A-Z]\b|\bcompany\s+(?:one|two|three|four|five|x|y|z)\b", re.I)

    @classmethod
    def _name_companies(cls, query: QueryBundle) -> QueryBundle:
        """'all five companies' does not name them, so the splitting model invented 'Company A..E'. Name them."""
        try:
            q = query.query_str
            if cls._ALL.search(q) and not retrieval.parse_scope(q)["companies"]:
                return QueryBundle(f"{q} (the companies are: {', '.join(stores.companies())})")
        except Exception:
            pass
        return query

    @classmethod
    def _fix_placeholders(cls, subs: List[SubQuestion], named: QueryBundle) -> List[SubQuestion]:
        bad = [sq for sq in subs if cls._PLACEHOLDER.search(sq.sub_question)]
        if not bad:
            return subs
        keep = [sq for sq in subs if sq not in bad]
        return keep + [SubQuestion(sub_question=named.query_str, tool_name=bad[0].tool_name)]

    def generate(self, tools: Sequence[ToolMetadata], query: QueryBundle) -> List[SubQuestion]:
        named = self._name_companies(query)
        timing.step("Splitting the question into sub-questions")
        try:
            subs = super().generate(tools, named)
            out = self._merge_text(self._pin_year(self._fix_placeholders(subs, named), query)) if subs else self._fallback(named)
        except Exception:
            out = self._fallback(named)
        timing.note(" | ".join(s.sub_question for s in out)[:400])
        return out

    async def agenerate(self, tools: Sequence[ToolMetadata], query: QueryBundle) -> List[SubQuestion]:
        named = self._name_companies(query)
        timing.step("Splitting the question into sub-questions")
        try:
            subs = await super().agenerate(tools, named)
            out = self._merge_text(self._pin_year(self._fix_placeholders(subs, named), query)) if subs else self._fallback(named)
        except Exception:
            out = self._fallback(named)
        timing.note(" | ".join(s.sub_question for s in out)[:400])
        return out


def collect_contexts(response: Response) -> List[str]:
    """Raw evidence behind an answer (retrieved passages, SQL result tables), not the intermediate sub-answers."""
    nodes = list(response.source_nodes or [])
    raw = [n for n in nodes if not n.node.get_content().startswith("Sub question:")]
    chosen = raw or nodes
    chosen.sort(key=lambda n: (n.node.metadata.get("node_kind") != "sql_result", -(n.score or 0)))
    return [n.node.get_content(metadata_mode="none")[:config.VERIFY_CONTEXT_CHARS] for n in chosen[:config.VERIFY_MAX_CONTEXTS]]


_ANY_TAG = re.compile(r"\[[^\[\]|]+\|[^\[\]]+\]")


def database_only(response: Response) -> bool:
    """True when every piece of evidence behind the answer is a database result (no report passages)."""
    raw = [n for n in (response.source_nodes or []) if not n.node.get_content().startswith("Sub question:")]
    return bool(raw) and all(n.node.metadata.get("node_kind") == "sql_result" for n in raw)


def retag_database(text: str) -> str:
    """Figures that came only from the database are tagged [Financial database], whatever tag the model wrote."""
    return _REPEAT_TAG.sub(r"\1", _ANY_TAG.sub("[Financial database]", text))


_DEPENDENT = re.compile(
    r"^(?P<a>.*?\b(?:highest|lowest|best|worst|top|largest|biggest|smallest|fastest|slowest)\b[^?]*?)[,;]?\s+"
    r"(?:and|then)\s+(?P<b>(?:what|why|how)\b[^?]*)\??\s*$", re.I)


def _winner_of(resp: Response) -> Optional[str]:
    """First company in the first database result of a ranking answer."""
    import io
    import pandas as pd
    for n in resp.source_nodes or []:
        if n.node.metadata.get("node_kind") == "sql_result":
            try:
                df = pd.read_json(io.StringIO(n.node.metadata["chart_table"]), orient="split")
                if "company" in df.columns and len(df):
                    return str(df.iloc[0]["company"])
            except Exception:
                continue
    return None


def basis_line(response: Response) -> str:
    """One exact line saying which companies are on which basis, built from the database rows themselves (the model garbled it)."""
    import io
    import pandas as pd
    by = {}
    for n in response.source_nodes or []:
        if n.node.metadata.get("node_kind") != "sql_result":
            continue
        try:
            df = pd.read_json(io.StringIO(n.node.metadata["chart_table"]), orient="split")
        except Exception:
            continue
        if "company" in df.columns and "basis" in df.columns:
            for comp, b in zip(df["company"].astype(str), df["basis"].astype(str)):
                if b in ("standalone", "consolidated"):
                    by.setdefault(b, set()).add(comp)
    if len(by) < 2:
        return ""
    return "Not like-for-like: " + "; ".join(f"{b}: {', '.join(sorted(c))}" for b, c in sorted(by.items())) + "."


def apply_basis_line(text: str, response: Response) -> str:
    line = basis_line(response)
    if not line:
        return text
    # remove only the model's own "Not like-for-like: ..." sentence at the end of a line; never a whole paragraph that merely mentions it
    kept = [re.sub(r"\s*\bnot like-for-like\s*:.*$", "", l, flags=re.I) for l in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).rstrip() + "\n\n" + line


_DUPONT = re.compile(r"du\s?pont|decompos\w*\s+(?:of\s+)?(?:roe|return on equity)", re.I)


def dupont_response(query_str: str) -> Optional[Response]:
    """DuPont split worked out in code from the database (net margin x asset turnover x equity multiplier)."""
    from llama_index.core.schema import NodeWithScore, TextNode
    import ratios
    scope = retrieval.parse_scope(query_str)
    comps = scope["companies"] or ratios.all_companies()
    period = scope["fys"][0] if scope["fys"] else ratios.latest_period(comps[0])
    df = ratios.dupont(comps, period)
    if df.empty:
        return None
    lines = [f"DuPont split of return on equity, {period} (year-end balances; calculated by the system from the financial database):", ""]
    cols = ["net_margin_pct", "asset_turnover_x", "equity_multiplier_x", "roe_dupont_pct", "roe_reported_pct", "roce_reported_pct"]
    names = ["Net margin %", "Asset turnover x", "Equity multiplier x", "ROE by DuPont %", "ROE reported %", "ROCE reported %"]
    lines += ["| Company | Basis | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    notes = []
    for _, r in df.iterrows():
        lines.append(f"| {r['company']} | {r.get('basis', '')} | " + " | ".join("" if c not in df.columns or r.get(c) != r.get(c) else f"{r[c]:.2f}" for c in cols) + " |")
        if str(r.get("note", "")) not in ("", "year-end balances", "nan"):
            notes.append(f"{r['company']}: {r['note']}")
    lines += ["", "ROE by DuPont = net margin x asset turnover x equity multiplier. It uses year-end balances, so it can differ from the reported ROE, which companies usually compute on average equity."]
    if notes:
        lines += ["", "Data notes: " + "; ".join(notes) + "."]
    lines += ["", "[Financial database]"]
    text = "\n".join(lines)
    node = TextNode(text=text, metadata={"node_kind": "sql_result", "sql": "DuPont (calculated)", "chart_table": df.head(120).to_json(orient="split", date_format="iso")})
    node.excluded_llm_metadata_keys = ["node_kind", "sql", "chart_table"]
    node.excluded_embed_metadata_keys = ["node_kind", "sql", "chart_table"]
    return Response(response=text, source_nodes=[NodeWithScore(node=node, score=1.0)], metadata={})


class VerifiedQueryEngine(CustomQueryEngine):
    """Runs the inner engine, checks the answer with FaithfulnessEvaluator and, if it fails, makes one targeted revision
    (a single LLM call over the draft and its evidence; the retrieval pipeline is NOT re-run)."""
    inner: BaseQueryEngine
    llm: Any
    evaluator: Any
    enabled: bool = True
    text: Any = None

    def _run(self, query_str: str) -> Response:
        timing.step("Understanding the question")
        """'Which company had the highest X, and why?': answer the ranking first, then search the reasons for that company only
        (otherwise the reasons are searched before the winner is known)."""
        if _DUPONT.search(query_str):
            try:
                r = dupont_response(query_str)
                if r is not None:
                    return r
            except Exception:
                pass
        if ranking.is_stat(query_str):                       # average / median / range / standard deviation of one measure
            try:
                timing.step("Calculating the statistic from the financial database")
                r = ranking.stat_response(query_str)
                if r is not None:
                    return r
            except Exception:
                pass
        if ranking.is_ranking(query_str) and not _DEPENDENT.match(query_str.strip()):                    # plain "which company / rank by X": straight from the database
            try:
                timing.step("Reading the ranking from the financial database")
                r = ranking.rank_response(query_str)
                if r is not None:
                    return r
            except Exception:
                pass
        m = _DEPENDENT.match(query_str.strip())
        if m and not retrieval.parse_scope(m.group("b"))["companies"]:       # the second part must not already name a company
            try:
                a, b = m.group("a").strip(), m.group("b").strip()
                resp_a = ranking.rank_response(a) or self.inner.query(a)
                winner = _winner_of(resp_a)
                if winner:
                    fys = retrieval.parse_scope(a)["fys"]
                    q_b = f"For {winner}{' in ' + fys[0] if fys else ''}, {b} (context: {a})"
                    if self.text is not None and not re.search(r"\b(how much|what was|what is)\b.*\b(revenue|profit|margin|roce|roe|eps|dividend|ebitda)\b", b, re.I):
                        resp_b = self.text.query(f"For {winner}{' in ' + fys[0] if fys else ''}, {b}")   # narrative part: straight to the text index
                    else:
                        resp_b = self.inner.query(q_b)
                    text = clean_answer(str(resp_a)) + "\n\n" + clean_answer(str(resp_b))
                    nodes = list(resp_a.source_nodes or []) + list(resp_b.source_nodes or [])
                    return Response(response=text, source_nodes=nodes, metadata={})
            except Exception:
                pass
        return self.inner.query(query_str)

    def custom_query(self, query_str: str) -> Response:
        resp = self._run(query_str)
        meta = resp.metadata if isinstance(resp.metadata, dict) else {}
        resp.metadata = meta
        answer = clean_answer(str(resp))
        if database_only(resp):
            answer = retag_database(answer)
        answer = apply_basis_line(answer, resp)
        resp.response = answer
        contexts = collect_contexts(resp)
        if not self.enabled:
            meta["verification"] = {"status": "disabled"}
            return resp
        if not contexts:
            meta["verification"] = {"status": "no_evidence"}
            return resp
        timing.step("Checking the answer against the sources")
        try:
            ev = self.evaluator.evaluate(query=query_str, response=answer, contexts=contexts)
        except Exception as e:
            meta["verification"] = {"status": "error", "detail": str(e)[:200]}
            return resp
        if ev.passing:
            meta["verification"] = {"status": "verified", "draft": answer}
            timing.finish()
            return resp
        timing.step("Correcting unsupported statements")
        revised = self.llm.complete(REVISE_PROMPT.format(
            evidence="\n\n---\n".join(contexts), query=query_str, answer=answer)).text.strip()
        meta["verification"] = {"status": "revised", "draft": answer, "feedback": ev.feedback}
        revised = clean_answer(revised) if revised else answer
        resp.response = apply_basis_line(retag_database(revised) if database_only(resp) else revised, resp)
        timing.finish()
        return resp


def build_query_engine(llm, embed_model, text_engine=None, sql_engine=None, reranker="auto") -> VerifiedQueryEngine:
    text_engine = text_engine or retrieval.build_text_engine(embed_model, llm, reranker=reranker)
    sql_engine = sql_engine or financials.SQLEngine()
    tools = [QueryEngineTool(query_engine=text_engine, metadata=ToolMetadata(name=TEXT_TOOL, description=TEXT_DESC)),
             QueryEngineTool(query_engine=sql_engine, metadata=ToolMetadata(name=SQL_TOOL, description=SQL_DESC))]
    sub = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=tools, llm=llm,
        question_gen=SafeQuestionGenerator.from_defaults(llm=llm),
        response_synthesizer=get_response_synthesizer(llm=llm, response_mode=ResponseMode.COMPACT, text_qa_template=FINAL_PROMPT),
        use_async=False, verbose=False)
    return VerifiedQueryEngine(inner=sub, llm=llm, evaluator=FaithfulnessEvaluator(llm=llm), enabled=config.VERIFY, text=text_engine)


_FOLLOWUP = re.compile(
    r"\b(it|its|they|them|their|theirs|that|those|this|these|there|same|former|latter|previous|above|earlier|also|"
    r"other|another|else|instead|more|rest|both)\b|^\s*(and|but|also|so|then|what about|how about|why)\b", re.I)


def needs_condense(question: str) -> bool:
    """Only rewrite a question when it can depend on the conversation. Rewriting complete questions let the LLM
    inject earlier topics into them (e.g. 'the five companies' became a list with an invented fifth company)."""
    q = question.strip()
    return len(q.split()) < 7 or bool(_FOLLOWUP.search(q))


class SelectiveCondenseChatEngine(CondenseQuestionChatEngine):
    """CondenseQuestionChatEngine that leaves self-contained questions untouched."""

    def _condense_question(self, chat_history, last_message):
        if not needs_condense(last_message):
            return last_message
        return super()._condense_question(chat_history, last_message)

    async def _acondense_question(self, chat_history, last_message):
        if not needs_condense(last_message):
            return last_message
        return await super()._acondense_question(chat_history, last_message)


def new_chat_engine(query_engine, llm) -> CondenseQuestionChatEngine:
    """One per chat session (each has its own memory)."""
    return SelectiveCondenseChatEngine.from_defaults(
        query_engine=query_engine, condense_question_prompt=CONDENSE_PROMPT, llm=llm,
        memory=ChatMemoryBuffer.from_defaults(token_limit=config.MEMORY_TOKEN_LIMIT), verbose=False)


# ---- runtime singleton (shared by the UI and the scripts) -------------------------------------
import threading as _threading

_rt_lock = _threading.Lock()
_rt = {}


def get_runtime():
    """Models, indexes and the query engine are built once per process."""
    with _rt_lock:
        if "qe" not in _rt:
            llm, embed = setup_models()
            stores.rebuild_bm25()
            stores.TABLES.reload()
            _rt["llm"], _rt["embed"] = llm, embed
            _rt["text"] = retrieval.build_text_engine(embed, llm)          # also used directly by the company brief
            _rt["qe"] = build_query_engine(llm, embed, text_engine=_rt["text"])
        return _rt


def unpack_chat_response(chat_resp) -> dict:
    """Pull the pieces the UI shows out of a CondenseQuestionChatEngine response."""
    raw, condensed = None, None
    for s in getattr(chat_resp, "sources", []) or []:
        if getattr(s, "raw_output", None) is not None:
            raw = s.raw_output
        if isinstance(getattr(s, "raw_input", None), dict):
            condensed = s.raw_input.get("query") or condensed
    meta = getattr(raw, "metadata", None) or {}
    return {"answer": str(chat_resp.response), "condensed_question": condensed,
            "verification": meta.get("verification", {"status": "unknown"}),
            "source_nodes": list(getattr(raw, "source_nodes", None) or chat_resp.source_nodes or [])}