"""Offline end-to-end test: no Ollama, no GPU. Uses hashing embeddings and a scripted LLM.

    python -m pytest tests -q        (or: python -m tests.test_pipeline)
It exercises: chunking -> Qdrant + BM25 -> scoped hybrid retrieval -> table expansion -> numbers extraction/DuckDB/SQL
-> sub-question routing -> faithfulness check + revision -> condense-question chat memory.
"""
import json
import pathlib
import tempfile

import pytest
from qdrant_client import QdrantClient

import chunking
import config
import engine
import financials
import retrieval
import stores
from corpus import infer_meta
from tests.fakes import HashEmbedding, ScriptedLLM

DATA = config.PROCESSED_DIR
REPORTS = ["annual-report-2023-24.md", "ITC-Report-and-Accounts-2025.md"]     # one HUL, one ITC report from data/processed
pytestmark = pytest.mark.skipif(not list(DATA.rglob("ITC-Report-and-Accounts-2025.md")), reason="put the parsed markdown in data/processed first")


@pytest.fixture(scope="module")
def env():
    tmp = pathlib.Path(tempfile.mkdtemp())
    config.NODE_DIR = tmp
    config.RETRIEVAL_NODES, config.TABLE_NODES = tmp / "r.jsonl", tmp / "t.jsonl"
    config.FIN_DIR = tmp; config.EXTRACTIONS = tmp / "extractions.jsonl"; config.DUCKDB_PATH = tmp / "fin.duckdb"
    stores.REPORTS_JSON = tmp / "reports.json"
    config.EMBED_DIM = 256
    stores.set_qdrant_client(QdrantClient(":memory:"))
    emb = HashEmbedding()
    parsers, rs = chunking.build_parsers(emb)
    for p in REPORTS:
        f = next(f for f in DATA.rglob(p) if (p != 'annual-report-2023-24.md' or f.parent.name.lower() == 'hul'))
        meta = infer_meta(f)
        ret, tabs = chunking.chunk_report(f.read_text(encoding="utf-8", errors="ignore")[:900000], meta, parsers, rs)
        stores.embed_and_store(ret, emb)
        chunking.append_jsonl(config.RETRIEVAL_NODES, ret)
        chunking.append_jsonl(config.TABLE_NODES, tabs)
        stores.save_report(meta, len(ret), len(tabs))
    stores.rebuild_bm25(); stores.TABLES.reload()
    return emb


def test_scope_and_hybrid_retrieval(env):
    sc = retrieval.parse_scope("Compare ITC and Hindustan Unilever in FY2024-25")
    assert sc["companies"] == ["ITC", "HUL"] and sc["fys"] == ["FY2024-25"]
    fusion = retrieval.build_fusion(env, ScriptedLLM())
    nodes = fusion.retrieve("What are the key risks for ITC?")
    assert nodes and {n.node.metadata["company"] for n in nodes} == {"ITC"}      # company scope respected
    nodes = fusion.retrieve("HUL and ITC business performance")
    assert {n.node.metadata["company"] for n in nodes} == {"HUL", "ITC"}          # both companies represented


def test_table_expansion(env):
    summ = [n for n in stores.read_jsonl(config.RETRIEVAL_NODES) if n.metadata["node_kind"] == "table_summary"][:3] \
        if hasattr(stores, "read_jsonl") else [n for n in chunking.read_jsonl(config.RETRIEVAL_NODES) if n.metadata["node_kind"] == "table_summary"][:3]
    from llama_index.core.schema import NodeWithScore
    out = retrieval.TableExpander().postprocess_nodes([NodeWithScore(node=n, score=1.0) for n in summ])
    assert all(o.node.metadata["node_kind"] == "table" and "|" in o.node.text for o in out)


def test_table_selection(env):
    tabs = chunking.read_jsonl(config.TABLE_NODES)
    sel = financials.select_tables(tabs, max_per_report=10)
    assert 0 < len(sel) <= 20


def _extraction(company, report_fy, rows, unit="crore", basis="consolidated"):
    return {"table_id": f"{company}-{report_fy}", "report_id": f"{company}_{report_fy}", "company": company, "report_fy": report_fy,
            "section": "financials", "basis": basis, "unit": unit, "unit_inferred": False, "model": "test", "rows": rows}


def _rows(metric, vals):
    return {"metric": metric, "segment": "", "label": metric, "values": [{"period": p, "value": v} for p, v in vals.items()]}


@pytest.fixture(scope="module")
def db(env):
    recs = [
        _extraction("ITC", "FY2024-25", [_rows("revenue_from_operations", {"FY2024-25": 75000.0, "FY2023-24": 70000.0}),
                                         _rows("net_profit", {"FY2024-25": 20000.0, "FY2023-24": 19000.0})]),
        _extraction("HUL", "FY2024-25", [_rows("revenue_from_operations", {"FY2024-25": 60000.0, "FY2023-24": 59000.0}),
                                         _rows("net_profit", {"FY2024-25": 10000.0, "FY2023-24": 9800.0})]),
        _extraction("Nestle India", "FY2024-25", [_rows("revenue_from_operations", {"FY2024-25": 200000.0})], unit="million"),
    ]
    with open(config.EXTRACTIONS, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    financials.build_db(progress=lambda *_: None)
    return True


def test_db_units_and_derived(db):
    import duckdb
    con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
    assert con.execute("select value_cr from facts_latest where company='Nestle India' and metric='revenue_from_operations'").fetchone()[0] == 20000.0
    m = con.execute("select value from facts_latest where company='ITC' and metric='net_profit_margin_pct' and period='FY2024-25'").fetchone()[0]
    assert abs(m - 100 * 20000 / 75000) < 0.01
    g = con.execute("select value from facts_latest where company='ITC' and metric='revenue_growth_yoy_pct' and period='FY2024-25'").fetchone()[0]
    assert abs(g - 100 * 5000 / 70000) < 0.01


def test_sql_engine_retry_and_readonly(db):
    eng = financials.SQLEngine()
    answers = iter(["SELECT nonexistent FROM facts_latest",                       # fails -> error fed back
                    "```sql\nSELECT company, period, value_cr FROM facts_latest WHERE metric='net_profit' ORDER BY value_cr DESC;\n```"])
    eng._ask_sql = lambda messages: next(answers)
    r = eng.query("highest net profit")
    assert "ITC" in str(r) and r.source_nodes and "SQL:" in r.source_nodes[0].node.text
    eng2 = financials.SQLEngine(); eng2._ask_sql = lambda m: "DROP TABLE facts"
    assert "No figures" in str(eng2.query("x"))


def _sql_engine():
    eng = financials.SQLEngine()
    eng._ask_sql = lambda m: "SELECT company, period, basis, value_cr FROM facts_latest WHERE company='ITC' AND metric='net_profit' AND period='FY2024-25'"
    return eng


def _llm():
    def qgen(prompt):
        q = prompt.split("<User Question>")[-1].split("<Output>")[0].strip()
        if "compare" in q.lower():
            items = [{"sub_question": "What was ITC net profit in FY2024-25?", "tool_name": engine.SQL_TOOL},
                     {"sub_question": "What are the main risks ITC reports?", "tool_name": engine.TEXT_TOOL}]
        else:
            items = [{"sub_question": q, "tool_name": engine.TEXT_TOOL}]
        return "```json\n" + json.dumps({"items": items}) + "\n```"
    state = {"faithful": True}
    llm = ScriptedLLM(rules=[
        ("Standalone question", lambda p: "What are the main risks ITC reports in FY2024-25?"),
        ("Please tell if a given piece of information is supported", lambda p: "YES" if state["faithful"] else "NO"),
        ("fact-checking a draft answer", "REVISED ANSWER: only supported claims remain."),
        ("sub-questions in json markdown", qgen),
        ("Context passages", "Answer based on passages [ITC | FY2024-25 | MD&A]."),
        ("Below are answers to sub-questions", "Final: ITC net profit Rs 20,000 crore [Financial database]; risks as in [ITC | FY2024-25 | Risk]."),
    ], calls=[])
    return llm, state


def test_end_to_end_chat(db, env):
    llm, state = _llm()
    qe = engine.build_query_engine(llm, env, sql_engine=_sql_engine(), reranker=None)
    r = qe.query("Compare ITC profit and risks")
    assert "20,000" in str(r) and r.metadata["verification"]["status"] == "verified"
    # evidence for verification includes the SQL result table, not just sub-answers
    assert any("SQL:" in c for c in engine.collect_contexts(r))
    # unfaithful draft -> single revision call, no re-run of retrieval
    state["faithful"] = False
    n_before = sum("Context passages" in c for c in llm.calls)
    r2 = qe.query("Compare ITC profit and risks")
    assert r2.metadata["verification"]["status"] == "revised" and str(r2).startswith("REVISED")
    assert sum("Context passages" in c for c in llm.calls) - n_before == 1   # retrieval ran once for this question (text sub-question), not twice


def test_chat_memory_condenses_followups(db, env):
    llm, _ = _llm()
    chat = engine.new_chat_engine(engine.build_query_engine(llm, env, sql_engine=_sql_engine(), reranker=None), llm)
    chat.chat("Compare ITC profit and risks")
    out = engine.unpack_chat_response(chat.chat("And what about the risks there?"))
    assert out["condensed_question"] == "What are the main risks ITC reports in FY2024-25?"
    assert out["verification"]["status"] == "verified" and out["source_nodes"]
    condense_calls = [c for c in llm.calls if "Standalone question" in c]
    assert len(condense_calls) == 1 and "Compare ITC profit and risks" in condense_calls[0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
