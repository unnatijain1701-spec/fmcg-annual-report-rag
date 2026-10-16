"""Chainlit chat UI.   Run:  chainlit run app.py -w"""
import asyncio
import re
import json
import shutil
import time
import traceback
from pathlib import Path

import chainlit as cl

import charts
import config
import engine
import exports
import ratios
import ingest
import stores
import timing
from corpus import guess_company, normalize_fy


def _catalogue_md() -> str:
    reps = stores.load_reports()
    if not reps:
        return "_No reports indexed yet. Run `python -m scripts.build_corpus` first._"
    by_c = {}
    for r in reps.values():
        by_c.setdefault(r["company"], []).append(r["fy"])
    return "\n".join(f"- **{c}**: {', '.join(sorted(v))}" for c, v in sorted(by_c.items()))


def _verification_line(v: dict) -> str:
    return {"verified": "✅ _Checked against the retrieved sources._",
            "revised": "✏️ _Some statements were not supported by the sources and were removed or corrected._",
            "no_evidence": "⚠️ _No source passages were found for this answer._",
            "disabled": "", "unknown": "", "error": "⚠️ _Verification could not run._"}.get(v.get("status"), "")


_PAGES = {}


def _page_map() -> dict:
    """data/nodes/page_map.json (made by scripts.build_page_map): passage id -> PDF and page. Empty if it does not exist."""
    if not _PAGES:
        try:
            _PAGES.update(json.load(open(config.NODE_DIR / "page_map.json", encoding="utf-8")))
        except Exception:
            _PAGES["map"] = {}
    return _PAGES


def _page_label(node) -> str:
    e = _page_map().get("map", {}).get(node.node_id)
    if not e:
        return ""
    m = node.metadata
    pg = f"p.{e['p1']}" if e["p1"] == e["p2"] else f"pp.{e['p1']}-{e['p2']}"
    return f"{m.get('company', '')} {m.get('fy', '')}, {pg}".strip()


def _page_refs(nodes, limit: int = 6):
    """Clickable PDF pages for the sources of an answer (opens the report at that page in the side panel).
    Picks them round-robin across companies, so one company's passages cannot fill every slot."""
    pm = _page_map()
    cands, labels = [], set()
    for n in nodes:
        meta = n.node.metadata
        if meta.get("node_kind") == "sql_result":                     # figures: the report table they were read from
            for t in meta.get("table_ids") or []:
                e = pm.get("map", {}).get(t["id"])
                if not e:
                    continue
                pg = f"p.{e['p1']}" if e["p1"] == e["p2"] else f"pp.{e['p1']}-{e['p2']}"
                rep = re.search(r"(?:FY|CY)\d{4}(?:-\d{2})?", e["pdf"])
                rep = rep.group(0) if rep else ""
                label = f"{t['company']} {rep} report, {pg}" + (f" ({t['period']} figure)" if rep and rep != t["period"] else "")
                cands.append((t["company"], label, e))
            continue
        if n.node.get_content(metadata_mode="none").startswith("Sub question:"):
            continue
        e = pm.get("map", {}).get(n.node.node_id)
        label = _page_label(n.node)
        if e and label:
            cands.append((meta.get("company", ""), label, e))
    by_company = {}
    for company, label, e in cands:
        if label not in labels:
            labels.add(label)
            by_company.setdefault(company, []).append((label, e))
    elements, out = [], []
    while by_company and len(elements) < limit:
        for company in list(by_company):
            label, e = by_company[company].pop(0)
            path = Path(pm["root"]) / e["pdf"]
            if path.exists():
                out.append(label)
                elements.append(cl.Pdf(name=label, path=str(path), display="side", page=e["p1"]))
                if len(elements) >= limit:
                    break
            if not by_company[company]:
                del by_company[company]
    return elements, ("📄 Open the report page: " + " · ".join(out)) if out else ""


def _sources_md(nodes) -> str:
    seen, out = set(), []
    for n in nodes:
        text = n.node.get_content(metadata_mode="none")
        if text.startswith("Sub question:"):
            continue
        key = n.node.node_id
        if key in seen:
            continue
        seen.add(key)
        if n.node.metadata.get("node_kind") == "sql_result":
            out.append("**Financial database**\n```\n" + text[:1800] + "\n```")
        else:
            head = text.split("\n", 1)[0]
            body = text.split("\n", 1)[1] if "\n" in text else text
            pg = _page_label(n.node)
            out.append(f"**{head}**" + (f"  ·  📄 {pg}" if pg else "") + "\n> " + re.sub(r"\s+", " ", body)[:420] + "…")
        if len(out) >= 8:
            break
    return "\n\n".join(out) or "_No sources._"


async def _send_answer(text, source_nodes, link, question, elements=None, n=None):
    """Sends an answer with its sources link and an Excel export button (the export keeps the answer, its database tables and sources)."""
    elements = list(elements or [])
    if n is None:
        n = cl.user_session.get("src_n", 0)
        elements.append(cl.Text(name=link, content=_sources_md(source_nodes), display="side"))
    pdfs, pdf_line = _page_refs(source_nodes)
    elements += pdfs
    if pdf_line:
        text += "\n\n" + pdf_line
    store = cl.user_session.get("exports") or {}
    store[n] = (question, text, source_nodes)
    cl.user_session.set("exports", store)
    await cl.Message(content=text + f"\n\n📎 {link}", elements=elements,
                     actions=[cl.Action(name="export_excel", payload={"n": n}, label="📥 Download as Excel")]).send()


@cl.action_callback("export_excel")
async def export_excel(action: cl.Action):
    store = cl.user_session.get("exports") or {}
    item = store.get(action.payload["n"])
    if not item:
        await cl.Message(content="That answer is no longer available for export.").send()
        return
    import tempfile
    path = tempfile.mktemp(suffix=".xlsx")
    await cl.make_async(exports.build_xlsx)(path, *item)
    await cl.Message(content="Excel file ready:", elements=[cl.File(name=f"answer_{action.payload['n']}.xlsx", path=path, display="inline")]).send()


@cl.on_chat_start
async def start():
    await cl.Message(content="Loading models and indexes…").send()
    rt = await cl.make_async(engine.get_runtime)()
    cl.user_session.set("chat", engine.new_chat_engine(rt["qe"], rt["llm"]))
    actions = [cl.Action(name="add_report", payload={"go": True}, label="📄 Add an annual report (PDF)")]
    try:
        actions += [cl.Action(name="brief", payload={"company": c}, label=f"📋 Brief: {c}") for c in ratios.all_companies()]
    except Exception:
        traceback.print_exc()
    await cl.Message(content=(
        "**FMCG annual-report research assistant.** Ask about figures (growth, margins, rankings across companies) "
        "or about strategy, risks and outlook. Follow-up questions work.\n\n**Indexed reports**\n" + _catalogue_md() +
        "\n\nTo add another company, attach its annual-report PDF to a message or press the button."),
        actions=actions).send()


@cl.action_callback("add_report")
async def add_report(action: cl.Action):
    files = await cl.AskFileMessage(content="Upload an annual-report PDF.", accept=["application/pdf"],
                                    max_size_mb=300, max_files=1, timeout=600).send()
    if files:
        await _handle_upload(files[0].path, files[0].name)


@cl.action_callback("brief")
async def brief(action: cl.Action):
    await _company_brief(action.payload["company"])


async def _company_brief(company: str):
    await _close_side_panel()
    """Key figures straight from the database, then strategy and risks from the reports (fresh chat so it does not mix with the conversation)."""
    await cl.Message(content=f"Preparing the brief on **{company}** (about a minute)…").send()
    table, period = await cl.make_async(ratios.brief_table)(company)
    parts = [f"## {company}: brief" + (f" ({period})" if period else "")]
    if table:
        parts.append("**Key figures** [Financial database]\n\n" + table)
    rt = await cl.make_async(engine.get_runtime)()
    nodes = []
    for title, q in (("Strategy and outlook", f"What is {company}'s strategy and outlook"),
                     ("Key risks", f"What are the key risks {company} reports")):
        try:                                                   # text engine directly: no database sub-question, no mixing with the chat
            r = await cl.make_async(rt["text"].query)(q + (f" in {period}?" if period else "?"))
            parts.append(f"**{title}**\n\n{engine.clean_answer(str(r))}")
            nodes += list(r.source_nodes or [])
        except Exception as e:
            parts.append(f"**{title}**\n\n_Could not be prepared: {type(e).__name__}_")
    n = cl.user_session.get("src_n", 0) + 1
    cl.user_session.set("src_n", n)
    link = f"View sources (answer {n})"
    text = "\n\n".join(parts)
    await _send_answer(text, nodes, link, f"Brief on {company}")


async def _ask_company_fy(filename: str):
    company, fy = guess_company(re.sub(r"[_\-]+", " ", filename)), None
    try:
        fy = normalize_fy(filename)[0]
    except ValueError:
        pass
    if company and fy:
        return company, fy
    reply = await cl.AskUserMessage(
        content="Which company and fiscal year is this report? For example: `Marico FY2024-25`", timeout=300).send()
    if not reply:
        return None, None
    text = reply["output"]
    m = re.search(r"(FY\s*'?\d{2,4}(?:\s*[-–/]\s*\d{2,4})?|20\d\d\s*[-–/]\s*\d{2,4}|CY\s*20\d\d|20\d\d)", text, re.I)
    if not m:
        return None, None
    company = text[:m.start()].strip(" ,-:") or company
    return (company.title() if company and company.islower() else company), normalize_fy(m.group(1))[0]


async def _handle_upload(path: str, name: str):
    company, fy = await _ask_company_fy(name)
    if not company or not fy:
        await cl.Message(content="I could not read a company and fiscal year, so nothing was added. "
                                 "Please try again with e.g. `Marico FY2024-25`.").send()
        return
    dest = config.UPLOAD_DIR / re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, dest)
    try:
        job = ingest.start_job(dest, company, fy)
    except RuntimeError as e:
        await cl.Message(content=f"⚠️ {e}").send()
        return
    status = cl.Message(content=f"Adding **{company} {fy}** in the background. You can keep chatting meanwhile.")
    await status.send()
    asyncio.create_task(_watch(job, status))


async def _watch(job: ingest.Job, msg: cl.Message):
    last = ""
    while not job.done:
        await asyncio.sleep(4)
        text = f"Adding **{job.company} {job.fy}** — step: `{job.step}`\n\n_{job.log[-1] if job.log else 'starting…'}_"
        if text != last:
            msg.content, last = text, text
            await msg.update()
    if job.status == "done":
        msg.content = (f"✅ **{job.company} {job.fy}** is now searchable and its figures are in the database. "
                       "You can compare it with the other companies.")
    elif job.status == "cancelled":
        msg.content = "Upload cancelled."
    else:
        msg.content = f"❌ Adding {job.company} {job.fy} failed: {job.error}"
    await msg.update()


async def _close_side_panel():
    """The side panel keeps the last opened PDF or source list; clear it when a new question starts."""
    try:
        await cl.ElementSidebar.set_elements([])
    except Exception:
        pass


@cl.on_message
async def on_message(message: cl.Message):
    await _close_side_panel()
    pdfs = [e for e in (message.elements or []) if getattr(e, "mime", "") == "application/pdf" or str(e.name).lower().endswith(".pdf")]
    if pdfs:
        for e in pdfs:
            await _handle_upload(e.path, e.name)
        if not message.content.strip():
            return
    mb = re.match(r"^\s*(?:company\s+)?brief(?:\s+(?:on|of|for))?\s+(.+?)\s*\??$", message.content, re.I)
    if mb:
        known = {c.lower(): c for c in ratios.all_companies()}
        hit = next((v for k, v in known.items() if k in mb.group(1).lower() or mb.group(1).lower() in k), None)
        if hit:
            await _company_brief(hit)
            return
    chat = cl.user_session.get("chat")
    t0 = time.perf_counter()
    async with cl.Step(name="Research steps", type="run") as step:
        try:
            timing.reset()
            job = asyncio.ensure_future(cl.make_async(chat.chat)(message.content))
            while not job.done():                                  # live progress: which stage the system is on
                step.output = timing.render()
                await step.update()
                await asyncio.sleep(1)
            resp = await job
            timing.finish()
        except ConnectionError:
            step.output = "Ollama is not reachable."
            await cl.Message(content="I cannot reach Ollama. Start it (`ollama serve`) and run `ollama signin` for the cloud model.").send()
            return
        except Exception as e:
            step.output = f"{type(e).__name__}: {e}"
            await cl.Message(content=f"Something went wrong: `{type(e).__name__}: {e}`").send()
            return
        info = engine.unpack_chat_response(resp)
        step.output = timing.render() + f"\n\nQuestion used for retrieval: {info['condensed_question'] or message.content}"
    footer = _verification_line(info["verification"])
    elements = []
    try:
        for i, fig in enumerate(charts.build_charts(info["source_nodes"])):
            elements.append(cl.Plotly(name=f"chart_{i + 1}", figure=fig, display="inline", size="large"))
    except Exception:
        traceback.print_exc()                    # shown in the terminal; a chart problem must never hide the answer
    # Sources open in the side panel (click the link under the answer); charts stay inline under the answer.
    n = cl.user_session.get("src_n", 0) + 1                      # a new name per answer, so the panel shows only this answer's sources
    cl.user_session.set("src_n", n)
    link = f"View sources (answer {n})"
    elements.append(cl.Text(name=link, content=_sources_md(info["source_nodes"]), display="side"))
    await _send_answer(info["answer"] + (f"\n\n{footer}" if footer else ""), info["source_nodes"], link, message.content, elements, n)
    try:                                                          # every question and answer is kept in chat_log.jsonl
        with open("chat_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.perf_counter() - t0, 1),
                                "question": message.content, "asked_as": info["condensed_question"],
                                "answer": info["answer"], "check": (info["verification"] or {}).get("status"),
                                "sources": [n.node.get_content(metadata_mode="none")[:200] for n in info["source_nodes"]][:10]},
                               ensure_ascii=False) + "\n")
    except Exception:
        traceback.print_exc()