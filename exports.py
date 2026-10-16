"""Excel export of one answer: the question and answer, every database table behind it, and the source list."""
import io
import re

import pandas as pd


def build_xlsx(path: str, question: str, answer: str, source_nodes) -> str:
    tables, sources = [], []
    for n in source_nodes or []:
        text = n.node.get_content(metadata_mode="none")
        if text.startswith("Sub question:"):
            continue
        if n.node.metadata.get("node_kind") == "sql_result":
            try:
                tables.append(pd.read_json(io.StringIO(n.node.metadata["chart_table"]), orient="split"))
            except Exception:
                pass
        else:
            sources.append(text.split("\n", 1)[0])
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame({"": ["Question", "Answer"], " ": [question, re.sub(r"\s+\n", "\n", answer)]}).to_excel(
            xw, sheet_name="Answer", index=False, header=False)
        ws = xw.sheets["Answer"]
        ws.column_dimensions["A"].width = 12
        ws.column_dimensions["B"].width = 120
        for i, df in enumerate(tables, 1):
            df.to_excel(xw, sheet_name=f"Data {i}", index=False)
        if sources:
            pd.DataFrame({"Source passages": sources}).to_excel(xw, sheet_name="Sources", index=False)
    return path