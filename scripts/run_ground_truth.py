"""Run eval/ground_truth.csv through the full chat engine (a fresh chat per question) and score the answers.

    python -m scripts.run_ground_truth            # all questions
    python -m scripts.run_ground_truth Q04 Q12    # only these ids

A question passes when every item in `must_contain` appears in the answer (commas, case and accents ignored).
Items are separated by |. An item written A>B>C must appear in that order; A~B means either A or B. Rows whose truth_source is
"manual" are only checked for the keyword and still need a human read.
"""
import csv
import re
import sys
import time
import unicodedata

import config
import engine

SRC = config.ROOT / "eval" / "ground_truth.csv"
OUT = config.ROOT / "eval" / "ground_truth_results.md"


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[,\s]+", " ", s.replace(",", ""))


def check(answer: str, must: str):
    a, missing = norm(answer), []
    for item in must.split("|"):
        item = item.strip()
        if not item:
            continue
        if ">" in item:
            pos, ok = -1, True
            for part in item.split(">"):
                i = a.find(norm(part).strip(), pos + 1)
                if i < 0:
                    ok = False
                    break
                pos = i
            if not ok:
                missing.append(item)
        else:
            alts = [x.strip() for x in item.split("~")]            # A~B means either A or B
            for x in list(alts):                                     # 81,612.78 may be written as 81,613
                if re.fullmatch(r"[\d,]+\.\d+", x):
                    alts.append(f"{round(float(x.replace(',', ''))):,}")
            if not any(norm(x).strip() in a.replace(",", "") for x in alts):
                missing.append(item)
    return missing


def main():
    wanted = set(sys.argv[1:])
    rows = [r for r in csv.DictReader(open(SRC, encoding="utf-8")) if not wanted or r["id"] in wanted]
    rt = engine.get_runtime()
    lines, passed = [], 0
    for r in rows:
        t = time.time()
        try:
            chat = engine.new_chat_engine(rt["qe"], rt["llm"])
            info = engine.unpack_chat_response(chat.chat(r["question"]))
            answer = info["answer"]
        except Exception as e:
            answer = f"ERROR: {type(e).__name__}: {e}"
        dt = time.time() - t
        missing = check(answer, r["must_contain"])
        ok = not missing
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {dt:4.0f}s  {r['id']}  {r['question']}" + ("" if ok else f"   missing: {missing}"))
        lines += [f"## {r['id']} {'PASS' if ok else 'FAIL'} ({dt:.0f}s)", f"**Q:** {r['question']}",
                  f"**Expected:** {r['must_contain']} ({r['truth_source']})" + (f"  \n**Missing:** {missing}" if missing else ""),
                  "", answer, ""]
    head = f"# Ground-truth run\n\n{passed} of {len(rows)} passed.\n"
    OUT.write_text(head + "\n" + "\n".join(lines), encoding="utf-8")
    print(f"\n{passed} of {len(rows)} passed. Details: {OUT}")


if __name__ == "__main__":
    main()