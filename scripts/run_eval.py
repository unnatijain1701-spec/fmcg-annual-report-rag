"""Run eval/questions.txt through the full chat engine (one session, so follow-ups use memory) and write eval/results.md.

    python -m scripts.run_eval
"""
import time

import config
import engine

QUESTIONS = config.ROOT / "eval" / "questions.txt"
OUT = config.ROOT / "eval" / "results.md"


def main():
    rt = engine.get_runtime()
    chat = engine.new_chat_engine(rt["qe"], rt["llm"])
    lines = ["# Evaluation run\n"]
    for q in [l.strip() for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]:
        t = time.time()
        try:
            info = engine.unpack_chat_response(chat.chat(q))
        except Exception as e:
            lines += [f"## {q}\n", f"ERROR: {type(e).__name__}: {e}\n"]
            print("ERR ", q, e)
            continue
        dt = time.time() - t
        lines += [f"## {q}\n", f"_Rewritten as:_ {info['condensed_question']}\n" if info["condensed_question"] != q else "",
                  info["answer"], f"\n_verification: {info['verification'].get('status')} | sources: {len(info['source_nodes'])} | {dt:.0f}s_\n"]
        print(f"{dt:5.0f}s  {info['verification'].get('status'):9}  {q}")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
