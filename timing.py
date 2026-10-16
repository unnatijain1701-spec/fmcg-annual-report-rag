"""Prints how long each model call and the reranker take, so the slow step is measured, not guessed.
Lines start with [TIMING] in the terminal. Switch off with TIMING=0 in .env."""
import os
import time

ENABLED = os.getenv("TIMING", "1") != "0"


def log(what: str, seconds: float, extra: str = ""):
    if ENABLED:
        print(f"[TIMING] {what:<14} {seconds:6.1f}s  {extra}", flush=True)


# ---- live progress shown in the chat window (one user at a time) ----
import threading

_lock = threading.Lock()
_events = []          # each: [label, start, end or None, detail]


def reset():
    with _lock:
        _events.clear()


def step(label: str, detail: str = ""):
    """Starts a new stage; the previous one is marked finished."""
    with _lock:
        now = time.time()
        if _events and _events[-1][2] is None:
            _events[-1][2] = now
        _events.append([label, now, None, detail])


def note(detail: str):
    with _lock:
        if _events:
            _events[-1][3] = detail


def finish():
    with _lock:
        if _events and _events[-1][2] is None:
            _events[-1][2] = time.time()


def render() -> str:
    with _lock:
        now = time.time()
        out = []
        for label, s, e, d in _events:
            out.append(f"{'✅' if e else '⏳'} {label} ({(e or now) - s:.0f}s)" + (f"\n    _{d}_" if d else ""))
        return "\n".join(out) or "⏳ Starting…"