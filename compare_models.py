"""Same passages, different answering models. Run:  PYTHONPATH=. python compare_models.py
Retrieval and reranking run once per question; each model then writes an answer from the SAME 10 passages,
so the difference you see is the model alone. Prints seconds and the answer for each."""
import time

from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.llms.ollama import Ollama

import config, models, retrieval
from engine import clean_answer

# (model, server).  The shared college server for most; NVIDIA's Nemotron models live on this PC's own Ollama.
SHARED = config.OLLAMA_URL
LOCAL = "http://localhost:11434"
CANDIDATES = [("granite4.1:8b", SHARED), ("mistral-nemo:latest", SHARED), ("qwen3:latest", SHARED),
              ("qwen3.5:latest", SHARED), ("nemotron-3-nano:4b", LOCAL), ("nemotron-3-nano:30b-cloud", LOCAL)]
QUESTIONS = [
    "What are the key risks ITC reports in its latest annual report?",
    "What reasons does ITC's FY2024-25 annual report give for changes in its EBITDA margin?",
    "What growth strategy and outlook does Tata Consumer Products describe for FY2025-26?",
]

_, embed = models.setup_models()
fusion = retrieval.build_fusion(embed)
rr = retrieval.get_reranker()
expand = retrieval.TableExpander()

for q in QUESTIONS:
    nodes = rr.postprocess_nodes(fusion.retrieve(q), query_str=q)
    nodes = expand.postprocess_nodes(nodes, query_str=q)
    print("\n" + "=" * 100 + "\nQ:", q, f"({len(nodes)} passages)")
    for name, url in CANDIDATES:
        llm = Ollama(model=name, base_url=url, temperature=0.0, request_timeout=900.0,
                     context_window=config.LLM_CONTEXT, thinking=(False if ("qwen" in name or "nemotron" in name) else None),
                     additional_kwargs={"num_ctx": config.LLM_CONTEXT})
        synth = get_response_synthesizer(llm=llm, response_mode=ResponseMode.COMPACT, text_qa_template=retrieval.QA_PROMPT)
        t = time.time()
        try:
            ans = clean_answer(str(synth.synthesize(q, nodes)))
        except Exception as e:                      # a model that is missing or too big for the server
            ans = f"FAILED: {type(e).__name__}: {str(e)[:200]}"
        print(f"\n--- {name}  ({time.time() - t:.0f}s) ---\n{ans}")