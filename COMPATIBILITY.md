# Compatibility check

Result: the chosen stack works together. Eight points needed a workaround, listed below with how each is handled in the code.
Tested with llama-index-core 0.14.25 against the 20 supplied reports; the Ollama-served models, Docling and the reranker
weights could not be run in the sandbox used to build this (see "Not tested").

| # | Pair / component | Finding | Handling |
|---|---|---|---|
| 1 | Docling -> MarkdownElementNodeParser | Compatible. 7,317 of 7,363 table blocks (99.4%) in the supplied files parse into tables. | Used as is. |
| 2 | MarkdownElementNodeParser | Makes one LLM call per table for the summary: about 7,300 calls for this corpus. | `RuleSummaryElementParser` builds the summary from the table (section heading, column headers, row labels) with no LLM. `TABLE_SUMMARY_MODE=llm` restores the stock behaviour. |
| 3 | SemanticSplitterNodeParser | Embeds every sentence and has no maximum chunk size. | Used only for MD&A and risk sections (about 4 MB of 18 MB narrative); other text uses SentenceSplitter; chunks over 4,500 characters are re-split. It is passed into MarkdownElementNodeParser as `nested_node_parser`. |
| 4 | bge-m3 via Ollama | Dense only, 1024 dimensions, as planned. Ollama's default context could truncate long chunks. | `num_ctx=8192` set on the embedder. |
| 5 | Qdrant + BM25Retriever + QueryFusionRetriever | Compatible, but (a) Qdrant gives BM25 no document store, (b) QueryFusionRetriever resolves a default OpenAI LLM unless given one, (c) embedded Qdrant allows one process only. | (a) Nodes are persisted to JSONL and BM25 is rebuilt in memory in seconds (also after each upload); (b) the LLM is passed explicitly and `num_queries=1`; (c) set `QDRANT_URL` to use a Qdrant server. |
| 6 | bge-reranker-v2-m3 | Ollama has no rerank endpoint, so it cannot run through Ollama like the other models. | Runs in-process via `SentenceTransformerRerank` (needs PyTorch, about 2 GB download). |
| 7 | SubQuestionQueryEngine + FaithfulnessEvaluator | The final response's `source_nodes` mix the sub-answers ("Sub question: ... Response: ...") with the raw evidence. Checking against sub-answers would verify the model against its own intermediate text. | `collect_contexts` checks against raw passages and SQL result tables only. The SQL tool is a custom query engine whose source node holds the SQL and the rows, so numbers can be verified. |
| 8 | SubQuestionQueryEngine question generator | Needs well-formed JSON from the LLM. | `SafeQuestionGenerator` falls back to one sub-question routed by keyword. |
| 9 | gemma4:31b-cloud | Runs on Ollama's servers (internet, Ollama account, usage limits). Structured-output (JSON schema) support is not documented for it. | Table extraction uses `EXTRACT_MODEL`; if gemma returns malformed JSON set `EXTRACT_MODEL=qwen2.5-coder:7b`. |
| 10 | Streaming + verification | The answer must be complete before it is checked. | Answers are shown after verification, not streamed. |

## Not tested
Real Ollama calls (gemma4, bge-m3, qwen2.5-coder), Docling PDF conversion, loading the bge-reranker weights, and a full browser
session in Chainlit (the server was started and answered HTTP 200). The offline tests replace the models with stand-ins; run
`python -m scripts.check_env` and `python -m scripts.run_eval` on your machine to cover these.
