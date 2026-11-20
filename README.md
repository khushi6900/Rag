# AWS Docs RAG: Evaluation & Performance Results

## Setup
- **Corpus:** AWS User Guide documents.
- **Pipeline:** Hybrid search (BM25 + FAISS vector) → cross-encoder rerank → LLM answer grounded only in context.
- **Golden set:** 44 in-scope + 6 out-of-scope questions.
- **Metrics:** DeepEval (LLM-judged), run on Groq.
- **Judge:** `gpt-oss-20b` for retriever. `qwen/qwen3.8-27b` for generator where generator is `gpt-oss-20b`.
- **Tracing:** Langfuse (`@observe` spans for retrieve, rerank, llm; tokens and cost per call).

---

## 1. Retrieval experiments

| # | Variant | Pool → final k | Ctx Recall | Ctx Precision | Recall pass | Precision pass |
|---|---|---|---|---|---|---|
| 1 | Hybrid, no rerank | 3 | 0.88 | 0.84 | 34/44 | 38/44 |
| 2 | + bge-reranker-v2-m3 | 3 | 0.85 | 0.98 | 33/44 | 44/44 |
| 3 | + bge-reranker-v2-m3 | 20 → 5 | **0.97** | 0.94 | 41/44 | 42/44 |
| 4 | bge-reranker-v2-m3 | 10 → 5 | 0.87 | 0.95 | 35/44 | 44/44 |
| 5 | ms-marco-MiniLM-L-6-v2 | 20 → 5 | 0.86 | 0.91 | 33/44 | 40/44 |
| 6 | bge-reranker-base | 20 → 5 | 0.88 | 0.94 | 34/44 | 43/44 |
| 7 | bge-reranker-v2-m3, `max_length=256` | 20 → 5 | 0.86 | 0.93 | 34/44 | 43/44 |
| 8 | **Jina hosted reranker (final)** | 20 → 5 | 0.88 | 0.91 | 35/44 | 42/44 |

v2-m3 with `max_length=512` was timed (latency only, ~13 s p50, chunks fit in 512 tokens so it behaves like full length). Cohere `rerank-v3.5` was attempted but blocked by the trial rate limit, so it is not reported.

### Findings
- **Reranking improves precision** (0.84 → 0.98 at k=3): relevant chunks rank first.
- **Recall depends on the candidate pool, not the reranker.** A reranker only reorders what it receives. Reranking the same 3 chunks did not raise recall (0.88 → 0.85 is a one-question, within-noise change). Widening the pool to 20 raised it to 0.97.
- **Shrinking the pool cost recall.** Pool 20 → 10 dropped recall 0.97 → 0.87.
- **Smaller rerankers also lost recall.** MiniLM (0.86) and bge-reranker-base (0.88) both trail v2-m3 (0.97) at the same pool size.
- **v2-m3 at full length is the quality ceiling (0.97)** with the same judge across runs, but it costs ~12 s p50 on CPU. Truncating to 256 tokens removed the gain (0.86).
- Differences of 1-2 points are noise (1 question ≈ 2.3%).

---

## 2. Generation evaluation
Run on frozen retrieval output (top-5 chunks), so scores reflect the real pipeline.

| Metric | Score | Pass rate |
|---|---|---|
| Faithfulness | 0.99 | 43/44 |
| Answer Relevancy | 0.95 | 42/44 |

The strict "answer only from context" prompt contributes to the high faithfulness.

---

## 3. Guardrail: out-of-scope questions
Scored with a deterministic refusal check, not Faithfulness/Answer Relevancy (Answer Relevancy scores a correct refusal as 0).

| Test | Result |
|---|---|
| Out-of-scope refusal rate | 6/6 |

Small sample; more "AWS-sounding but not in the docs" questions are planned.

---

## 4. Latency and cost (Langfuse + local timing, 44 queries each, 0 errors)

### End-to-end

| Reranker config | p50 | p95 |
|---|---|---|
| bge-reranker-v2-m3, full length | 11.7 s | 18.8 s |
| bge-reranker-v2-m3, `max_length=256`, `batch_size=16` | 8.4 s | 11.1 s |
| bge-reranker-v2-m3, `max_length=512` | ~13 s | ~18 s |
| ms-marco-MiniLM-L-6-v2 | 4.0 s | 6.0 s |
| **Jina hosted (final)** | **3.8 s** | **5.0 s** |

### Per-stage average

| Stage | v2-m3 (full) | MiniLM |
|---|---|---|
| Retrieve | 175 ms | 124 ms |
| Rerank | 10,667 ms | 323 ms |
| LLM | 849 ms | 2,920 ms |

- **Cost:** about $0.0001 per query at list price (~$0.10 per 1,000 queries).
- **Cold start:** first query is reported separately from warm p50/p95.

---

## 5. Quality vs latency trade-off

| Reranker | Pool → k | Recall | Precision | p50 |
|---|---|---|---|---|
| bge-reranker-v2-m3 | 20 → 5 | **0.97** | 0.94 | 11.7 s |
| bge-reranker-v2-m3 (`max_length=256`) | 20 → 5 | 0.86 | 0.93 | 8.4 s |
| bge-reranker-base | 20 → 5 | 0.88 | 0.94 | not measured |
| MiniLM-L-6-v2 | 20 → 5 | 0.86 | 0.91 | 4.0 s |
| **Jina hosted (final)** | 20 → 5 | 0.88 | 0.91 | 3.8 s |

---

## 6. Postmortem summary

| Problem | Cause | Fix | Result |
|---|---|---|---|
| Rerank was ~90% of latency | 568M cross-encoder scoring 20 long chunks on CPU | Truncation (`max_length=256`), batching | p50 11.7 s → 8.4 s |
| Swapped to MiniLM for speed | Small model trained on short web passages | Measured quality before adopting | Rerank 33x faster, recall fell 0.97 → 0.86 |
| Reduced pool to save time | Reranker can't recover chunks it never sees | Reverted to pool 20 | Recall 0.87 → 0.97 |
| Judge returned empty / truncated JSON | Reasoning model burned `max_tokens` on hidden reasoning; free-tier output-token cap | Switched to non-reasoning Llama judge | Evals completed |
| Generator answered "cannot find" for most questions | Joined chunks into a string, so `"\n\n".join()` iterated characters | Passed the list directly | Correct answers |

---

## 7. Known limitations
1. Small dataset (44 + 6); one question ≈ 2.3%.
2. LLM-judge noise; the judge (`gpt-oss-20b`) is the same model as the generator, which may favour its own answers.
3. Remaining failures (retrieval recall, precision, faithfulness, relevancy) are logged in `POSTMORTEMS.md`, split into dataset bugs and retrieval misses.
4. Latency measured on a local CPU, not the deployment host.
5. Golden answers were drafted by ChatGPT; some contain details not in the docs (dataset bugs).

## 8. Final decision
**Hybrid search → Jina hosted reranker (pool 20 → top 5) → grounded LLM answer, with local MiniLM as fallback.**

- **Why not bge-reranker-v2-m3:** best recall (0.97) but ~12 s p50 on CPU and ~2 GB of weights to host.
- **Why Jina:** recall 0.88 and p50 3.8 s, with no model weights to deploy. Hosted rerankers need an API key and are subject to rate limits, so the local fallback keeps the app up.
- **Accepted trade-off:** about 9 points of recall versus the heaviest local model, in exchange for ~3x lower latency and a small deployment.

## 9. Next steps
- Audit the ~9 failed recall cases; fix golden answers containing facts not in the docs.
- Reduce LLM latency (non-reasoning model or lower `reasoning_effort`, `max_tokens` cap, streaming).
- Re-check generation scores with a judge different from the generator.
- FastAPI + Docker + deploy, user feedback in Langfuse, CI evals.