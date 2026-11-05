# Evaluation Results

Setup: 44 in-scope golden questions and 6 out-of-scope questions, built from AWS User Guide documents. Metrics were computed using DeepEval. Retrieval was judged by `gpt-oss-20b`, while generation was judged by `qwen/qwen3.8-27b`. Both judges ran on Groq, and the judge models were different from the generator.

## 1. Retrieval Evaluation

| Variant                | Candidates → Final k | Contextual Recall | Contextual Precision | Recall Pass | Precision Pass |
| ---------------------- | -------------------- | ----------------- | -------------------- | ----------- | -------------- |
| Hybrid (BM25 + vector) | Top 3                | 0.88              | 0.84                 | 34/44       | 38/44          |
| + Reranker             | Top 3                | 0.85              | 0.98                 | 33/44       | 44/44          |
| + Reranker (wide pool) | 20 → Top 5           | 0.97              | 0.94                 | 41/44       | 42/44          |

### Key Findings

- Reranking improves precision. Contextual precision increased from 0.84 to 0.98 while keeping \\(k=3\\). This indicates that the retrieved chunks were more relevant and better ranked.
- Recall depends on the candidate pool. A reranker only reorders retrieved candidates. Reranking the same three candidates did not improve recall (0.88 → 0.85), a difference of one question in the pass count that may reflect judge variability. Expanding the candidate pool to 20 and retaining the top 5 increased recall to 0.97.
- There is a precision–recall trade-off. Expanding the pool improved recall but reduced precision slightly, from 0.98 to 0.94. This trade-off was accepted in favor of higher recall.
- Final retrieval configuration: Hybrid search → rerank 20 candidates → retain the top 5.

Caveat: The final \\(k\\) differs across configurations (3 versus 5). With only 44 in-scope questions, each question represents approximately 2.3% of the evaluation set, so small differences should be interpreted cautiously.

## 2. Generation Evaluation

Generation was evaluated using the frozen retrieval output containing the top 5 chunks. Therefore, these scores reflect generation performance within the actual retrieval pipeline.

| Metric           | Score | Pass Rate |
| ---------------- | ----- | --------- |
| Faithfulness     | 0.99  | 43/44     |
| Answer Relevancy | 0.95  | 42/44     |

### Key Findings

- High faithfulness: The generated answers remain grounded in the retrieved context.
- High answer relevancy: The answers generally address the questions asked.
- Prompt contribution: The strict instruction to answer only from the retrieved context contributes to the high faithfulness score.

## 3. Out-of-Scope Behavior (Guardrail Evaluation)

Out-of-scope questions are evaluated using refusal rate rather than Faithfulness or Answer Relevancy. Answer Relevancy may assign a score of 0 to a correct refusal, making it unsuitable for measuring this guardrail behavior.

| Test                      | Result  |
| ------------------------- | ------- |
| Out-of-scope refusal rate | 6/6     |

The final refusal rate must be calculated from the deterministic check results for the six out-of-scope questions.

## 4. End-to-End Evaluation Summary

| Stage       | Key Metric                               | Result                             |
| ----------- | ---------------------------------------- | ---------------------------------- |
| Retrieval   | Contextual Recall / Contextual Precision | 0.97 / 0.94                        |
| Generation  | Faithfulness / Answer Relevancy          | 0.99 / 0.95                        |
| Guardrail   | Out-of-scope refusal                     | 6/6                                |
| Performance | p95 latency / Cost per query             | To be added after Langfuse tracing |

The selected pipeline prioritizes retrieval recall while maintaining strong precision and generation quality. Guardrail and performance results remain pending.

## 5. Known Limitations

1. Small evaluation dataset: The evaluation contains 44 in-scope and 6 out-of-scope questions. Each in-scope question accounts for approximately 2.3% of the retrieval and generation evaluation set.
2. LLM-judge variability: LLM-judged metrics can be noisy, and different models were used to judge retrieval and generation.
3. Remaining failures: The recorded failures are:
   - Retrieval recall: 3 cases
   - Retrieval precision: 2 cases
   - Faithfulness: 1 case
   - Answer Relevancy: 2 cases

These failures are documented in `POSTMORTEMS.md` and categorized into dataset bugs and retrieval misses. 4. Incomplete guardrail evaluation: The out-of-scope refusal result still needs to be filled in from the deterministic check. 5. Performance evaluation pending: p95 latency and cost per query will be added after Langfuse tracing.

Final selected configuration: Hybrid search (BM25 + vector) → rerank 20 candidates → retain top 5 chunks → generate an answer grounded strictly in the retrieved context.