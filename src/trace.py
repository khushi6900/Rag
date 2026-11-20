from dotenv import load_dotenv
from langfuse import observe, get_client
from src._3retrieval.retriever import (hybrid_retrieve, rerank_chunks, load_index, load_bm25)
from src._5llm.llm_client import generate_response_from_context
import json, time
import numpy as np

load_dotenv()
GOLDEN_DATA = "GoldenDataSet/golden_dataset.json"
INDEX_FILE  = "faiss_index.bin"
BM25_FILE = "bm25_index.pkl"
MetaData = "chunk_metadata.json"

with open(GOLDEN_DATA) as f:
    goldend = json.load(f)

index , chunks = load_index(index_file = INDEX_FILE, metadata_file = MetaData)
bm25 = load_bm25(BM25_FILE)

@observe(name="retrieve")
def retrieve(query):
    return hybrid_retrieve(query=query, index=index, chunks=chunks,bm25=bm25,k=20)

@observe(name="rerank")
def rerank(query, chunks):
    return rerank_chunks(query=query, candidates=chunks, top_k=5)

@observe(as_type="generation",name="llm")
def generate(query, context):
    answer, usage = generate_response_from_context(query=query, context=context)
    get_client().update_current_generation(
        model="openai/gpt-oss-20b",
        usage_details={
            "input": usage.get("input_tokens", 0),
            "output": usage.get("output_tokens", 0)
        },
    )
    return answer

@observe(name="rag_pipeline")
def ask(query):
    get_client().update_current_span(metadata={"run": "eval-run-6"})
    chunks = retrieve(query)
    reranked = rerank(query,chunks)
    text = [c["text"] if isinstance(c, dict) else c for c in reranked]
    return generate(query, text)

if __name__ == "__main__":
    lat, errors, rows = [], 0, []
    for i, data in enumerate(goldend, 1):
        question = data["question"]
        t = time.perf_counter()
        try:
            ask(question)
            dt = time.perf_counter() - t
            lat.append(dt)
            rows.append({"question": question, "latency": dt, "ok": True})
        except Exception as e:
            errors += 1
            rows.append({"question": question, "ok": False, "error": str(e)})
            print(f"FAIL {i}: {e}")
        time.sleep(2)

    get_client().flush()
    json.dump(rows, open("Evals/latency_run6.json","w"), indent=2)
    warm = lat[1:]
    print(f"n={len(lat)} errors={errors}")
    print(f"warm p50={np.percentile(warm,50):.2f}s p95={np.percentile(warm,95):.2f}s")