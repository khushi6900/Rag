import json
from src._3retrieval.retriever import hybrid_retrieve, rerank_chunks, load_index, load_bm25

index, chunks = load_index(index_file="faiss_index.bin", metadata_file="chunk_metadata.json")
bm25 = load_bm25("bm25_index.pkl")

def top_score(q):
    pool = hybrid_retrieve(query=q, index=index, chunks=chunks, bm25=bm25, k=30)
    return rerank_chunks(q, pool, top_k=5)[0]["rerank_score"]

ins = [g["question"] for g in json.load(open("GoldenDataSet/golden_dataset.json"))]
oos = [g["question"] for g in json.load(open("GoldenDataSet/golden_out_of_scope.json"))]

import time
a = []
for q in ins: a.append(top_score(q)); time.sleep(2)     # keeps Jina under its rate limit
b = []
for q in oos: b.append(top_score(q)); time.sleep(2)
print("in-scope:  min", round(min(a), 3), "p10", round(sorted(a)[len(a)//10], 3))
print("out-scope: max", round(max(b), 3), "scores", [round(x, 3) for x in sorted(b)])