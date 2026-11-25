from dotenv import load_dotenv

from src._3retrieval.retriever import (
    BM25_FILE,
    INDEX_FILE,
    METADATA_FILE,
    hybrid_retrieve,
    load_bm25,
    load_index,
    rerank_chunks,
)
from src._5llm.llm_client import generate_response_from_context

load_dotenv()
REFUSE_BELOW = 0.5
NO_ANSWER = "I cannot find the answer in the provided context."

def retrieve_context(query, index, chunks, bm25, retrieve_k=30, top_k=5):
    candidates = hybrid_retrieve(query=query, index=index, chunks=chunks, bm25=bm25, k=retrieve_k)
    if not candidates:
        return [], True
    top = rerank_chunks(query, candidates, top_k=top_k)
    refused = not top[0].get("fallback") and top[0]["rerank_score"] < REFUSE_BELOW
    return top, refused

def ask(query, index, chunks, bm25, retrieve_k=5, top_k=3):
    top, refused = retrieve_context(query, index, chunks, bm25, retrieve_k=retrieve_k, top_k=top_k)
    if refused:
        return {"answer": NO_ANSWER, "sources": [], "usage": {}}
    answer, usage = generate_response_from_context(query, [c["content"] for c in top])
    return {"answer": answer, "sources": sorted({c["source"] for c in top}), "retrieved_chunks": top, "usage": usage}

if __name__ == "__main__":
    index, chunks = load_index(INDEX_FILE, METADATA_FILE)
    bm25 = load_bm25(BM25_FILE)
    query = "What is Amazon CloudSearch?"
    result = ask(query, index, chunks, bm25)
    # print(result)
    print(f"Query: {query}")
    print(f"Answer: {result['answer']}")
    # print(f"Sources: {result['sources']}")
