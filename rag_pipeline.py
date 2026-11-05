from src._3retrieval.retriever import hybrid_retrieve, load_bm25, load_index, INDEX_FILE, METADATA_FILE, BM25_FILE
from src._4prompts.prompt_templates import build_prompt
from src._5llm.llm_client import generate_response
from dotenv import load_dotenv

load_dotenv()
def ask(query, index, chunks, bm25, retrieve_k=5, top_k=3):
    candidates = hybrid_retrieve(query,index, chunks, bm25, k=retrieve_k)
    if not candidates:
        return {"answer": "I don't have enough information to answer that.", "sources": []}
    # print(f"Retrieved {len(candidates)} candidate chunks for query: '{query}'")

    # top_chunks = rerank_chunks(query, candidates, top_k=top_k)
    # print(f"Reranked to top {len(top_chunks)} chunks for query: '{query}'")
    # print(f"Top chunks: {[chunk['text'] for chunk in top_chunks]}")
    prompt = build_prompt(query, candidates)
    answer = generate_response(prompt)

    sources = list({r["source"] for r in candidates})
    return {"answer": answer, "sources": sources, "retrieved_chunks": candidates}

if __name__ == "__main__":
    index, chunks = load_index(INDEX_FILE, METADATA_FILE)
    bm25 = load_bm25(BM25_FILE)
    query = "What is Amazon CloudSearch?"
    result = ask(query, index, chunks, bm25)
    # print(result)
    print(f"Query: {query}")
    print(f"Answer: {result['answer']}")
    # print(f"Sources: {result['sources']}")
