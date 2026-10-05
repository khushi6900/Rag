from src._3retrieval.retriever import retrieve_top_k, load_index,rerank_chunks, INDEX_FILE, METADATA_FILE
from src._4prompts.prompt_templates import build_prompt
from src._5llm.llm_client import generate_response
from dotenv import load_dotenv
import boto3
import os

load_dotenv()
def ask(query, index, chunks, retrieve_k=10, top_k=3):
    candidates = retrieve_top_k(query,index, chunks, k=retrieve_k)
    if not candidates:
        return {"answer": "I don't have enough information to answer that.", "sources": []}
    # print(f"Retrieved {len(candidates)} candidate chunks for query: '{query}'")

    top_chunks = rerank_chunks(query, candidates, top_k=top_k)
    # print(f"Reranked to top {len(top_chunks)} chunks for query: '{query}'")
    # print(f"Top chunks: {[chunk['text'] for chunk in top_chunks]}")
    prompt = build_prompt(query, top_chunks)
    answer = generate_response(prompt)

    sources = list({r["source"] for r in top_chunks})
    return {"answer": answer, "sources": sources, "retrieved_chunks": top_chunks}

if __name__ == "__main__":
    index, chunks = load_index(INDEX_FILE, METADATA_FILE)
    query = "What is Amazon CloudSearch?"
    result = ask(query, index, chunks)
    # print(result)
    print(f"Query: {query}")
    print(f"Answer: {result['answer']}")
    # print(f"Sources: {result['sources']}")
