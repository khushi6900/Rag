from src._3retrieval.retriever import (
    INDEX_FILE,
    METADATA_FILE,
    load_index,
    retrieve_top_k,
)

RAG_ANSWER_PROMPT = """You are a helpful assistant answering questions using ONLY the context provided below.
If the answer is not contained in the context, say "I don't have enough information to answer that" — do not make up an answer.

Context:
{context}

Question: {question}

Answer:"""

def build_prompt(query, retrieved_chunks):
    context = "\n\n".join(f"Source: {chunk['source']}\nText: {chunk['text']}" for chunk in retrieved_chunks)
    return RAG_ANSWER_PROMPT.format(context=context, question=query)

if __name__ == "__main__":
    index, chunks = load_index(INDEX_FILE, METADATA_FILE)
    query = "What is Amazon API Gateway?"
    retrieved_chunks = retrieve_top_k(query, index, chunks, k=3)
    prompt = build_prompt(query, retrieved_chunks)
    print(prompt)