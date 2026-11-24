import json
import os
import pickle

import faiss
import numpy as np
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
# ---- CONFIG ----
# REGION = "us-east-1"
# EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
INDEX_FILE = "faiss_index.bin"
BM25_FILE = "bm25_index.pkl"
METADATA_FILE = "chunk_metadata.json"
# RERANK_REGION = "us-west-2"
# RERANK_MODEL_ID = "amazon.rerank-v1:0" 
OPENROUTER_RERANK_MODEL = "nvidia/nemotron-3.5-rerank-8b:free"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

model = SentenceTransformer(EMBED_MODEL_ID)
# aws_credentials = {
#     "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
#     "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
# }

# bedrock = boto3.client(
#     "bedrock-runtime",
#       region_name=REGION , 
#       **aws_credentials)

# rerank_client = boto3.client(
#     "bedrock-agent-runtime",
#       region_name=RERANK_REGION,
#       **aws_credentials)

def load_bm25(bm25_file):
    with open(bm25_file, "rb") as f:
        return pickle.load(f)
    
# def get_embedding(text):
#     body = json.dumps({"inputText": text})

#     response = bedrock.invoke_model(
#         modelId=EMBED_MODEL_ID,
#         body=body,
#         contentType="application/json",
#         accept="application/json"
#     )   

#     response_body = json.loads(response["body"].read())
#     return response_body["embedding"]

def get_embedding(text):
    """calls BGE embedding model to get embedding for a given text."""
    return model.encode(text)

def load_index(index_file, metadata_file):
    """Loads a previously saved FAISS index + chunk metadata."""
    index = faiss.read_index(index_file)
    with open(metadata_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks

def retrieve_top_k(query, index, chunks, k=3):
    q_emb = get_embedding(query)
    q_emb_np = np.array([q_emb], dtype=np.float32)

    distance, indices = index.search(q_emb_np, k)
    # print(f"Distances: {distance}, Indices: {indices}")

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        results.append({
            "source": chunk['source'],
            "text": chunk['text'],
            "distance": float(distance[0][rank])
        })
    return results

def retrieve_bm25(query, bm25, chunks, k=3):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:k]

    results = []
    for idx in top_indices:
        results.append({
            "source": chunks[idx]["source"],
            "text": chunks[idx]["text"],
            "bm25_score": float(scores[idx])
        })
    return results

def hybrid_retrieve(query, index, chunks, bm25, k=3, rrf_k=60):
    """Combines dense (FAISS) and sparse (BM25) results using Reciprocal Rank Fusion."""
    dense_results = retrieve_top_k(query, index, chunks, k=k)
    sparse_results = retrieve_bm25(query, bm25, chunks, k=k)

    # RRF: score = sum(1 / (rrf_k + rank)) across every list a chunk appears in
    scores = {}
    text_lookup = {}

    for rank, r in enumerate(dense_results):
        key = r["text"]  # use chunk text as the dedup key
        scores[key] = scores.get(key, 0) + 1 / (rrf_k + rank + 1)
        text_lookup[key] = r

    for rank, r in enumerate(sparse_results):
        key = r["text"]
        scores[key] = scores.get(key, 0) + 1 / (rrf_k + rank + 1)
        text_lookup[key] = r

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for key, score in ranked[:k]:
        chunk = text_lookup[key]
        results.append({**chunk, "rrf_score": score})
    return results

# =========================== Aws rerank ============================
# def rerank_chunks(query, candidates, top_k=4, max_retries=5):
#     """Reranks the candidate chunks using the Rerank model."""
#     sources = [
#         {
#             "inlineDocumentSource": {
#                 "type": "TEXT",
#                 "textDocument": {"text": c["text"]}
#             },
#             "type": "INLINE"
#         }
#         for c in candidates
#     ]

#     for attempt in range(max_retries):
#         try:
#             response = rerank_client.rerank(
#                 queries=[{"type": "TEXT", "textQuery": {"text": query}}],
#                 sources=sources,
#                 rerankingConfiguration={
#                     "type": "BEDROCK_RERANKING_MODEL",
#                     "bedrockRerankingConfiguration":{
#                         "modelConfiguration": {
#                             "modelArn": f"arn:aws:bedrock:{RERANK_REGION}::foundation-model/{RERANK_MODEL_ID}"
#                         },
#                         "numberOfResults": top_k
#                     }
#                 }
#             )
#             break
#         except Exception as e:
#             if "Throttling" in str(e) and attempt < max_retries -1:
#                 wait = 2 ** attempt
#                 print(f"  Rerank throttled, waiting {wait}s...")
#                 time.sleep(wait)
#                 continue
#             raise
#     reranked = []
#     for r in response["results"]:
#         original_chunk = candidates[r["index"]]
#         reranked.append({
#             **original_chunk,
#             "rerank_score": r["relevanceScore"]
#         })

#     return reranked

# =========================== open router rerank ============================
# def rerank_chunks(query, candidates, top_k=5, max_retries=5):
#     url = "https://openrouter.ai/api/v1/rerank"
#     headers = {
#         "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#         "Content-Type": "application/json"
#     }
#     documents = [c["text"] for c in candidates]

#     payload= {
#         "model": OPENROUTER_RERANK_MODEL,
#         "query": query,
#         "documents": documents,
#         "top_n": top_k
#     }
#     response = None
#     for attempt in range(max_retries):
#         try:
#             res = requests.post(url, headers=headers, json=payload, timeout=30)
#             if res.status_code == 429 or res.status_code>=500:
#                 if attempt < max_retries -1:
#                     wait = 2**attempt
#                     print(f" Rerank rate-limited/error ({res.status_code}), waiting {wait}s...")
#                     time.sleep(wait)
#                     continue
#             res.raise_for_status()
#             response=res.json()
#             # print(f"reranker response=========================================: {response}")
#             break
#         except Exception as e:
#             if attempt < max_retries -1:
#                 wait = 2**attempt
#                 print(f" Rerank rconnection error, waiting {wait}s... Error:{e}")
#                 time.sleep(wait)
#                 continue
#             raise
#     reranked=[]
#     for r in response.get("results", []):
#         original_idx = r["index"]
#         original_chunk = candidates[original_idx]
#         reranked.append({
#             **original_chunk,
#             "rerank_score": r.get("relevance_score", 0.0)
#         })
#     return reranked

# =========================== BAAI rerank =================================
# reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
# reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
# reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
# reranker = CrossEncoder("BAAI/bge-reranker-base")
# def rerank_chunks(query, candidates, top_k=5):

#     pairs = [[query,c["text"]] for c in candidates]
#     scores = reranker.predict(pairs) #batch_size=16
    
#     scored_cand = []
#     for i, c in enumerate(candidates):
#         scored_cand.append({
#             **c,
#             "rerank_score": float(scores[i])
#         })

#     scored_cand.sort(key=lambda x: x["rerank_score"], reverse=True)
#     return scored_cand[:top_k]

# =========================== JINA rerank =================================
def rerank_chunks(query, candidates, top_k=5):
    req = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={"Authorization": f"Bearer {os.getenv('JINA_API_KEY')}"},
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": [c["text"] for c in candidates],
            "top_n": top_k
        },
        timeout=30
    )
    req.raise_for_status()
    return [
        {**candidates[r["index"]], "rerank_score": r["relevance_score"]}
        for r in req.json()["results"]
    ]

# =========================== COHERE rerank =================================
# def rerank_chunks(query, candidates, top_k=5):
#     req = requests.post(
#         "https://api.cohere.com/v2/rerank",
#         headers={"Authorization": f"Bearer {os.getenv('COHERE_API_KEY')}"},
#         json={
#             "model": "rerank-v3.5",
#             "query": query,
#             "documents": [c["text"] for c in candidates],
#             "top_n": top_k
#         },
#         timeout=30
#     )
#     req.raise_for_status()
#     return [
#         {**candidates[r["index"]], "rerank_score": r["relevance_score"]}
#         for r in req.json()["results"]
#     ]

def retrieve_and_rerank(query, index, chunks, bm25, retrieve_k=30, rerank_top_k=5):
    """
    End-to-end retrieval pipeline:
    1. Dense Vector Search via FAISS (retrieve_top_k)
    2. Cross-Encoder Reranking via Bedrock Agent Runtime (rerank_chunks)
    a
    Returns the final top reranked chunks ready for LLM prompt context or evaluation.
    """
    # 1. First-stage retrieval (Hybrid Embeddings)
    candidates = hybrid_retrieve(query, index, chunks,bm25, k=retrieve_k)
    # print(f"==========================Candidates retrieved============================: {len(candidates)}")
    if not candidates:
        return []
        
    # 2. Second-stage reranking (BAAI Reranker)
    final_chunks = rerank_chunks(query, candidates, top_k=rerank_top_k)
    # print(f"======================Final chunksssssssss========================: {final_chunks}")
    return final_chunks

if __name__ == "__main__":
    index, chunks = load_index(INDEX_FILE, METADATA_FILE)
    bm25= load_bm25(BM25_FILE)
    print(f"Loaded index with {index.ntotal} vectors and {len(chunks)} chunks.")

    query = "What is Amazon API Gateway?"
    results = retrieve_and_rerank(query, index, chunks,bm25, retrieve_k=20, rerank_top_k=5)
    print(f"Query: {query}")
    # print(f"Final Results: {results}")
    for i, r in enumerate(results):
        print(f"Result{i+1}: Source: {r['source']}")
        print(f"Text: {r['text']}")
        print("-" * 80)