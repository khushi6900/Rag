import json
import boto3
import numpy as np
import faiss
import time

# ---- CONFIG ----
REGION = "us-east-1"
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "chunk_metadata.json"

RERANK_REGION = "us-west-2"
RERANK_MODEL_ID = "amazon.rerank-v1:0" 

bedrock = boto3.client("bedrock-runtime", region_name=REGION)
rerank_client = boto3.client("bedrock-agent-runtime", region_name=RERANK_REGION)

def get_embedding(text):
    body = json.dumps({"inputText": text})

    response = bedrock.invoke_model(
        modelId=EMBED_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json"
    )

    response_body = json.loads(response["body"].read())
    return response_body["embedding"]

def load_index(index_file, metadata_file):
    """Loads a previously saved FAISS index + chunk metadata."""
    index = faiss.read_index(index_file)
    with open(metadata_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks

def retrieve_top_k(query, index, chunks, k=10):
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

def rerank_chunks(query, candidates, top_k=4, max_retries=5):
    """Reranks the candidate chunks using the Rerank model."""
    sources = [
        {
            "inlineDocumentSource": {
                "type": "TEXT",
                "textDocument": {"text": c["text"]}
            },
            "type": "INLINE"
        }
        for c in candidates
    ]

    for attempt in range(max_retries):
        try:
            response = rerank_client.rerank(
                queries=[{"type": "TEXT", "textQuery": {"text": query}}],
                sources=sources,
                rerankingConfiguration={
                    "type": "BEDROCK_RERANKING_MODEL",
                    "bedrockRerankingConfiguration":{
                        "modelConfiguration": {
                            "modelArn": f"arn:aws:bedrock:{RERANK_REGION}::foundation-model/{RERANK_MODEL_ID}"
                        },
                        "numberOfResults": top_k
                    }
                }
            )
            break
        except Exception as e:
            if "Throttling" in str(e) and attempt < max_retries -1:
                wait = 2 ** attempt
                print(f"  Rerank throttled, waiting {wait}s...")
                time.sleep(wait)
                continue
            raise
    reranked = []
    for r in response["results"]:
        original_chunk = candidates[r["index"]]
        reranked.append({
            **original_chunk,
            "rerank_score": r["relevanceScore"]
        })

    return reranked

if __name__ == "__main__":
    index, chunks = load_index(INDEX_FILE, METADATA_FILE)
    print(f"Loaded index with {index.ntotal} vectors and {len(chunks)} chunks.")

    query = "What is Amazon API Gateway?"
    results = retrieve_top_k(query, index, chunks, k=3)
    print(f"Query: {query}")
    for i, r in enumerate(results):
        print(f"Result{i+1}: Source: {r['source']}, Distance: {r['distance']:.4f}")
        print(f"Text: {r['text']}")
        print("-" * 80)
