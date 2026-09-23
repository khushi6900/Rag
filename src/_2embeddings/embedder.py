import json
import boto3
import numpy as np
import faiss
from tqdm import tqdm
import re

# ---- CONFIG ----
REGION = "us-east-1" 
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"  # Titan Embeddings v2
CHUNKS_FILE = "all_chunks.json"
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "chunk_metadata.json"

bedrock = boto3.client("bedrock-runtime", region_name=REGION)

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

def is_low_quality(text):
    if re.search(r'\.{10,}', text): # consecutive dots
        return True

    # digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
    # if digit_ratio > 0.15:
    #     return True

    # lines = [l for l in text.split("\n") if l.strip()]
    # if lines:
    #     short_lines = sum(1 for l in lines if len(l.strip()) < 40)
    #     if short_lines / len(lines) > 0.7 and len(lines) > 3:
    #         return True
        
    return False

def embed_all_chunks(chunks):
    vectors=[]
    valid_chunks=[]

    for chunk in tqdm(chunks, desc="Embedding chunks", unit="chunk"):
        text = chunk["text"]
        if not text.strip() or is_low_quality(text):
            print(f"Skipping low-quality or empty chunk : {text[:30]}...")
            continue

        try:
            embedding = get_embedding(text)
            vectors.append(embedding)
            valid_chunks.append(chunk)
        except Exception as e:
            tqdm.write(f"Error embedding chunk: {e}. Skipping this chunk.")

    vectors_np = np.array(vectors, dtype="float32")
    return vectors_np, valid_chunks

def build_faiss_index(vectors_np):
    dimension = vectors_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors_np)
    return index

if __name__ == "__main__":
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    vectors_np, valid_chunks = embed_all_chunks(chunks)
    print(f"Successfully embedded {len(valid_chunks)} chunks")
    print(f"Embedding shape: {vectors_np.shape}")
    print(f"Vector dimension: {vectors_np.shape[1]}")

    index = build_faiss_index(vectors_np)
    print(f"FAISS index built with {index.ntotal} vectors")

    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(valid_chunks, f, ensure_ascii=False, indent=2)

    print(f"FAISS index saved to {INDEX_FILE}")