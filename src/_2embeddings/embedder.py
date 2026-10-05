import json
import boto3
import numpy as np
import faiss
from tqdm import tqdm
import re
import time
from rank_bm25 import BM25Okapi
import pickle

# ---- CONFIG ----
REGION = "us-east-1"
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
CHUNKS_FILE = "all_chunks.json"
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "chunk_metadata.json"
CHECKPOINT_FILE = "embed_checkpoint.json"
CHECKPOINT_EVERY = 500

bedrock = boto3.client("bedrock-runtime", region_name=REGION)

def get_embedding(text, max_retries=5):
    """Calls Titan embedding with exponential backoff on throttling."""
    body = json.dumps({"inputText": text})

    for attempt in range(max_retries):
        try:
            response = bedrock.invoke_model(
                modelId=EMBED_MODEL_ID,
                body=body,
                contentType="application/json",
                accept="application/json"
            )
            response_body = json.loads(response["body"].read())
            return response_body["embedding"]
        except Exception as e:
            if "Throttling" in str(e) and attempt < max_retries - 1:
                wait = 2 ** attempt
                tqdm.write(f"  Throttled, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            raise  # real error, or out of retries — let caller handle/log it

def is_low_quality(text):
    if re.search(r'\.{10,}', text):  # consecutive dots
        return True
    return False

def embed_all_chunks(chunks):
    vectors = []
    valid_chunks = []

    for i, chunk in enumerate(tqdm(chunks, desc="Embedding chunks", unit="chunk")):
        text = chunk["text"]
        if not text.strip() or is_low_quality(text):
            tqdm.write(f"Skipping low-quality or empty chunk: {text[:30]}...")
            continue

        try:
            embedding = get_embedding(text)
            vectors.append(embedding)
            valid_chunks.append(chunk)
        except Exception as e:
            tqdm.write(f"Error embedding chunk (permanently skipping): {e}")

        # periodic checkpoint — protects against losing hours of progress on a crash
        if (i + 1) % CHECKPOINT_EVERY == 0:
            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "processed_count": i + 1,
                    "valid_chunks": valid_chunks,
                    "vectors": vectors
                }, f)
            tqdm.write(f"  Checkpoint saved at chunk {i+1}/{len(chunks)}")

    vectors_np = np.array(vectors, dtype="float32")
    return vectors_np, valid_chunks


def build_faiss_index(vectors_np):
    dimension = vectors_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors_np)
    return index

def build_bm25_index(chunks):
    """Tokenizes chunk text and builds a BM25 index. Run once, save alongside FAISS."""
    tokenized_corpus = [chunk["text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

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