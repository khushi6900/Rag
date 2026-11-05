import json
import boto3
import numpy as np
import faiss
from tqdm import tqdm
import re
import time
from rank_bm25 import BM25Okapi
import pickle
from sentence_transformers import SentenceTransformer


# ---- CONFIG ----
REGION = "us-east-1"
# EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
CHUNKS_FILE = "all_chunks.json"
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "chunk_metadata.json"
BM25_FILE = "bm25_index.pkl"
CHECKPOINT_FILE = "embed_checkpoint.json"
CHECKPOINT_EVERY = 500

# bedrock = boto3.client("bedrock-runtime", region_name=REGION)
model = SentenceTransformer(EMBED_MODEL_ID)

# def get_embedding(text, max_retries=5):
#     """Calls Titan embedding with exponential backoff on throttling."""
#     body = json.dumps({"inputText": text})

#     for attempt in range(max_retries):
#         try:
#             response = bedrock.invoke_model(
#                 modelId=EMBED_MODEL_ID,
#                 body=body,
#                 contentType="application/json",
#                 accept="application/json"
#             )
#             response_body = json.loads(response["body"].read())
#             return response_body["embedding"]
#         except Exception as e:
#             if "Throttling" in str(e) and attempt < max_retries - 1:
#                 wait = 2 ** attempt
#                 tqdm.write(f"  Throttled, waiting {wait}s (attempt {attempt+1}/{max_retries})")
#                 time.sleep(wait)
#                 continue
#             raise  # real error, or out of retries — let caller handle/log it

def is_low_quality(text):
    if re.search(r'\.{10,}', text):  # consecutive dots
        return True
    return False

# def get_embedding(text):
#     """calls BGE embedding model to get embedding for a given text."""
#     result = model.encode(text)
#     print(f"shapeeeeee: {result.shape}")
#     return model.encode(text)

def embed_all_chunks(chunks, batch_size=32):
    # vectors = []
    valid_chunks = []
    texts=[]

    # 1. Filter out low-quality or empty chunks first
    for chunk in tqdm(chunks, desc="Embedding chunks", unit="chunk"):
        text = chunk["text"]
        if not text.strip() or is_low_quality(text):
            tqdm.write(f"Skipping low-quality or empty chunk: {text[:30]}...")
            continue
        valid_chunks.append(chunk)
        texts.append(text)
    print(f"Embedding {len(text)} valid chunks in batches..")

    # 2. Encode all texts in parallel/batches natively
    vectors_np = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype("float32")

    print(f"Shape of all embeddings: {vectors_np.shape}")
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

    vectors_np, valid_chunks = embed_all_chunks(chunks,batch_size=32)
    print(f"Successfully embedded {len(valid_chunks)} chunks")
    print(f"Embedding shape: {vectors_np.shape}")
    print(f"Vector dimension: {vectors_np.shape[1]}")

    index = build_faiss_index(vectors_np)
    print(f"FAISS index built with {index.ntotal} vectors")

    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(valid_chunks, f, ensure_ascii=False, indent=2)

    print(f"FAISS index saved to {INDEX_FILE}")

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        valid_chunks = json.load(f)
 
    print(f"Loaded {len(valid_chunks)} chunks from {METADATA_FILE} for BM25 indexing")
 
    bm25 = build_bm25_index(valid_chunks)
    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25, f)
 
    print(f"BM25 index saved to {BM25_FILE}")