import json
import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from langfuse import get_client
from pydantic import BaseModel

from src._3retrieval.retriever import (
    hybrid_retrieve,
    load_bm25,
    load_index,
    rerank_chunks,
)
from src._5llm.llm_client import generate_response_from_context

state = {}
REFUSE_BELOW = 0.30
NO_ANSWER = "I cannot find the answer in the provided context."
hits = defaultdict(list)     # in-memory rate limit per client

log = logging.getLogger("rag")
logging.basicConfig(level=logging.INFO, format="%(message)s")

def timed(fn, *args, **kargs):
    t= time.perf_counter()
    out = fn(*args, **kargs)
    return out, round((time.perf_counter() -t) *1000)

@asynccontextmanager
async def lifespan(app):
    print("Langfuse auth:", get_client().auth_check())
    state["index"], state["chunks"] = load_index(index_file="faiss_index.bin", metadata_file="chunk_metadata.json")
    state["bm25"] = load_bm25("bm25_index.pkl")
    yield
    get_client().flush() 

app = FastAPI(lifespan=lifespan)

class Ask(BaseModel):
    query: str
class Feedback(BaseModel):
    trace_id: str
    thumbs_up: bool

def run(q: str):
    pool, t_ret = timed(hybrid_retrieve, query=q, index=state["index"], chunks=state["chunks"], bm25=state["bm25"], k=20)
    top, t_rr = timed(rerank_chunks, query=q, candidates=pool, top_k=5)
    if not top[0].get("fallback") and top[0]["rerank_score"] < REFUSE_BELOW:
        return NO_ANSWER, []
    # print("inside run, trace id:", get_client().get_current_trace_id())
    (answer, usage), t_llm = timed(generate_response_from_context, query=q, context=[c["text"] for c in top])
    log.info(json.dumps({
        "event": "ask", "retrieve_ms": t_ret, "rerank_ms": t_rr, "llm_ms": t_llm,
        "in_tok": usage.get("input_tokens"), "out_tok": usage.get("output_tokens"),
        "top_score": top[0]["rerank_score"], "refused": False
    }))
    sources = [{"source": c.get("source"), "score": round(c["rerank_score"],3)} for c in top]
    return answer, sources

@app.post("/ask")
def ask(body: Ask, request: Request):
    ip, now = request.client.host, time.time()
    hits[ip] = [t for t in hits[ip] if now - t < 60]
    if len(hits[ip]) >= 10:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Max 10 requests per minute.")
    hits[ip].append(now)
    q = body.query.strip()
    if not q or len(q) > 500:
        raise HTTPException(status_code=400, detail="Invalid question. Must be non-empty and <= 500 characters.")
    trace_id = get_client().create_trace_id()
    answer, sources = run(q)
    get_client().flush()
    return {"answer": answer, "sources": sources, "trace_id": trace_id}

@app.post("/feedback")
def feedback(body: Feedback):
    get_client().create_score(trace_id=body.trace_id, name="user_feedback", value=1 if body.thumbs_up else 0)
    return {"ok": True}

@app.get("/health")
def health():
    return {"status": "ok"}