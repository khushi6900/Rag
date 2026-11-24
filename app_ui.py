import os, json, time
import requests
import streamlit as st
from rag_pipeline import ask, load_index, load_bm25

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
NO_ANSWER = "I cannot find the answer in the provided context."
INDEX_FILE = "faiss_index.bin"
BM25_FILE = "bm25_index.pkl"
METADATA_FILE = "chunk_metadata.json"

st.set_page_config(page_title="AWS Docs Assistant", page_icon="☁️")
st.title("☁️ AWS Docs Assistant")
st.caption("Answers come only from the indexed AWS user guides. Questions outside them are declined.")

http = requests.Session()
http.trust_env = False  # ignore system proxy settings for local calls

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything about the indexed AWS docs "
         "(Agent Toolkit, CloudSearch, CDK, EventBridge, VPC).", "meta": None}
    ]

@st.cache_resource
def load_resources():
    index, chunks = load_index(INDEX_FILE, METADATA_FILE)
    bm25 = load_bm25(BM25_FILE)
    return index, chunks, bm25

def send_feedback(idx: int, value: bool):
    meta = st.session_state.messages[idx]["meta"]
    record = {"ts": time.time(), "question": meta["question"], "answer": meta["answer"], "thumbs_up": value}
    print(json.dumps(record), flush=True)          # appears in the Streamlit Cloud logs
    meta["feedback"] = "up" if value else "down"

index, chunks, bm25 = load_resources()
def ask_api(question: str) -> dict:
    n = st.session_state.get("n_questions", 0)
    if n >= 20:
        return {"content": "Session limit reached (20 questions). Please come back later.", "meta": None}
    st.session_state["n_questions"] = n + 1
    try:
        data = ask(question, index, chunks, bm25)
    except Exception as e:
        return {"content": f"Something went wrong ({type(e).__name__}). Please try again.", "meta": None}
    refused = data["answer"].strip() == NO_ANSWER or not data.get("sources")
    content = "I couldn't find this in the AWS documentation I have indexed." if refused else data["answer"]
    return {"content": content,
            "meta": {"sources": [] if refused else data["sources"], "question": question,
                     "answer": data["answer"], "feedback": None, "refused": refused}}

# render history
for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        meta = m.get("meta")
        if m["role"] == "assistant" and meta:
            if meta["sources"]:
                with st.expander("Sources"):
                    for s in meta["sources"]:
                        if isinstance(s, dict):
                            source = s.get("source", "unknown")
                            score = s.get("score", "N/A")
                            st.write(f"- {source} (relevance: {score})")
                        else:
                            st.write(f"- {s}")
            if meta["feedback"] is None:
                c1, c2, _ = st.columns([1, 1, 8])
                c1.button("👍", key=f"up_{i}", on_click=send_feedback, args=(i, True))
                c2.button("👎", key=f"down_{i}", on_click=send_feedback, args=(i, False))
            else:
                st.caption("Thanks for the feedback!")

# new question
if prompt := st.chat_input("Ask about the AWS docs..."):
    prompt = prompt.strip()[:500]
    st.session_state.messages.append({"role": "user", "content": prompt, "meta": None})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"), st.spinner("Searching the docs..."):
        reply = ask_api(prompt)
    st.session_state.messages.append({"role": "assistant", **reply})
    st.rerun()
