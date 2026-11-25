import json
import time

import streamlit as st

import rag_pipeline as rp
from rag_pipeline import load_bm25, load_index, retrieve_context
from src._3retrieval.retriever import BM25_FILE, INDEX_FILE, METADATA_FILE
from src._5llm.llm_client import stream_response_from_context

NO_ANSWER = getattr(rp, "NO_ANSWER", "I couldn't find this in the AWS documentation I have indexed (Agent Toolkit, CloudSearch, CDK, EventBridge, VPC). Try rephrasing, or ask about one of those topics.")
MAX_QUESTIONS_PER_SESSION = 20

st.set_page_config(page_title="AWS Docs Assistant")
st.title("AWS Docs Assistant")
st.caption("Answers come only from the indexed AWS user guides. Questions outside them are declined.")


@st.cache_resource(show_spinner="Loading the knowledge base...")
def load_all():
    index, chunks = load_index(INDEX_FILE, METADATA_FILE)
    return index, chunks, load_bm25(BM25_FILE)


index, chunks, bm25 = load_all()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything about the indexed AWS docs "
         "(Agent Toolkit, CloudSearch, CDK, EventBridge, VPC).", "meta": None}
    ]
    st.session_state.n_questions = 0

def typewriter(stream, delay=0.02):
    for piece in stream:
        for word in piece.split(" "):
            yield word + " "
            time.sleep(delay)
            
def send_feedback(idx: int, value: bool):
    meta = st.session_state.messages[idx]["meta"]
    record = {"ts": time.time(), "question": meta["question"],
              "answer": meta["answer"], "thumbs_up": value}
    print(json.dumps(record), flush=True)  # visible in the Streamlit Cloud logs
    meta["feedback"] = "up" if value else "down"

import re

GREETING = (
    "Hi! I'm an assistant for AWS documentation. I can answer questions about the Agent Toolkit for AWS, "
    "Amazon CloudSearch, the AWS CDK, Amazon EventBridge and Amazon VPC. Try: "
    "\"What is an event pattern in EventBridge?\""
)
_HI = r"(?:hi+|hello|hey+|good (?:morning|afternoon|evening))(?: there)?"
SMALLTALK = [
    (rf"{_HI}", GREETING),
    (rf"(?:{_HI}\s*)?(?:how can you help(?: me)?|what can you do|who are you|what are you|what do you do)", GREETING),
    (r"(?:thanks|thank you|thx|great|cool|ok|okay)", "You're welcome! Ask me anything else about the AWS docs."),
    (r"(?:bye|goodbye|see you)", "Goodbye! Come back anytime."),
]

def smalltalk_reply(q: str):
    t = " ".join(re.sub(r"[^\w\s']", " ", q.lower()).split())
    for pattern, reply in SMALLTALK:
        if re.fullmatch(pattern, t):
            return reply
    return None

def run_question(question: str) -> dict:
    reply = smalltalk_reply(question)
    if reply:
        return {"content": reply, "meta": None}
    if st.session_state.n_questions >= MAX_QUESTIONS_PER_SESSION:
        return {"content": "Session limit reached. Please come back later.", "meta": None}
    st.session_state.n_questions += 1
    try:
        with st.spinner("Searching the docs..."):
            top, refused = retrieve_context(question, index, chunks, bm25)
        if refused:
            return {"content": "I couldn't find this in the AWS documentation I have indexed.", "meta": None}
        answer = st.write_stream(typewriter(stream_response_from_context(question, [c["text"] for c in top])))
    except Exception as e:
        return {"content": f"Something went wrong ({type(e).__name__}). Please try again.", "meta": None}
    if answer.strip() == NO_ANSWER:
        return {"content": "I couldn't find this in the AWS documentation I have indexed.", "meta": None}
    return {"content": answer,
            "meta": {"sources": sorted({c["source"] for c in top}), "question": question,
                     "answer": answer, "feedback": None}}


for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        meta = m.get("meta")
        if m["role"] == "assistant" and meta:
            if meta["sources"]:
                with st.expander("Sources"):
                    for s in meta["sources"]:
                        st.write(f"- {s}")
            if meta["feedback"] is None:
                c1, c2, _ = st.columns([1, 1, 8])
                c1.button("👍", key=f"up_{i}", on_click=send_feedback, args=(i, True))
                c2.button("👎", key=f"down_{i}", on_click=send_feedback, args=(i, False))
            else:
                st.caption("Thanks for the feedback!")

if prompt := st.chat_input("Ask about the AWS docs..."):
    prompt = prompt.strip()[:500]
    st.session_state.messages.append({"role": "user", "content": prompt, "meta": None})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        reply = run_question(prompt)
    st.session_state.messages.append({"role": "assistant", **reply})
    st.rerun()