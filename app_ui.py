import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
NO_ANSWER = "I cannot find the answer in the provided context."

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


def send_feedback(idx: int, value: bool):
    msg = st.session_state.messages[idx]
    try:
        http.post(f"{API_URL}/feedback",
                  json={"trace_id": msg["meta"].get("trace_id", ""), "thumbs_up": value},
                  timeout=10)
    except requests.RequestException:
        pass
    msg["meta"]["feedback"] = "up" if value else "down"


def ask_api(question: str) -> dict:
    try:
        r = http.post(f"{API_URL}/ask", json={"query": question}, timeout=60)
        if r.status_code == 429:
            return {"content": "Too many requests. Please wait a minute and try again.", "meta": None}
        if r.status_code == 400:
            return {"content": "Please enter a question of 1-500 characters.", "meta": None}
        r.raise_for_status()
        data = r.json()
        refused = data["answer"].strip() == NO_ANSWER or not data.get("sources")
        content = ("I couldn't find this in the AWS documentation I have indexed."
                   if refused else data["answer"])
        return {"content": content,
                "meta": {"sources": [] if refused else data.get("sources", []),
                         "trace_id": data.get("trace_id", ""), "feedback": None, "refused": refused}}
    except requests.RequestException as e:
        return {"content": f"The service is unavailable right now ({type(e).__name__}). Please try again shortly.",
                "meta": None}


# render history
for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        meta = m.get("meta")
        if m["role"] == "assistant" and meta:
            if meta["sources"]:
                with st.expander("Sources"):
                    for s in meta["sources"]:
                        st.write(f"- {s.get('source', 'unknown')} (relevance {s.get('score')})")
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
    with st.chat_message("assistant"):
        with st.spinner("Searching the docs..."):
            reply = ask_api(prompt)
    st.session_state.messages.append({"role": "assistant", **reply})
    st.rerun()
