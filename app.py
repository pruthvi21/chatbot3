"""
app.py — the Cancer Specialist web app (Streamlit UI).
=======================================================

Big picture: this is a chatbot that only answers from a library of cancer
research papers. It combines THREE ideas:

  1. RAG (see rag.py) — find the paper excerpts relevant to the question.
  2. Memory          — remember earlier turns so follow-ups make sense.
  3. Streamlit UI    — a chat window + a sidebar of separate conversations.

WHAT HAPPENS WHEN A USER ASKS A QUESTION (the request flow):

    user types a question
        │
        ▼
    retriever.invoke(question)     ← rag.py: find the 4 most relevant chunks
        │
        ▼
    format_context(chunks)         ← paste those chunks in as {context}
        │
        ▼
    prompt = system + history + (context + question)
        │
        ▼
    Groq Llama model               ← generates the answer, streamed live
        │
        ▼
    answer + 📎 sources shown, and saved into this session's memory

Read this file top-to-bottom; the sections are ordered the way they run.
"""

import os
import uuid

import streamlit as st
from dotenv import load_dotenv

# LangChain pieces for the CHAT side (the RAG side is imported from rag.py).
from langchain_groq import ChatGroq                                    # the LLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser             # msg -> str
from langchain_core.chat_history import InMemoryChatMessageHistory    # a memory box
from langchain_core.runnables.history import RunnableWithMessageHistory

# Our own RAG helpers.
from rag import (
    clear_retriever_cache,
    format_context,
    get_retriever,
    save_uploaded_pdfs,
    source_list,
)

# ==========================================================================
# 1. CONFIG — load the API key and define the assistant's personality
# ==========================================================================

# Read GROQ_API_KEY: first from this folder's .env, then the parent project's.
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

MODEL_NAME = "llama-3.3-70b-versatile"  # the Groq model that writes the answers

# The "system" message = the assistant's standing instructions. This is where
# we turn a generic chatbot into a grounded, careful cancer specialist.
SPECIALIST_SYSTEM = (
    "You are an oncology research assistant grounded in a curated library of "
    "cancer research papers. Answer precisely and clearly, based ONLY on the "
    "excerpts provided to you. Cite what the excerpts say. If the excerpts do "
    "not contain the answer, say you don't have that information in your "
    "library rather than guessing. Use the conversation history to resolve "
    "follow-up questions. You are an educational research tool, not a "
    "substitute for a qualified oncologist — remind users to consult a medical "
    "professional for any personal diagnosis or treatment decision."
)

st.set_page_config(page_title="Cancer Specialist", page_icon="🔬")


# ==========================================================================
# 2. SESSION STORE — keep several separate conversations (like ChatGPT tabs)
# ==========================================================================
# Streamlit reruns this whole script on every click. Normal variables reset
# each rerun, so we need somewhere durable to keep conversations.
#
#   @st.cache_resource  -> one shared object for the whole app run. It even
#                          survives a browser refresh (which is why past
#                          sessions stay in the sidebar). It resets only when
#                          the app process is stopped/restarted.
@st.cache_resource
def get_store():
    return {
        "sessions": {},   # session_id -> {label, messages, history, titled}
        "order": [],      # session_ids in creation order (for the sidebar list)
        "counter": 0,     # running number used for default labels
        "active": None,   # the session_id currently being viewed
    }


def new_session(store):
    """Start a brand-new conversation and make it the active one."""
    store["counter"] += 1
    sid = str(uuid.uuid4())               # unique id used as the memory key
    store["sessions"][sid] = {
        "label": f"Consult {store['counter']}",   # sidebar name (until 1st msg)
        "messages": [],                            # what we DISPLAY on screen
        "history": InMemoryChatMessageHistory(),   # what the MODEL remembers
        "titled": False,                           # renamed from 1st question?
    }
    store["order"].append(sid)
    store["active"] = sid
    return sid


def delete_session(store, sid):
    """Remove a conversation; fall back to another (or none)."""
    store["sessions"].pop(sid, None)
    if sid in store["order"]:
        store["order"].remove(sid)
    if store["active"] == sid:
        store["active"] = store["order"][-1] if store["order"] else None


# ==========================================================================
# 3. THE CHAIN — prompt + memory + LLM, wired together with LCEL pipes
# ==========================================================================
# cache_resource again: build the chain once and reuse it. We key it by
# `temperature` so changing the slider rebuilds it with the new setting.
@st.cache_resource
def build_chat(temperature: float):
    # Guard: fail clearly if the API key is missing/wrong.
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not api_key.startswith("gsk_"):
        st.error(
            "GROQ_API_KEY not found. Add it to memory_chatbot/.env or the "
            "project .env (must start with 'gsk_')."
        )
        st.stop()

    llm = ChatGroq(model=MODEL_NAME, temperature=temperature)

    # The prompt template has FOUR parts. {context} and {input} get filled in
    # at question time; the history is injected automatically (see below).
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SPECIALIST_SYSTEM),      # who the assistant is
            MessagesPlaceholder("history"),     # <- past turns land here
            ("human",                           # the actual user turn
             "Relevant excerpts from the cancer research library:\n\n"
             "{context}\n\n---\n"
             "Question: {input}\n\n"
             "Answer grounded in the excerpts above. For follow-up questions "
             "(e.g. 'summarize that') you may also use the earlier "
             "conversation. If the needed facts are in neither, say you don't "
             "have that information in your library."),
        ]
    )

    # LCEL "pipe": data flows prompt -> model -> string parser.
    convo = prompt | llm | StrOutputParser()

    # Memory wiring: for a given session_id, return that session's history box.
    store = get_store()

    def get_history(session_id: str):
        session = store["sessions"].get(session_id)
        return session["history"] if session else InMemoryChatMessageHistory()

    # RunnableWithMessageHistory automatically:
    #   - reads the session's past turns and injects them into {history}
    #   - saves the new question + answer back into that history
    # so each session keeps its own separate memory.
    return RunnableWithMessageHistory(
        convo,
        get_history,
        input_messages_key="input",     # which input key is the user's message
        history_messages_key="history",  # which prompt slot holds the history
    )


# ==========================================================================
# 4. BOOT — index the library and make sure a session exists
# ==========================================================================
# This runs the ingestion pipeline in rag.py (or loads the cached index).
with st.spinner("Indexing the cancer research library… (first run downloads the embedding model)"):
    retriever, n_docs, n_chunks = get_retriever(k=4)

store = get_store()
if store["active"] is None:   # first ever load -> open one conversation
    new_session(store)


# ==========================================================================
# 5. SIDEBAR — conversation list, library stats, settings
# ==========================================================================
st.sidebar.title("🔬 Consultations")

# ＋ button: start a fresh conversation.
if st.sidebar.button("＋ New consult", use_container_width=True):
    new_session(store)
    st.rerun()   # rerun immediately so the new session shows as active

# One row per conversation (newest first): a name button + a delete button.
for sid in reversed(store["order"]):
    session = store["sessions"][sid]
    is_active = sid == store["active"]
    cols = st.sidebar.columns([0.8, 0.2])
    # Clicking the name switches to that conversation.
    if cols[0].button(
        session["label"], key=f"pick_{sid}", use_container_width=True,
        type="primary" if is_active else "secondary",  # highlight the active one
    ):
        store["active"] = sid
        st.rerun()
    # 🗑️ deletes it.
    if cols[1].button("🗑️", key=f"del_{sid}", use_container_width=True):
        delete_session(store, sid)
        if store["active"] is None:
            new_session(store)
        st.rerun()

st.sidebar.divider()
st.sidebar.subheader("📚 Library")
if n_docs:
    st.sidebar.caption(f"{n_docs} papers · {n_chunks} chunks indexed")
else:
    st.sidebar.warning("No PDFs indexed yet. Upload one below.")

# --------------------------------------------------------------------------
# Upload PDFs and add them to the RAG library while the app is running.
# --------------------------------------------------------------------------
# accept_multiple_files lets a student add several papers in one operation.
# The files are not vectorized in this UI block itself. Instead:
#   1. rag.py saves them into docs/
#   2. we clear Streamlit's cached retriever
#   3. st.rerun() starts the boot section again
#   4. get_retriever() detects changed docs and rebuilds the FAISS index
uploaded_pdfs = st.sidebar.file_uploader(
    "Add research PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    help="Uploaded PDFs are saved in docs/ and indexed immediately.",
)

if st.sidebar.button(
    "Upload and index",
    use_container_width=True,
    disabled=not uploaded_pdfs,
):
    with st.sidebar.status("Adding papers to the library…", expanded=True):
        saved, skipped = save_uploaded_pdfs(uploaded_pdfs)

        if saved:
            st.write(f"Saved {len(saved)} PDF(s). Rebuilding the index…")
            clear_retriever_cache()
            # Store a one-time message because the current page is about to rerun.
            st.session_state["upload_result"] = {
                "saved": saved,
                "skipped": skipped,
            }
        elif skipped:
            st.session_state["upload_result"] = {
                "saved": [],
                "skipped": skipped,
            }
    st.rerun()

# Show upload feedback once, after indexing has completed on the rerun.
if result := st.session_state.pop("upload_result", None):
    if result["saved"]:
        st.sidebar.success(
            f"Indexed {len(result['saved'])} new PDF(s): "
            + ", ".join(result["saved"])
        )
    if result["skipped"]:
        st.sidebar.info(
            "Already in the library: " + ", ".join(result["skipped"])
        )

st.sidebar.subheader("⚙️ Settings")
# Temperature: 0 = focused/factual, 1 = more creative. Low is best for RAG.
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
st.sidebar.caption(f"Model: `{MODEL_NAME}`")

# Build (or reuse) the chain and grab the conversation we're viewing.
chat = build_chat(temperature)
active = store["active"]
current = store["sessions"][active]


# ==========================================================================
# 6. MAIN AREA — header + the current conversation
# ==========================================================================
st.title("🔬 Cancer Specialist")
st.caption(
    "Answers are grounded in an indexed library of cancer research papers, with "
    "sources cited. Educational tool — **not** medical advice; consult a "
    "qualified oncologist for personal decisions."
)

# Replay this session's past turns so the chat history is visible on screen.
for msg in current["messages"]:
    with st.chat_message(msg["role"]):          # "user" or "assistant"
        st.markdown(msg["content"])
        if msg.get("sources"):                  # show citations under answers
            with st.expander("📎 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")


# ==========================================================================
# 7. HANDLE A NEW QUESTION — the RAG request flow, step by step
# ==========================================================================
# st.chat_input returns the typed text (or None if nothing was submitted).
if question := st.chat_input("Ask about the cancer research library…"):

    # 7a. Record + show the user's message.
    current["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 7b. Rename the session after its first question (ChatGPT-style titles).
    if not current["titled"]:
        current["label"] = (question[:30] + "…") if len(question) > 30 else question
        current["titled"] = True

    # 7c. RETRIEVE: ask the vector index for the most relevant paper excerpts.
    docs = retriever.invoke(question) if retriever else []
    context = format_context(docs) if docs else "(no library available)"
    sources = source_list(docs)

    # 7d. GENERATE: run the chain. The config's session_id tells the memory
    #     wrapper which conversation's history to use. .stream() yields tokens
    #     as they arrive, and st.write_stream prints them live.
    with st.chat_message("assistant"):
        cfg = {"configurable": {"session_id": active}}
        stream = chat.stream({"input": question, "context": context}, config=cfg)
        answer = st.write_stream(stream)
        # 7e. Show which papers grounded this answer.
        if sources:
            with st.expander("📎 Sources"):
                for s in sources:
                    st.markdown(f"- {s}")

    # 7f. Save the answer (with its sources) into what we display next rerun.
    current["messages"].append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    st.rerun()  # rerun so the sidebar title/state refreshes cleanly
