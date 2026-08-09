# 🔬 Cancer Specialist (RAG)

A Streamlit app that answers questions **grounded in a library of cancer
research papers**. On boot it indexes every PDF in `./docs` into a FAISS vector
store, then retrieves relevant excerpts to ground each answer — with sources
cited. It also keeps per-session memory, so you can ask a **sequence** of
follow-up questions.

Built on the LangChain sessions: Part 7 memory + Session 43 RAG.

> ⚠️ **Educational research tool — not medical advice.** Answers come from the
> indexed papers only. Consult a qualified oncologist for any personal decision.

## How it works
- **RAG** (`rag.py`) — `PyPDFLoader` → `RecursiveCharacterTextSplitter` →
  `all-MiniLM-L6-v2` embeddings → **FAISS**. The index is cached in
  `.faiss_index/` and only rebuilt when the docs change.
- **Grounding** — top-k excerpts are injected into the prompt; the assistant is
  told to answer only from them and cite sources.
- **Memory** — `MessagesPlaceholder("history")` + `RunnableWithMessageHistory`
  give each session its own conversation memory.
- **Multiple consults** — a sidebar lists conversations, **＋ New consult**
  starts a fresh one. Sessions live in memory: they survive a browser refresh
  but reset when the app restarts.

## The document library
Cancer PDFs live in `memory_chatbot/docs/`. You can add documents in two ways:

1. **From the app:** open the sidebar, choose one or more files under
   **Add research PDFs**, then click **Upload and index**. The app saves and
   vectorizes them immediately; no restart is needed.
2. **Manually:** copy PDFs into `docs/` and restart the app.

Only PDF uploads are accepted. An identical file is skipped. If a different
PDF has the same filename, the app keeps both by adding `_2`, `_3`, etc.

---

## 🧠 Under the hood — how it actually works

This section walks through the two flows that make the app tick: **ingestion**
(done once, on boot) and **request processing** (done per question). Everything
below maps to real functions in `rag.py` and `app.py`.

### Flow A — Ingestion (building the searchable library)

Runs on boot, inside `get_retriever()` in [`rag.py`](rag.py). The goal: turn a
pile of PDFs the LLM has never read into something we can *search by meaning*.

```
docs/*.pdf
   │  ① PyPDFLoader.load()            read each PDF → one "Document" per page
   ▼                                   (text + metadata: source file, page #)
pages
   │  ② RecursiveCharacterTextSplitter cut pages into ~1000-char chunks
   ▼      (chunk_overlap=150)           overlap so sentences aren't cut in half
chunks
   │  ③ HuggingFaceEmbeddings          each chunk → a 384-number vector
   ▼      (all-MiniLM-L6-v2, local)     "similar meaning ⇒ nearby vectors"
vectors
   │  ④ FAISS.from_documents()         store vectors in a searchable index
   ▼
.faiss_index/  ─── ⑤ saved to disk + a signature.txt fingerprint
```

**Why chunks, not whole pages?** Embedding models have a limited window, and a
question usually matches *one paragraph*, not a whole page. Small chunks =
sharper, more relevant retrieval.

**Why the fingerprint?** `_docs_signature()` hashes every PDF's name + size +
modified-time. On the next boot we compare it to the saved one:
- **same** → load the cached index instantly (no re-embedding).
- **changed** (you added/edited/removed a PDF) → rebuild automatically.

That's why the first run is slow (embed everything) and later runs start in
seconds.

### What happens when a PDF is uploaded

The upload UI lives in `app.py`, while the document-processing logic remains
in `rag.py`:

```
file_uploader (app.py)
   │
   ▼
save_uploaded_pdfs() (rag.py)  validate the name and save into docs/
   │
   ▼
clear_retriever_cache()        discard the old in-memory retriever
   │
   ▼
st.rerun()                     run the app's boot section again
   │
   ▼
get_retriever() (rag.py)       fingerprint changed → load, split, embed
   │
   ▼
new FAISS index                old + newly uploaded PDFs are now searchable
```

The index is rebuilt from **all** PDFs in `docs/`. This keeps the teaching
example simple and guarantees that the index matches the folder. For a very
large production library, incremental insertion into a persistent vector
database would be more efficient.

### Flow B — Request processing (answering one question)

Runs every time the user hits enter, in **section 7** of [`app.py`](app.py):

```
① user types a question
      │
      ▼
② retriever.invoke(question)      embed the QUESTION the same way, then ask
      │                            FAISS for the k=4 nearest chunks
      ▼
③ format_context(chunks)          stitch those chunks into a numbered,
      │                            source-tagged {context} block
      ▼
④ prompt = system + history + (context + question)
      │        │        │           │
      │   persona   past turns   this turn, grounded in the excerpts
      ▼
⑤ ChatGroq (Llama 3.3)            generates the answer, streamed token-by-token
      │
      ▼
⑥ st.write_stream → screen        + a 📎 Sources expander (which papers)
      │
      ▼
⑦ RunnableWithMessageHistory      saves Q + A into THIS session's memory,
                                   so the next follow-up has context
```

**The key idea (RAG):** the LLM never memorised these papers. Step ② finds the
right passages and step ④ pastes them into the prompt, so the model answers
*from the excerpts in front of it* rather than from vague training memory. The
system prompt tells it to say *"I don't have that in my library"* when the
excerpts don't cover the question — that's what keeps it from making things up.

**Where memory fits:** `RunnableWithMessageHistory` (built in `build_chat()`)
automatically injects the session's earlier turns into the `{history}` slot and
saves each new turn back. Each sidebar consultation has its own history box, so
conversations stay separate.

### Which function does what

| Step | Function | File |
|------|----------|------|
| Load PDFs → pages | `PyPDFLoader` (in `get_retriever`) | `rag.py` |
| Split pages → chunks | `RecursiveCharacterTextSplitter` | `rag.py` |
| Embed + index + cache | `FAISS.from_documents` / `save_local` | `rag.py` |
| Retrieve relevant chunks | `retriever.invoke(question)` | `app.py` §7c |
| Format chunks for the prompt | `format_context` / `source_list` | `rag.py` |
| Build prompt + memory + LLM | `build_chat` | `app.py` §3 |
| Stream the answer | `chat.stream` + `st.write_stream` | `app.py` §7d |

## Quick start (one command)

```bash
cd memory_chatbot
./run.sh
```

`run.sh` creates a `.venv`, installs `requirements.txt`, and launches the app at
http://localhost:8501. Press `Ctrl+C` to stop it.

> **First run is slower:** it installs `sentence-transformers`/`faiss`, downloads
> the ~90 MB embedding model, and builds the index. Later runs load the cached
> index and start in seconds.

## Manual setup (step by step)

If you'd rather run each step yourself:

```bash
# 1. Go into the app folder
cd memory_chatbot

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate it
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows (PowerShell)

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Launch the app
streamlit run app.py
```

When you're done:

```bash
deactivate                          # exit the virtualenv
```

## API key
This folder ships with its own `.env` holding the `GROQ_API_KEY`, so anyone can
just run it. To use a different key, edit `.env` (or copy the template):

```bash
cp .env.example .env    # then paste your gsk_... key into .env
```

`.env` is gitignored, so the key is never committed.

## Try it
1. "What machine learning approaches are discussed for cancer detection?"
   → grounded answer with a **📎 Sources** expander.
2. "Summarize that in one sentence." → uses conversation memory.

Click **＋ New consult** for a fresh conversation, switch between sessions in the
sidebar, or 🗑️ to delete one. Each session's first message becomes its title.
