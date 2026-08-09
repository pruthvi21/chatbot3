"""
rag.py — the RETRIEVAL layer for the Cancer Specialist app.
================================================================

WHAT IS RAG?
------------
RAG = "Retrieval-Augmented Generation". The LLM (Groq's Llama) has never read
our cancer papers. So instead of hoping it "knows" the answer, we:

    1. INGEST the papers once  -> turn them into searchable "embeddings"
    2. On each question, RETRIEVE the few most relevant paragraphs
    3. Paste those paragraphs into the prompt so the LLM answers FROM them

This file does steps 1 and 2. The chat/prompt/LLM part lives in app.py.

THE INGESTION PIPELINE (built in get_retriever() below):

    PDFs in ./docs
        │  PyPDFLoader           read each PDF, one Document per page
        ▼
    pages (text + metadata)
        │  RecursiveCharacter…   cut pages into ~1000-char overlapping chunks
        ▼
    chunks
        │  HuggingFaceEmbeddings turn each chunk into a vector (list of numbers)
        ▼
    vectors
        │  FAISS                 store vectors in a searchable index
        ▼
    .faiss_index/  (saved to disk so we don't redo this every boot)

Later, retriever.invoke("some question") embeds the QUESTION the same way and
returns the chunks whose vectors are closest — i.e. the most semantically
similar passages.
"""

import os
import hashlib
import re
from pathlib import Path

import streamlit as st

# --- The four RAG building blocks from LangChain --------------------------
from langchain_community.document_loaders import PyPDFLoader          # read PDFs
from langchain_community.vectorstores import FAISS                    # vector DB
from langchain_huggingface import HuggingFaceEmbeddings              # text -> vectors
from langchain_text_splitters import RecursiveCharacterTextSplitter  # page -> chunks

# --- Where things live ----------------------------------------------------
HERE = os.path.dirname(__file__)                # this file's folder
DOCS_DIR = os.path.join(HERE, "docs")           # put your PDFs here
INDEX_DIR = os.path.join(HERE, ".faiss_index")  # cached vector index (auto-made)

# The embedding model that converts text into vectors. all-MiniLM-L6-v2 is a
# small, fast, free model (384 numbers per chunk). It runs locally on your
# machine — no API call needed for embeddings.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _safe_pdf_name(original_name: str) -> str:
    """Return a safe filename for an uploaded PDF.

    Browser uploads include a user-controlled filename.  We keep only the
    filename (not any folder path), replace unusual characters, and force the
    extension to ``.pdf``.  This prevents an upload from writing outside the
    app's ``docs`` folder.
    """
    name = Path(original_name).name
    stem = Path(name).stem
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return f"{safe_stem or 'uploaded_document'}.pdf"


def save_uploaded_pdfs(uploaded_files) -> tuple[list[str], list[str]]:
    """Save Streamlit uploaded PDFs into ``docs/``.

    Returns ``(saved, skipped)`` filename lists.

    - An identical existing file is skipped, avoiding duplicate vectors.
    - If a different file has the same name, ``_2``, ``_3``, etc. is added so
      the original research paper is never overwritten.

    The caller should run ``clear_retriever_cache()`` after at least one file
    is saved.  On Streamlit's next rerun, ``get_retriever()`` sees the changed
    document fingerprint and rebuilds the FAISS index.
    """
    os.makedirs(DOCS_DIR, exist_ok=True)
    saved, skipped = [], []

    for uploaded in uploaded_files:
        data = uploaded.getvalue()
        filename = _safe_pdf_name(uploaded.name)
        destination = Path(DOCS_DIR) / filename

        # Same filename + same bytes means this PDF is already in the library.
        if destination.exists() and destination.read_bytes() == data:
            skipped.append(filename)
            continue

        # Preserve an existing different file by selecting a new filename.
        if destination.exists():
            stem = destination.stem
            number = 2
            while destination.exists():
                destination = Path(DOCS_DIR) / f"{stem}_{number}.pdf"
                number += 1

        destination.write_bytes(data)
        saved.append(destination.name)

    return saved, skipped


def clear_retriever_cache() -> None:
    """Forget the in-memory retriever so changed PDFs are indexed next rerun."""
    get_retriever.clear()


def _docs_signature(docs_dir: str) -> str:
    """Make a short 'fingerprint' of the current PDF set.

    We include each file's name, size and last-modified time. If you add,
    remove or edit a PDF, the fingerprint changes — that's how we know the
    cached index is stale and must be rebuilt.
    """
    parts = []
    for name in sorted(os.listdir(docs_dir)):
        if name.lower().endswith(".pdf"):
            stat = os.stat(os.path.join(docs_dir, name))
            parts.append(f"{name}:{stat.st_size}:{int(stat.st_mtime)}")
    # Hash the combined string down to one short, comparable value.
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


# @st.cache_resource makes Streamlit run this function only ONCE per app run
# and reuse the result on every rerun/refresh. Without it we'd re-index the
# PDFs every time the user clicks anything — very slow.
@st.cache_resource(show_spinner=False)
def get_retriever(k: int = 4):
    """Build (or load) the FAISS retriever over the PDFs in ./docs.

    Returns a tuple: (retriever, num_pdfs, num_chunks).
      - retriever : object with .invoke("question") -> list of relevant chunks
                    (None if there are no PDFs to index)
      - num_pdfs  : how many PDFs were indexed  (for the sidebar display)
      - num_chunks: how many text chunks are searchable
    `k` is how many chunks to return per question.
    """
    os.makedirs(DOCS_DIR, exist_ok=True)

    # Find every PDF in the docs folder.
    pdfs = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf")]
    if not pdfs:
        return None, 0, 0  # nothing to index

    # Load the local embedding model (downloads ~90 MB the very first time).
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Current fingerprint of the docs, and where we stored the last one.
    signature = _docs_signature(DOCS_DIR)
    sig_path = os.path.join(INDEX_DIR, "signature.txt")

    # ---- FAST PATH: reuse the saved index if the docs haven't changed -----
    if os.path.exists(sig_path):
        with open(sig_path) as f:
            if f.read().strip() == signature:
                # Load the previously-built FAISS index straight from disk.
                # (allow_dangerous_deserialization is safe here: WE created it.)
                vs = FAISS.load_local(
                    INDEX_DIR, embeddings, allow_dangerous_deserialization=True
                )
                retriever = vs.as_retriever(search_kwargs={"k": k})
                return retriever, len(pdfs), vs.index.ntotal

    # ---- SLOW PATH: (re)build the index from scratch ----------------------

    # STEP 1 — LOAD: read every PDF into Document objects (one per page).
    # Each Document carries page_content (text) + metadata (source file, page#).
    all_docs = []
    for name in sorted(pdfs):
        all_docs.extend(PyPDFLoader(os.path.join(DOCS_DIR, name)).load())

    # STEP 2 — SPLIT: a whole page is too big to embed usefully, so we cut it
    # into ~1000-character chunks. chunk_overlap=150 repeats a little text
    # between neighbours so a sentence split across a boundary isn't lost.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(all_docs)

    # STEP 3 — EMBED + STORE: FAISS.from_documents() embeds every chunk and
    # builds the searchable vector index in one call.
    vs = FAISS.from_documents(chunks, embeddings)

    # STEP 4 — CACHE: save the index + the fingerprint so next boot is instant.
    vs.save_local(INDEX_DIR)
    with open(sig_path, "w") as f:
        f.write(signature)

    retriever = vs.as_retriever(search_kwargs={"k": k})
    return retriever, len(pdfs), len(chunks)


def format_context(docs) -> str:
    """Turn the retrieved chunks into one numbered, source-tagged text block.

    This block is what we paste into the prompt as {context}. Tagging each
    chunk with its source file + page helps the model (and us) trace answers.
    """
    blocks = []
    for i, d in enumerate(docs, 1):
        src = os.path.basename(d.metadata.get("source", "unknown"))
        page = d.metadata.get("page")
        tag = f"[{i}] {src}"
        if isinstance(page, int):
            tag += f" (p.{page + 1})"  # PDF pages are 0-indexed internally
        blocks.append(f"{tag}\n{d.page_content}")
    return "\n\n".join(blocks)


def source_list(docs):
    """Collect the unique 'file · p.N' labels to show under an answer."""
    seen = []
    for d in docs:
        src = os.path.basename(d.metadata.get("source", "unknown"))
        page = d.metadata.get("page")
        label = f"{src}" + (f" · p.{page + 1}" if isinstance(page, int) else "")
        if label not in seen:
            seen.append(label)
    return seen
