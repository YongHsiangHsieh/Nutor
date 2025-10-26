#!/usr/bin/env python3
"""
RAG.py
--------------
Ingest a (very large) PDF into a local vector index using LangChain:
- Load pages lazily
- Chunk with RecursiveCharacterTextSplitter
- Embed in batches (OpenAI or HuggingFace)
- Persist to Chroma (only)
- De-duplicate chunks via stable content hash

Usage:
  python RAG.py --pdf data/big.pdf --index ./chroma-index
"""

import argparse
import hashlib
import pathlib
import sys
from typing import Iterable, List, Tuple
from dotenv import load_dotenv
load_dotenv()

# --- LangChain core bits
import re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Vector stores
from langchain_chroma import Chroma

# Quality of life
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

# --- OCR / layout cleanup helpers (run BEFORE chunking/embedding) ---
_LIGATURES = {
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬀ": "ff",
    "ﬅ": "ft",
    "ﬆ": "st",
}

def normalize_ligatures(text: str) -> str:
    for bad, good in _LIGATURES.items():
        text = text.replace(bad, good)
    return text

def clean_ocr_artifacts(raw: str) -> str:
    """
    Heuristic cleanup for common PDF/OCR issues:
    - de-hyphenate across line breaks: 'W ater-\nSoluble' -> 'WaterSoluble'
    - fix intra-word random spaces: 'o f' -> 'of', 'W ater' -> 'Water'
    - collapse excessive whitespace
    - normalize ligatures (ﬁ -> fi, etc.)
    """
    if not raw:
        return raw

    text = normalize_ligatures(raw)

    # 1) join hyphenated line-breaks: word-\nword -> wordword
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # 2) convert hard newlines to spaces to reduce layout breaks
    #    (keep paragraph intent: double newlines -> single newline)
    text = text.replace("\r", "")
    text = re.sub(r"\n{2,}", "\n", text)   # squeeze multiple newlines
    text = text.replace("\n", " ")

    # 3) remove intra-word single-letter spacing (common OCR artifact)
    #    e.g., 'o f' -> 'of', 'W ater' -> 'Water'
    text = re.sub(r"(?<=\b\w)\s+(?=\w\b)", "", text)

    # 4) collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    # 5) minor known patterns (safe)
    text = text.replace(" o f ", " of ")

    return text.strip()


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()


def chunk_documents(
    pdf_path: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """
    Loads a PDF lazily and returns chunked Documents with useful metadata.
    """
    loader = PyMuPDFLoader(pdf_path)  # layout-aware, robust text extraction
    pages = loader.load()

    # Clean OCR/layout artifacts on each page BEFORE splitting
    cleaned_pages: List[Document] = []
    for d in pages:
        cleaned_pages.append(
            Document(
                page_content=clean_ocr_artifacts(d.page_content),
                metadata=d.metadata,
            )
        )
    pages = cleaned_pages

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(pages)

    # enrich metadata & create stable ids for deduping
    for i, d in enumerate(chunks):
        # page may exist in metadata from PyPDFLoader; normalize
        page = d.metadata.get("page", d.metadata.get("page_number"))
        source = d.metadata.get("source", pdf_path)
        stable_id = sha1(f"{source}|{page}|{len(d.page_content)}|{d.page_content[:64]}")
        d.metadata.update(
            {
                "source": source,
                "page": page,
                "chunk_index": i,
                "chunk_id": stable_id,
            }
        )
    return chunks


def batched(iterable: List[Document], n: int) -> Iterable[List[Document]]:
    """Yield successive n-sized batches from a list."""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def build_embeddings(backend: str, hf_model: str) -> object:
    if backend == "openai":
        # Requires OPENAI_API_KEY in env
        return OpenAIEmbeddings()
    elif backend == "hf":
        # Downloaded locally; good default model can be "sentence-transformers/all-MiniLM-L6-v2"
        return HuggingFaceEmbeddings(model_name=hf_model, show_progress=True)
    else:
        raise ValueError("Unknown embeddings backend. Use 'openai' or 'hf'.")


def build_or_load_vectorstore(
    index_dir: pathlib.Path,
    embeddings,
    collection_name: str,
):
    """
    Create or open a persisted Chroma collection at the given directory.
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    vs = Chroma(
        collection_name=collection_name,
        persist_directory=str(index_dir),
        embedding_function=embeddings,
    )
    return vs


def already_indexed_chunk_ids(vs) -> set:
    """
    Recover existing chunk_ids from Chroma to avoid duplicates on re-ingest.
    """
    ids = set()
    try:
        all_docs = vs.get(include=["metadatas"])
        for meta in all_docs.get("metadatas", []):
            cid = (meta or {}).get("chunk_id")
            if cid:
                ids.add(cid)
    except Exception:
        pass
    return ids


@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(5))
def add_batch(vs, batch_docs: List[Document]):
    """
    Add a batch of Documents to the Chroma vector store with retries.
    """
    vs.add_documents(batch_docs)
    return vs


def main():
    parser = argparse.ArgumentParser(description="Chunk, embed, and store a large PDF into a local vector DB.")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--index", default="./chroma-index", help="Folder to persist index (Chroma)")
    parser.add_argument("--chunk-size", type=int, default=1200, help="Characters per chunk")
    parser.add_argument("--chunk-overlap", type=int, default=180, help="Overlap between chunks")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size (docs per add)")
    parser.add_argument("--emb", choices=["openai", "hf"], default="openai", help="Embedding backend")
    parser.add_argument("--hf-model", default="sentence-transformers/all-MiniLM-L6-v2", help="HF model name if --emb hf")
    parser.add_argument("--collection", default="", help="Chroma collection name (defaults to PDF stem)")
    args = parser.parse_args()

    pdf_path = pathlib.Path(args.pdf)
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    collection_name = args.collection or pdf_path.stem
    index_dir = pathlib.Path(args.index)
    embeddings = build_embeddings(args.emb, args.hf_model)

    print(f"📄 Loading & chunking: {pdf_path} (this may take a while for 1000+ pages)…")
    chunks = chunk_documents(str(pdf_path), args.chunk_size, args.chunk_overlap)
    print(f"✅ Created {len(chunks):,} chunks")

    print(f"📦 Preparing Chroma vector store @ {index_dir} (collection='{collection_name}')")
    vs = build_or_load_vectorstore(index_dir, embeddings, collection_name)

    # Dedupe against existing chunks (best-effort)
    existing_ids = already_indexed_chunk_ids(vs) if vs is not None else set()
    if existing_ids:
        before = len(chunks)
        chunks = [d for d in chunks if d.metadata.get("chunk_id") not in existing_ids]
        removed = before - len(chunks)
        if removed > 0:
            print(f"🧹 Skipped {removed:,} duplicate chunks (already in index)")

    # Incremental batching for memory safety
    total_added = 0
    for batch in tqdm(list(batched(chunks, args.batch_size)), desc="Embedding & adding", unit="batch"):
        vs = add_batch(vs, batch)
        total_added += len(batch)

    print(f"🎉 Done. Added {total_added:,} new chunks to Chroma at: {index_dir} (collection='{collection_name}')")


if __name__ == "__main__":
    main()