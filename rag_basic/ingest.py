import os
from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


# 1. Paths and constants
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "working_at_heights_nsw"

EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def load_pdfs(pdf_dir: Path) -> list[dict]:
    """
    Load all PDFs and return a list of dicts:
    { 'doc_id': str, 'page_num': int, 'text': str }
    """
    docs = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue
            docs.append({
                "doc_id": pdf_path.stem,
                "page_num": page_idx + 1,
                "text": text
            })
    return docs


def simple_chunk(text: str, max_chars: int = 800) -> list[str]:
    """
    Very simple chunking by character count.
    For MVP this is enough. You can improve later.
    """
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        # try to cut on last period if possible
        last_dot = chunk.rfind(".")
        if last_dot != -1 and last_dot > max_chars * 0.6:
            end = start + last_dot + 1
            chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end
    return chunks


def build_chunks(pages: list[dict]) -> tuple[list[str], list[dict]]:
    """
    From page dicts to:
    - list of text chunks
    - list of metadata dicts aligned with chunks
    """
    texts = []
    metadatas = []
    for p in pages:
        page_chunks = simple_chunk(p["text"], max_chars=800)
        for idx, ch in enumerate(page_chunks):
            texts.append(ch)
            metadatas.append({
                "doc_id": p["doc_id"],
                "page_num": p["page_num"],
                "chunk_idx": idx
            })
    return texts, metadatas


def get_chroma_client() -> chromadb.Client:
    CHROMA_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(allow_reset=True)
    )
    return client


def main():
    print(f"Loading PDFs from: {PDF_DIR}")
    pages = load_pdfs(PDF_DIR)
    print(f"Loaded {len(pages)} pages of text")

    print("Chunking pages...")
    texts, metadatas = build_chunks(pages)
    print(f"Created {len(texts)} chunks")

    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Computing embeddings...")
    embeddings = embed_model.encode(texts, batch_size=16, show_progress_bar=True)

    print("Saving to ChromaDB...")
    client = get_chroma_client()

    # drop existing collection for rebuild
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    ids = [f"chunk_{i}" for i in range(len(texts))]

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        documents=texts
    )

    print("Ingestion complete.")
    print(f"Collection name: {COLLECTION_NAME}")
    print(f"Chroma directory: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
