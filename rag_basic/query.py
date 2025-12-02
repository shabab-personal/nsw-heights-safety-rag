import os
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "working_at_heights_nsw"

EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_chroma_collection():
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(allow_reset=False)
    )
    return client.get_collection(COLLECTION_NAME)


def retrieve_chunks(question: str, top_k: int = 4):
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    q_vec = embed_model.encode([question])[0]

    collection = get_chroma_collection()
    res = collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=top_k
    )

    docs = res["documents"][0]
    metadatas = res["metadatas"][0]
    ids = res["ids"][0]

    return list(zip(ids, docs, metadatas))


def print_retrieved(chunks):
    print("\nRetrieved chunks:")
    for cid, doc, meta in chunks:
        print("=" * 80)
        print(f"ID: {cid}")
        print(f"Doc: {meta['doc_id']} page {meta['page_num']} chunk {meta['chunk_idx']}")
        preview = doc[:400].replace("\n", " ")
        print(f"Text preview: {preview}...")
    print("=" * 80)


def call_llm(question: str, chunks: list[tuple]) -> str:
    """
    Optional: use OpenAI to generate an answer using retrieved context.
    Requires OPENAI_API_KEY in your .env
    """
    if not OPENAI_API_KEY:
        return "No OPENAI_API_KEY set. Skipping LLM call."

    client = OpenAI(api_key=OPENAI_API_KEY)

    context_parts = []
    for _, doc, meta in chunks:
        context_parts.append(
            f"[{meta['doc_id']} p{meta['page_num']} c{meta['chunk_idx']}]\n{doc}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""
You are a construction safety assistant for worksites in New South Wales, Australia.

Use only the information in the context below, which comes from official safety and WHS documents about working at heights.

If the answer cannot be found in the context, say you do not know and suggest the user consult the official SafeWork NSW material.

Context:
{context}

Question: {question}

Answer in clear, practical language. Where relevant, mention fall prevention hierarchy and PPE requirements. Cite the chunks you used at the end as [doc_id pX].
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output[0].content[0].text


def main():
    while True:
        q = input("\nAsk a question about working at heights (or 'quit'): ").strip()
        if not q or q.lower() == "quit":
            break

        chunks = retrieve_chunks(q, top_k=4)
        print_retrieved(chunks)

        answer = call_llm(q, chunks)
        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()
