import os
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import ollama  # local LLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "working_at_heights_nsw"

EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

LOCAL_LLM_MODEL = "llama3.1"  # make sure you have run: ollama pull llama3.1


def get_chroma_collection():
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(allow_reset=False)
    )
    return client.get_collection(COLLECTION_NAME)


def retrieve_chunks(question: str, top_k: int = 4):
    """
    Embed the question and retrieve the top_k most similar chunks
    from the Chroma collection.
    """
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
    """
    Pretty print the retrieved chunks to inspect what the retriever is using.
    Helpful for debugging and sanity checking.
    """
    print("\nRetrieved chunks:")
    for cid, doc, meta in chunks:
        print("=" * 80)
        print(f"ID: {cid}")
        print(f"Doc: {meta['doc_id']} page {meta['page_num']} chunk {meta['chunk_idx']}")
        preview = doc[:400].replace("\n", " ")
        print(f"Text preview: {preview}...")
    print("=" * 80)


def call_local_llm(question: str, chunks: list[tuple]) -> str:
    """
    Use a local LLM via Ollama (llama3.1) to generate an answer,
    grounded in the retrieved context.
    """
    if not chunks:
        return "I could not retrieve any relevant context from the documents."

    # Build context string from retrieved chunks
    context_parts = []
    for _, doc, meta in chunks:
        context_parts.append(
            f"[{meta['doc_id']} p{meta['page_num']} c{meta['chunk_idx']}]\n{doc}"
        )
    context = "\n\n".join(context_parts)

    system_prompt = (
        "You are a construction safety assistant for worksites in New South Wales, Australia. "
        "You answer questions about working at heights using only the provided context, which "
        "comes from official WHS and SafeWork NSW style documents.\n\n"
        "If you cannot find the answer in the context, say that you do not know and suggest "
        "consulting the official SafeWork NSW materials or a competent safety professional.\n\n"
        "Always be conservative and safety focused in your advice. "
        "Where relevant, mention the hierarchy of control for fall prevention and appropriate PPE. "
        "At the end of your answer, cite the chunks you used in square brackets, for example [code_of_practice_falls p12]."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    response = ollama.chat(
        model=LOCAL_LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "temperature": 0.2,
            "num_predict": 512,
        },
    )

    return response["message"]["content"]


def main():
    print("Local RAG assistant for Working at Heights (NSW).")
    print(f"Using Chroma collection: {COLLECTION_NAME}")
    print(f"Using local LLM model via Ollama: {LOCAL_LLM_MODEL}")

    while True:
        q = input("\nAsk a question about working at heights (or 'quit'): ").strip()
        if not q or q.lower() == "quit":
            break

        chunks = retrieve_chunks(q, top_k=4)
        print_retrieved(chunks)

        answer = call_local_llm(q, chunks)
        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()
