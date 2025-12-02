from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel

from rag_basic.query import retrieve_chunks, call_local_llm, COLLECTION_NAME, LOCAL_LLM_MODEL


app = FastAPI(
    title="NSW Working At Heights Safety RAG",
    description="RAG backed safety assistant for NSW working at heights guidance.",
    version="0.1.0",
)
origins = [
    "http://localhost:4200",
    "http://127.0.0.1:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # or ["*"] in dev if you want
    allow_credentials=True,
    allow_methods=["*"],            # allow GET, POST, OPTIONS, etc
    allow_headers=["*"],
)

class ChunkMetadata(BaseModel):
    doc_id: str
    page_num: int
    chunk_idx: int


class RetrievedChunk(BaseModel):
    id: str
    text_preview: str
    metadata: ChunkMetadata


class AskRequest(BaseModel):
    question: str
    top_k: int = 4


class AskResponse(BaseModel):
    answer: str
    chunks: List[RetrievedChunk]


@app.get("/health")
def health() -> Dict[str, Any]:
    """
    Simple health check so you and any frontend can verify
    that the backend is alive.
    """
    return {
        "status": "ok",
        "collection": COLLECTION_NAME,
        "llm_model": LOCAL_LLM_MODEL,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """
    Main RAG endpoint.
    1. Retrieve top_k chunks for the question.
    2. Call local LLM with those chunks.
    3. Return answer plus chunk previews and metadata.
    """
    chunks = retrieve_chunks(req.question, top_k=req.top_k)

    # Prepare chunk previews for frontend
    retrieved_for_response: List[RetrievedChunk] = []
    for cid, doc, meta in chunks:
        preview = doc[:400].replace("\n", " ")
        retrieved_for_response.append(
            RetrievedChunk(
                id=cid,
                text_preview=preview,
                metadata=ChunkMetadata(
                    doc_id=meta["doc_id"],
                    page_num=meta["page_num"],
                    chunk_idx=meta["chunk_idx"],
                ),
            )
        )

    answer = call_local_llm(req.question, chunks)

    return AskResponse(
        answer=answer,
        chunks=retrieved_for_response,
    )
