
# NSW Working at Heights Safety Assistant  
A fully local, high‑accuracy Retrieval‑Augmented Generation (RAG) system designed to answer questions about **Working at Heights safety regulations in NSW, Australia**, built with production‑style architecture: ingestion, embeddings, vector search, local LLM inference, FastAPI backend, and Angular frontend.

This project demonstrates real machine learning engineering: RAG systems, embeddings, vector databases, LLM orchestration, API design, and UI integration.

---

## ⭐ Overview

The NSW Working at Heights Safety Assistant allows users to ask complex safety questions and receive clear, accurate answers sourced from real NSW government documents.

Everything runs **fully offline** using:
- Local embeddings  
- Local vector database (ChromaDB)  
- Local LLM via Ollama  
- Local backend (FastAPI)  
- Local frontend (Angular)

No cloud services, no API fees, no privacy risk.

---

## 🚀 Key Features

### 🔹 High‑Accuracy Retrieval  
- Uses document chunking + embedding similarity search  
- Powered by ChromaDB  
- Supports adding/removing documents dynamically  

### 🔹 Local LLM Reasoning  
- Default model: **llama3.1** via Ollama  
- Can swap to any offline model  
- No token limits or API cost  

### 🔹 Full API Backend  
- FastAPI async server  
- `/ask` endpoint that handles retrieval and generation  
- Clean, modular code  

### 🔹 Angular Frontend  
- Modern standalone Angular app  
- Real chat interface  
- Smooth typing animation  
- Clean, minimalist UI  

### 🔹 Real‑World Use Case  
- Construction industry safety compliance is confusing  
- The assistant simplifies regulatory text into usable guidance  
- Perfect for workers, supervisors, and training environments  

---

## 🧠 Architecture

```
PDFs → ingest.py → ChromaDB → query.py → FastAPI → Angular UI → User
                     ↑                                     ↓
                 Embeddings                           Local LLM (Ollama)
```

- `ingest.py` converts PDF documents into chunks → embeddings → vector DB  
- `query.py` retrieves relevant chunks and generates final answers  
- `FastAPI` exposes a REST endpoint for the UI  
- `Angular` provides a user-friendly chat interface  

---

## 📁 Folder Structure

```
project/
│── data/pdf/                 # Source PDFs (NSW safety documents)
│── embeddings/               # Local ChromaDB vector store
│── rag_basic/
│     ├── ingest.py           # Document loading + chunking + embeddings
│     ├── query.py            # RAG retrieval + generation pipeline
│     └── __init__.py
│── api/
│     └── main.py             # FastAPI backend with /ask endpoint
│── safety-rag-ui/            # Angular frontend (chat UI)
│── README.md
```

---

## 📌 Setup Instructions

### 1. Install dependencies

#### Backend
```
pip install fastapi uvicorn chromadb sentence-transformers pypdf python-dotenv
```

#### Frontend
```
npm install
```

#### Ollama  
Download from: https://ollama.com

Then pull your model:
```
ollama pull llama3.1
```

---

## 🔧 Step 1: Ingest Documents

Place NSW safety PDFs into:

```
data/pdf/
```

Run:
```
python -m rag_basic.ingest
```

This:
- Loads PDFs  
- Splits into chunks  
- Generates embeddings  
- Stores vectors in ChromaDB  

---

## 🔍 Step 2: Test Retrieval + LLM Locally

```
python -m rag_basic.query
```

Example:
```
What is considered working at heights in NSW?
```

You will see:
- Retrieved chunks  
- Final answer  

---

## 🌐 Step 3: Run FastAPI Backend

```
uvicorn api.main:app --reload
```

Endpoint:
```
POST http://127.0.0.1:8000/ask
```

---

## 💬 Step 4: Run Angular Frontend

```
cd safety-rag-ui
ng serve
```

Visit:
```
http://localhost:4200
```

You now have a **fully working chat application**.

---

## 🏗️ Technologies Used

### Machine Learning  
- Python  
- ChromaDB  
- SentenceTransformers  
- Ollama (local LLM)  
- RAG design (retrieve → rank → generate)  

### Backend  
- FastAPI  
- Pydantic  
- CORS middleware  
- Async REST endpoints  

### Frontend  
- Angular standalone components  
- Fetch-based communication  
- Modern chat interface  
- Responsive styling  






## 🛠️ Future Enhancements

- Add a second safety domain (e.g., confined spaces, electrical hazards)
- Switch to hierarchical retrieval  
- Add rerankers (e.g., bge-reranker)  
- Multi-turn conversation with memory  
- Deployment to Docker  
- Optional cloud mode with OpenAI fallback  

---

## 👤 Author

**Shabab Saleheen**  
Machine Learning Engineer • Full-Stack Developer  
Sydney, Australia  

---

