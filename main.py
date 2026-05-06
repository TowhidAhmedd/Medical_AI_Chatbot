"""
main.py – FastAPI server for the Medical Chatbot.

Endpoints:
  POST /api/chat          – Send a message
  POST /api/upload        – Upload medical PDFs/TXTs to the knowledge base
  DELETE /api/session     – Clear chat history for a session
  GET  /api/health        – Health check
  GET  /                  – Serve the frontend
"""

import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ── Import RAG (initialised lazily to keep startup fast) ─────
from backend.medical_rag import MedicalRAG, DATA_DIR

app = FastAPI(
    title="MediAssist – Medical Chatbot API",
    description="AI-powered medical information chatbot using LangChain, Groq, RAG & Tavily",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static files
FRONTEND_DIR = Path(__file__).parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# ── Lazy singleton ────────────────────────────────────────────
# _rag: MedicalRAG | None = None

# def get_rag() -> MedicalRAG:
#     global _rag
#     if _rag is None:
#         _rag = MedicalRAG()
#     return _rag

_rag: MedicalRAG | None = None

@app.on_event("startup")
async def startup_event():
    global _rag
    _rag = MedicalRAG()

def get_rag() -> MedicalRAG:
    return _rag

# ── Request / Response models ─────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="default")

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    is_emergency: bool
    web_search_used: bool
    session_id: str

class ClearRequest(BaseModel):
    session_id: str = "default"


# ── Routes ────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return HTMLResponse(content=index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>MediAssist API running. Open /docs for API reference.</h1>")


@app.get("/api/health")
async def health():
    groq_key   = bool(os.getenv("GROQ_API_KEY"))
    tavily_key = bool(os.getenv("TAVILY_API_KEY"))
    return {
        "status": "healthy",
        "groq_configured": groq_key,
        "tavily_configured": tavily_key,
        "rag_loaded": _rag is not None,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        rag    = get_rag()
        result = rag.chat(req.message, req.session_id)
        return ChatResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    allowed = {".pdf", ".txt"}
    suffix  = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Only PDF and TXT files are supported. Got: {suffix}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex}{suffix}"
    dest      = DATA_DIR / safe_name

    try:
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)

        rag    = get_rag()
        result = rag.add_documents([str(dest)])
        return JSONResponse({
            "status": "success",
            "filename": file.filename,
            "chunks_added": result.get("chunks_added", 0),
            "message": "Document ingested into knowledge base.",
        })
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/session")
async def clear_session(req: ClearRequest):
    rag = get_rag()
    rag.clear_session(req.session_id)
    return {"status": "cleared", "session_id": req.session_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", 8000)),
        reload=os.getenv("ENVIRONMENT") == "development",
    )
