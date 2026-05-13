# 🩺 MediAssist – AI Medical Chatbot

A **complete end-to-end medical chatbot** built with:

| Component | Technology |
|-----------|-----------|
| **LLM**   | Groq (`llama-3.3-70b-versatile`) – free tier |
| **RAG Framework** | LangChain v0.3 |
| **Vector Store** | ChromaDB (local, persistent) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` (runs locally, free) |
| **Web Search** | Tavily API (medical domains) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Vanilla HTML/CSS/JS (dark medical theme) |
| **Memory** | LangChain `ConversationBufferWindowMemory` |

---

## Project Structure

```
medical-chatbot/
├── main.py                   # FastAPI server (entry point)
├── requirements.txt          # Python dependencies
├── setup.sh                  # One-shot setup & run script
├── .env.example              # Environment variables template
├── .env                      # Your actual keys (gitignored!)
│
├── backend/
│   ├── __init__.py
│   └── medical_rag.py        # Core RAG pipeline
│
├── frontend/
│   └── index.html            # Full chat UI
│
├── data/                     # Drop PDFs/TXTs here for ingestion
└── vectorstore/
    └── chroma_db/            # Persisted ChromaDB (auto-created)
```

---

## Quick Start

### 1. Add your API keys to `.env`

```env
GROQ_API_KEY=gsk_your_groq_key_here
TAVILY_API_KEY=tvly-your_tavily_key_here
```

Get your keys (both free):
- **Groq**: https://console.groq.com/keys
- **Tavily**: https://app.tavily.com/home

### 2. Run the setup script

```bash
chmod +x setup.sh
./setup.sh
```

### 3. Open in browser

Navigate to **http://localhost:8000**

---

## Manual Setup (Alternative)

```bash
# Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy & fill environment variables
cp .env.example .env
# Edit .env with your keys

# Run the server
python main.py
```

---

## 🔬 How It Works

```
User Question
      │
      ▼
┌─────────────────────────────────────────────┐
│              LangChain Pipeline              │
│                                             │
│  ┌──────────────────┐  ┌─────────────────┐  │
│  │  ChromaDB (MMR)  │  │  Tavily Search  │  │
│  │  Local RAG       │  │  Web RAG        │  │
│  │  (HF Embeddings) │  │  (Medical URLs) │  │
│  └────────┬─────────┘  └────────┬────────┘  │
│           │                     │           │
│           └──────────┬──────────┘           │
│                      ▼                      │
│              Context Assembly               │
│                      │                      │
│                      ▼                      │
│           Medical System Prompt             │
│                      │                      │
│                      ▼                      │
│         Groq LLM (llama-3.3-70b)           │
│                      │                      │
│                      ▼                      │
│        Conversation Memory (k=8)           │
└─────────────────────────────────────────────┘
      │
      ▼
 Response + Sources + Emergency Flag
```

### RAG Sources (Priority Order)

1. **Local ChromaDB** – Seeded with 10 medical topics + any PDFs/TXTs you add
2. **Tavily Web Search** – Searches trusted medical domains:
   - mayoclinic.org, webmd.com, nih.gov, medlineplus.gov
   - healthline.com, who.int, cdc.gov, nhs.uk, pubmed.ncbi.nlm.nih.gov

---

## Adding Medical Documents

**Via the UI**: Click the upload zone in the sidebar → select a PDF or TXT file.

**Via the data/ folder**: Drop PDF/TXT files into `data/` before starting the server. They'll be auto-ingested on first run.

**Via the API**:
```bash
curl -X POST http://localhost:8000/api/upload \
     -F "file=@medical_textbook.pdf"
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat` | Send a chat message |
| `POST` | `/api/upload` | Upload a PDF/TXT document |
| `DELETE` | `/api/session` | Clear conversation history |
| `GET` | `/api/health` | Check API key status |
| `GET` | `/docs` | Interactive API docs (Swagger) |

### Chat Request

```json
POST /api/chat
{
  "message": "What are the symptoms of hypertension?",
  "session_id": "user-123"
}
```

### Chat Response

```json
{
  "answer": "Hypertension often has no symptoms…",
  "sources": ["MedicalRAG_Seed", "mayoclinic.org"],
  "is_emergency": false,
  "web_search_used": true,
  "session_id": "user-123"
}
```

---

## Built-in Medical Knowledge Base

The chatbot ships with seed knowledge on:

- Common Cold
- Hypertension (High Blood Pressure)
- Type 2 Diabetes
- Chest Pain & Cardiac Emergencies
- Migraines
- Asthma
- Depression & Mental Health
- Adult Vaccination Schedule
- Drug Interactions
- First Aid – Burns

---

## Medical Disclaimer

This chatbot provides **medical information only**, not medical advice. It does not diagnose conditions, prescribe treatments, or replace professional medical consultation. Always consult a qualified healthcare provider for personal health decisions.

---

## License

MIT – use freely, modify, and build upon.
