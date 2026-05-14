# 🩺 MediAssist — AI Medical Chatbot

> A production-grade medical chatbot with RAG-powered context, real-time web search, conversation memory, and emergency detection — deployed on Render.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Environment Variables](#environment-variables)
- [Local Development](#local-development)
- [Deploying to Render](#deploying-to-render)
  - [Prerequisites](#prerequisites)
  - [1. Deploy the Backend](#1-deploy-the-backend)
  - [2. render.yaml — Infrastructure as Code](#2-renderyaml--infrastructure-as-code)
  - [Deployment Notes & Gotchas](#deployment-notes--gotchas)
- [Adding Medical Documents](#adding-medical-documents)
- [API Reference](#api-reference)
- [Built-in Knowledge Base](#built-in-knowledge-base)
- [Medical Disclaimer](#medical-disclaimer)

---

## Features

- **Dual RAG Sources** — Local ChromaDB (PDF/TXT ingestion) + Tavily live web search across trusted medical domains
- **Conversation Memory** — LangChain `ConversationBufferWindowMemory` retains the last 8 turns for contextual replies
- **Emergency Detection** — Flags critical symptoms and escalates with emergency guidance automatically
- **Source Attribution** — Every response cites its sources (local RAG, web domains, or both)
- **Document Upload** — Add your own PDFs/TXTs via the UI, the `data/` folder, or the REST API
- **MMR Retrieval** — Maximal Marginal Relevance ensures diverse, non-redundant context chunks

---

## Architecture

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

1. **Local ChromaDB** — Seeded with 10 medical topics + any PDFs/TXTs you add
2. **Tavily Web Search** — Restricted to trusted medical domains:
   `mayoclinic.org` · `webmd.com` · `nih.gov` · `medlineplus.gov` · `healthline.com` · `who.int` · `cdc.gov` · `nhs.uk` · `pubmed.ncbi.nlm.nih.gov`

---

## Project Structure

```
medical-chatbot/
├── main.py                   # FastAPI server (entry point)
├── requirements.txt          # Python dependencies
├── setup.sh                  # One-shot setup & run script
├── render.yaml               # Render deployment config (IaC)
├── .env.example              # Environment variables template
├── .env                      # Your actual keys (gitignored)
│
├── backend/
│   ├── __init__.py
│   └── medical_rag.py        # Core RAG pipeline
│
├── frontend/
│   └── index.html            # Full chat UI (dark medical theme)
│
├── data/                     # Drop PDFs/TXTs here for ingestion
└── vectorstore/
    └── chroma_db/            # Persisted ChromaDB (auto-created)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Groq `llama-3.3-70b-versatile` |
| **RAG Framework** | LangChain v0.3 |
| **Vector Store** | ChromaDB (local, persistent) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` (runs locally) |
| **Web Search** | Tavily API (medical domains only) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Vanilla HTML/CSS/JS (dark medical theme) |
| **Memory** | LangChain `ConversationBufferWindowMemory` |
| **Hosting** | Render |

---

## Environment Variables

Create a `.env` file from the template and fill in your keys:

```bash
cp .env.example .env
```

```env
GROQ_API_KEY=gsk_your_groq_key_here        # https://console.groq.com/keys
TAVILY_API_KEY=tvly-your_tavily_key_here   # https://app.tavily.com/home
```

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | Powers all LLM inference via Groq's LPU (free tier available) |
| `TAVILY_API_KEY` | ✅ Yes | Powers web RAG via trusted medical domain search |

> **Security**: Never commit `.env` to version control. It is listed in `.gitignore` by default.

---

## Local Development

### Quick Start (recommended)

```bash
chmod +x setup.sh
./setup.sh
```

Then open **http://localhost:8000** in your browser.

### Manual Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
nano .env                      # Add your API keys

# Start the server
python main.py
```

Server: **http://localhost:8000**  
API docs: **http://localhost:8000/docs**

---

## Deploying to Render

MediAssist deploys as a single **Web Service** on Render. The FastAPI backend serves both the API and the static frontend from the same process — no separate static site needed.

### Prerequisites

- A [Render account](https://render.com) (free tier works)
- Your project pushed to a GitHub or GitLab repository
- `GROQ_API_KEY` and `TAVILY_API_KEY` ready

---

### 1. Deploy the Backend

1. Go to [Render Dashboard](https://dashboard.render.com) → **New** → **Web Service**
2. Connect your GitHub repository
3. Configure the service:

| Setting | Value |
|---|---|
| **Name** | `mediassist` |
| **Region** | Choose closest to your users |
| **Branch** | `main` |
| **Root Directory** | *(leave blank — repo root)* |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| **Instance Type** | Free (or Starter for production) |

4. Under **Environment Variables**, add:

```
GROQ_API_KEY     = gsk_your_groq_key_here
TAVILY_API_KEY   = tvly-your_tavily_key_here
```

5. Click **Create Web Service**

Your app will be live at:
```
https://mediassist.onrender.com
```

> **Cold Starts on Free Tier**: Render's free tier spins down services after 15 minutes of inactivity. The first request after sleep may take 30–60 seconds. Upgrade to a **Starter** instance ($7/mo) to eliminate cold starts in production.

---

### 2. `render.yaml` — Infrastructure as Code

For repeatable, one-click deployments, add a `render.yaml` at the repo root. Render auto-detects it and provisions the service:

```yaml
# render.yaml  (place at repo root)

services:
  - type: web
    name: mediassist
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: GROQ_API_KEY
        sync: false          # Render will prompt for this secret on first deploy
      - key: TAVILY_API_KEY
        sync: false
    healthCheckPath: /api/health
```

To deploy via Blueprint:

1. Push `render.yaml` to your repository
2. Go to **Render Dashboard → New → Blueprint**
3. Connect your repository — Render provisions everything automatically

---

### Deployment Notes & Gotchas

| Topic | Notes |
|---|---|
| **Ephemeral Filesystem** | Render's free instances have an ephemeral disk. The ChromaDB vector store (`vectorstore/chroma_db/`) and any uploaded files in `data/` are **wiped on every redeploy or restart**. For persistence, attach a **Render Disk** (paid) or migrate to a hosted vector DB like [Pinecone](https://www.pinecone.io) or [Qdrant Cloud](https://cloud.qdrant.io). |
| **HuggingFace Model Download** | `all-MiniLM-L6-v2` is downloaded from HuggingFace on first startup (~90MB). This adds ~60s to cold start. The model is cached in the build layer on subsequent deploys. |
| **Free Tier Spin-Down** | Services on the free tier sleep after 15 min of inactivity. Wake-up on the first request takes 30–60s. Use an uptime monitor (e.g. [UptimeRobot](https://uptimerobot.com)) to ping `/api/health` every 10 minutes and keep it warm. |
| **Build Timeout** | `sentence-transformers` and `chromadb` are large packages. If builds time out, add `--no-cache-dir` to pip: `pip install --no-cache-dir -r requirements.txt`. |
| **Port Binding** | Always use `--port $PORT` in the start command. Hardcoding port 8000 will cause the service to fail — Render assigns the port dynamically via the `$PORT` environment variable. |
| **Secrets Management** | Set `GROQ_API_KEY` and `TAVILY_API_KEY` in Render's **Environment** panel — never in `render.yaml` directly. `sync: false` means Render will prompt you to enter the value securely on first deploy. |
| **Custom Domain** | Custom domains with auto-provisioned TLS are supported on all plans. Configure under **Settings → Custom Domains** on your service. |

---

## Adding Medical Documents

**Via the UI**: Click the upload zone in the sidebar → select a PDF or TXT file.

**Via the `data/` folder**: Drop files into `data/` before starting the server — they are auto-ingested on first run.

**Via the API**:
```bash
curl -X POST https://mediassist.onrender.com/api/upload \
     -F "file=@medical_textbook.pdf"
```

> **Reminder on Render**: Uploaded files are stored on the ephemeral filesystem and will not survive a redeploy. Attach a Render Disk or use cloud storage (e.g. S3) for permanent document persistence.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat` | Send a chat message |
| `POST` | `/api/upload` | Upload a PDF or TXT document |
| `DELETE` | `/api/session` | Clear conversation history |
| `GET` | `/api/health` | Check API key status |
| `GET` | `/docs` | Interactive API docs (Swagger UI) |

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
  "answer": "Hypertension often has no symptoms...",
  "sources": ["MedicalRAG_Seed", "mayoclinic.org"],
  "is_emergency": false,
  "web_search_used": true,
  "session_id": "user-123"
}
```

---

## Built-in Knowledge Base

The chatbot ships with seed knowledge covering:

- Common Cold
- Hypertension (High Blood Pressure)
- Type 2 Diabetes
- Chest Pain & Cardiac Emergencies
- Migraines
- Asthma
- Depression & Mental Health
- Adult Vaccination Schedule
- Drug Interactions
- First Aid — Burns

---

## Medical Disclaimer

This chatbot provides **medical information only**, not medical advice. It does not diagnose conditions, prescribe treatments, or replace professional medical consultation. Always consult a qualified healthcare provider for personal health decisions.

---

## License

MIT — use freely, modify, and build upon.
