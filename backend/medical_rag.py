

"""
medical_rag.py – RAG pipeline for the Medical Chatbot.

Sources (in priority order):
  1. Local ChromaDB vector store (PDF / text knowledge base)
  2. Tavily medical web search (live results)

Embeddings : FastEmbed BAAI/bge-small-en-v1.5  (lightweight, ~50MB, no torch)
LLM        : Groq  (llama-3.3-70b-versatile – free tier)
"""

import os
import logging
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from tavily import TavilyClient

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Paths ──────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "chroma_db"

# ─── Constants ──────────────────────────────────────────────
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"
GROQ_MODEL    = "llama-3.3-70b-versatile"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150
TOP_K_LOCAL   = 4
TOP_K_WEB     = 3

# ─── Medical-tuned system prompt ────────────────────────────
MEDICAL_PROMPT = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template="""You are MediAssist, an expert AI medical information assistant powered by a curated medical knowledge base and live web search.

IMPORTANT GUIDELINES:
- Provide accurate, evidence-based medical information
- Always recommend consulting a qualified healthcare professional for diagnosis and treatment
- Never diagnose conditions or prescribe medications
- Flag any urgent/emergency symptoms immediately (e.g. chest pain, difficulty breathing, stroke symptoms)
- Cite whether information comes from your knowledge base or web search
- Use plain language while being medically precise
- Structure answers clearly with sections when appropriate

CONTEXT FROM KNOWLEDGE BASE & WEB SEARCH:
{context}

CONVERSATION HISTORY:
{chat_history}

PATIENT QUESTION: {question}

MEDIASSIST RESPONSE:""",
)


class MedicalRAG:
    """End-to-end RAG pipeline for medical Q&A."""

    def __init__(self):
        self.groq_api_key   = os.getenv("GROQ_API_KEY", "")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY", "")

        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not set in environment / .env file")

        logger.info("Loading embedding model (FastEmbed) …")
        self.embeddings = FastEmbedEmbeddings(
            model_name=EMBED_MODEL,
        )

        self.vectorstore = self._load_or_create_vectorstore()

        logger.info("Initialising Groq LLM …")
        self.llm = ChatGroq(
            api_key=self.groq_api_key,
            model=GROQ_MODEL,
            temperature=0.2,
            max_tokens=1500,
        )

        if self.tavily_api_key:
            self.tavily = TavilyClient(api_key=self.tavily_api_key)
            logger.info("Tavily web search enabled.")
        else:
            self.tavily = None
            logger.warning("TAVILY_API_KEY not set – web search disabled.")

        # Per-session memory stores  {session_id: ConversationBufferWindowMemory}
        self._memories: dict[str, ConversationBufferWindowMemory] = {}

    # ── Vector store ─────────────────────────────────────────
    def _load_or_create_vectorstore(self) -> Chroma:
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        existing = list(VECTORSTORE_DIR.glob("*.sqlite3")) + \
                   list(VECTORSTORE_DIR.glob("chroma.sqlite3"))

        if existing:
            logger.info("Loading existing ChromaDB vector store …")
            return Chroma(
                persist_directory=str(VECTORSTORE_DIR),
                embedding_function=self.embeddings,
                collection_name="medical_kb",
            )

        logger.info("No existing vector store found – building from data/ …")
        return self._build_vectorstore()

    def _build_vectorstore(self) -> Chroma:
        documents: List[Document] = []

        # ── built-in seed knowledge (always present) ─────────
        seed_docs = self._get_seed_medical_docs()
        documents.extend(seed_docs)

        # ── load user PDFs / TXTs from data/ ─────────────────
        pdf_files = list(DATA_DIR.glob("*.pdf"))
        txt_files = list(DATA_DIR.glob("*.txt"))

        for pdf in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf))
                documents.extend(loader.load())
                logger.info(f"Loaded PDF: {pdf.name}")
            except Exception as e:
                logger.warning(f"Could not load {pdf.name}: {e}")

        for txt in txt_files:
            try:
                loader = TextLoader(str(txt), encoding="utf-8")
                documents.extend(loader.load())
                logger.info(f"Loaded TXT: {txt.name}")
            except Exception as e:
                logger.warning(f"Could not load {txt.name}: {e}")

        # ── chunk & embed ──────────────────────────────────────
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", " "],
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents.")

        vs = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(VECTORSTORE_DIR),
            collection_name="medical_kb",
        )
        logger.info("Vector store built and persisted.")
        return vs

    def _get_seed_medical_docs(self) -> List[Document]:
        """Hardcoded seed medical knowledge so the bot works out of the box."""
        entries = [
            ("Common Cold", "The common cold is a viral infection of the upper respiratory tract. Symptoms include runny nose, sore throat, cough, congestion, mild headache, sneezing, and low-grade fever. Most colds resolve within 7-10 days. Treatment is supportive: rest, hydration, and OTC medications for symptom relief. Antibiotics are NOT effective against viral infections. Seek medical attention if fever exceeds 103°F (39.4°C), symptoms worsen after 10 days, or you experience difficulty breathing."),
            ("Hypertension", "Hypertension (high blood pressure) is defined as systolic BP ≥130 mmHg or diastolic BP ≥80 mmHg (AHA 2017 guidelines). It is a major risk factor for heart disease, stroke, and kidney disease. Lifestyle modifications: DASH diet, limit sodium to <2300 mg/day, regular aerobic exercise (150 min/week), limit alcohol, stop smoking, manage stress. First-line medications include ACE inhibitors, ARBs, thiazide diuretics, and calcium channel blockers. Regular BP monitoring is essential."),
            ("Type 2 Diabetes", "Type 2 diabetes is characterised by insulin resistance and relative insulin deficiency. Symptoms: polyuria, polydipsia, polyphagia, fatigue, blurred vision, slow wound healing. Diagnosis: fasting glucose ≥126 mg/dL, HbA1c ≥6.5%, or 2-hour OGTT ≥200 mg/dL. Management: diet (low glycaemic index, calorie control), regular exercise, weight loss, metformin (first-line), other oral agents or insulin as needed. Monitor HbA1c every 3 months until stable, then every 6 months."),
            ("Chest Pain – Emergency", "EMERGENCY: Chest pain can indicate a life-threatening condition. Call 911 / emergency services immediately if chest pain is: crushing, squeezing, or pressure-like; radiates to arm, jaw, neck, or back; accompanied by shortness of breath, sweating, nausea, or dizziness. These are signs of a potential heart attack (myocardial infarction). Do NOT drive yourself. Chew aspirin 325 mg if not allergic while waiting for help. Other causes of chest pain include angina, pulmonary embolism, aortic dissection, and pneumothorax – all require urgent evaluation."),
            ("Migraine", "Migraines are recurrent headaches causing moderate to severe pulsating pain, usually on one side of the head. Accompanied by nausea, vomiting, and sensitivity to light/sound. May be preceded by aura (visual disturbances, numbness). Triggers: stress, hormonal changes, certain foods (tyramine, caffeine, alcohol), sleep disruption, bright lights. Acute treatment: triptans (e.g. sumatriptan), NSAIDs, antiemetics. Preventive medications: beta-blockers, topiramate, amitriptyline, CGRP inhibitors. Keep a headache diary to identify triggers."),
            ("Asthma", "Asthma is a chronic inflammatory airway disease causing reversible airflow obstruction. Symptoms: wheezing, shortness of breath, chest tightness, cough (especially at night). Triggers: allergens, exercise, cold air, respiratory infections, smoke, strong odours. Diagnosis: spirometry showing reversible obstruction, peak flow monitoring. Treatment: reliever inhalers (SABA – salbutamol/albuterol) for acute symptoms; preventer inhalers (ICS – beclometasone, fluticasone) for long-term control. Severe asthma attack: use reliever inhaler, call emergency services if no improvement."),
            ("Depression", "Major depressive disorder (MDD) involves persistent depressed mood or loss of interest for ≥2 weeks, plus additional symptoms: changes in weight/appetite, sleep disturbances, fatigue, feelings of worthlessness, difficulty concentrating, and recurrent thoughts of death or suicide. Treatment: psychotherapy (CBT, IPT), antidepressants (SSRIs first-line – sertraline, fluoxetine), or combination. If you or someone you know is experiencing suicidal thoughts, contact the 988 Suicide & Crisis Lifeline (US) or local emergency services immediately."),
            ("Vaccination Schedule Adults", "Key adult vaccines (CDC recommendations): Influenza – annually; Tdap – once, then Td booster every 10 years; COVID-19 – stay up to date with current recommendations; Pneumococcal – PCV15 or PCV20 at age 65+; Shingles (Zoster) – Shingrix 2 doses at age 50+; MMR – 2 doses if not previously vaccinated; Hepatitis B – 3-dose series if not vaccinated; HPV – through age 26, shared decision up to 45. Consult your healthcare provider for personalised schedule."),
            ("Drug Interactions Warning", "Common dangerous drug interactions: Warfarin + NSAIDs (increased bleeding risk); MAOIs + SSRIs (serotonin syndrome); ACE inhibitors + potassium-sparing diuretics (hyperkalaemia); Statins + certain antibiotics/antifungals (myopathy risk); Metformin + contrast dye (lactic acidosis risk – hold 48h before/after); Opioids + benzodiazepines (respiratory depression). Always inform all your healthcare providers and pharmacist about every medication, supplement, and herbal remedy you take."),
            ("First Aid – Burns", "Burn treatment by degree – 1st degree (redness, pain): cool running water 10-20 min, do NOT use ice or butter, apply aloe vera, OTC pain reliever. 2nd degree (blisters): same cooling, do NOT pop blisters, cover loosely with sterile gauze, seek medical care if >3 inches or on face/hands/feet/genitals. 3rd degree (charring, white/black): Call 911, do NOT remove burned clothing, do NOT apply water to large areas (hypothermia risk), cover loosely with clean cloth. Chemical burns: remove clothing, brush off dry chemicals, flush with water 20+ min, call Poison Control."),
        ]

        docs = []
        for title, content in entries:
            docs.append(Document(
                page_content=f"[MEDICAL KNOWLEDGE: {title}]\n\n{content}",
                metadata={"source": "MedicalRAG_Seed", "topic": title},
            ))
        return docs

    # ── Memory ────────────────────────────────────────────────
    def _get_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        if session_id not in self._memories:
            self._memories[session_id] = ConversationBufferWindowMemory(
                k=8,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
            )
        return self._memories[session_id]

    def clear_session(self, session_id: str) -> None:
        self._memories.pop(session_id, None)

    # ── Tavily web search ─────────────────────────────────────
    def _web_search(self, query: str) -> List[Document]:
        if not self.tavily:
            return []
        try:
            results = self.tavily.search(
                query=f"medical health {query}",
                search_depth="advanced",
                max_results=TOP_K_WEB,
                include_domains=[
                    "mayoclinic.org", "webmd.com", "nih.gov",
                    "medlineplus.gov", "healthline.com", "who.int",
                    "cdc.gov", "nhs.uk", "pubmed.ncbi.nlm.nih.gov",
                ],
            )
            docs = []
            for r in results.get("results", []):
                docs.append(Document(
                    page_content=f"[WEB SOURCE: {r.get('url', '')}]\n{r.get('content', '')}",
                    metadata={"source": r.get("url", "web"), "title": r.get("title", "")},
                ))
            return docs
        except Exception as e:
            logger.warning(f"Tavily search failed: {e}")
            return []

    # ── Main chat method ──────────────────────────────────────
    def chat(self, question: str, session_id: str = "default") -> dict:
        try:
            memory = self._get_memory(session_id)
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": TOP_K_LOCAL, "fetch_k": 20},
            )

            # Gather context
            local_docs = retriever.invoke(question)
            web_docs   = self._web_search(question)
            all_docs   = local_docs + web_docs

            context_str = "\n\n---\n\n".join(d.page_content for d in all_docs)

            # Build prompt
            chat_history_str = ""
            for msg in memory.chat_memory.messages:
                role = "Patient" if msg.type == "human" else "MediAssist"
                chat_history_str += f"{role}: {msg.content}\n"

            prompt_value = MEDICAL_PROMPT.format(
                context=context_str or "No relevant context retrieved.",
                chat_history=chat_history_str,
                question=question,
            )

            response = self.llm.invoke(prompt_value)
            answer   = response.content

            # Save to memory
            memory.chat_memory.add_user_message(question)
            memory.chat_memory.add_ai_message(answer)

            # Detect emergency
            emergency_keywords = [
                "chest pain", "difficulty breathing", "stroke", "unconscious",
                "severe bleeding", "overdose", "anaphylaxis", "heart attack",
                "call 911", "emergency services",
            ]
            is_emergency = any(
                kw in answer.lower() or kw in question.lower()
                for kw in emergency_keywords
            )

            sources = []
            for d in all_docs[:6]:
                src = d.metadata.get("source", "Knowledge Base")
                if src not in sources:
                    sources.append(src)

            return {
                "answer": answer,
                "sources": sources,
                "is_emergency": is_emergency,
                "web_search_used": len(web_docs) > 0,
                "session_id": session_id,
            }

        except Exception as e:
            logger.error(f"Chat error: {e}", exc_info=True)
            return {
                "answer": f"I encountered an error processing your question: {str(e)}. Please try again.",
                "sources": [],
                "is_emergency": False,
                "web_search_used": False,
                "session_id": session_id,
            }

    # ── Document ingestion ────────────────────────────────────
    def add_documents(self, file_paths: List[str]) -> dict:
        documents = []
        for fp in file_paths:
            path = Path(fp)
            try:
                if path.suffix.lower() == ".pdf":
                    loader = PyPDFLoader(str(path))
                else:
                    loader = TextLoader(str(path), encoding="utf-8")
                documents.extend(loader.load())
                logger.info(f"Loaded: {path.name}")
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

        if not documents:
            return {"status": "error", "message": "No documents could be loaded."}

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(documents)
        self.vectorstore.add_documents(chunks)
        return {"status": "success", "chunks_added": len(chunks)}


# """
# medical_rag.py – RAG pipeline for the Medical Chatbot.

# Sources (in priority order):
#   1. Local ChromaDB vector store (PDF / text knowledge base)
#   2. Tavily medical web search (live results)

# Embeddings : HuggingFace all-MiniLM-L6-v2  (free, runs locally)
# LLM        : Groq  (llama-3.3-70b-versatile – free tier)
# """

# import os
# import logging
# from pathlib import Path
# from typing import List, Optional

# from dotenv import load_dotenv
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_groq import ChatGroq
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferWindowMemory
# from langchain.prompts import PromptTemplate
# from tavily import TavilyClient

# load_dotenv()
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ─── Paths ──────────────────────────────────────────────────
# BASE_DIR        = Path(__file__).parent
# DATA_DIR        = BASE_DIR / "data"
# VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "chroma_db"

# # ─── Constants ──────────────────────────────────────────────
# EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
# GROQ_MODEL    = "llama-3.3-70b-versatile"
# CHUNK_SIZE    = 800
# CHUNK_OVERLAP = 150
# TOP_K_LOCAL   = 4
# TOP_K_WEB     = 3

# # ─── Medical-tuned system prompt ────────────────────────────
# MEDICAL_PROMPT = PromptTemplate(
#     input_variables=["chat_history", "context", "question"],
#     template="""You are MediAssist, an expert AI medical information assistant powered by a curated medical knowledge base and live web search.

# IMPORTANT GUIDELINES:
# - Provide accurate, evidence-based medical information
# - Always recommend consulting a qualified healthcare professional for diagnosis and treatment
# - Never diagnose conditions or prescribe medications
# - Flag any urgent/emergency symptoms immediately (e.g. chest pain, difficulty breathing, stroke symptoms)
# - Cite whether information comes from your knowledge base or web search
# - Use plain language while being medically precise
# - Structure answers clearly with sections when appropriate

# CONTEXT FROM KNOWLEDGE BASE & WEB SEARCH:
# {context}

# CONVERSATION HISTORY:
# {chat_history}

# PATIENT QUESTION: {question}

# MEDIASSIST RESPONSE:""",
# )


# class MedicalRAG:
#     """End-to-end RAG pipeline for medical Q&A."""

#     def __init__(self):
#         self.groq_api_key   = os.getenv("GROQ_API_KEY", "")
#         self.tavily_api_key = os.getenv("TAVILY_API_KEY", "")

#         if not self.groq_api_key:
#             raise ValueError("GROQ_API_KEY not set in environment / .env file")

#         logger.info("Loading embedding model …")
#         self.embeddings = HuggingFaceEmbeddings(
#             model_name=EMBED_MODEL,
#             model_kwargs={"device": "cpu"},
#             encode_kwargs={"normalize_embeddings": True},
#         )

#         self.vectorstore = self._load_or_create_vectorstore()

#         logger.info("Initialising Groq LLM …")
#         self.llm = ChatGroq(
#             api_key=self.groq_api_key,
#             model=GROQ_MODEL,
#             temperature=0.2,
#             max_tokens=1500,
#         )

#         if self.tavily_api_key:
#             self.tavily = TavilyClient(api_key=self.tavily_api_key)
#             logger.info("Tavily web search enabled.")
#         else:
#             self.tavily = None
#             logger.warning("TAVILY_API_KEY not set – web search disabled.")

#         # Per-session memory stores  {session_id: ConversationBufferWindowMemory}
#         self._memories: dict[str, ConversationBufferWindowMemory] = {}

#     # ── Vector store ─────────────────────────────────────────
#     def _load_or_create_vectorstore(self) -> Chroma:
#         VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
#         DATA_DIR.mkdir(parents=True, exist_ok=True)

#         existing = list(VECTORSTORE_DIR.glob("*.sqlite3")) + list(VECTORSTORE_DIR.glob("chroma.sqlite3"))

#         if existing:
#             logger.info("Loading existing ChromaDB vector store …")
#             return Chroma(
#                 persist_directory=str(VECTORSTORE_DIR),
#                 embedding_function=self.embeddings,
#                 collection_name="medical_kb",
#             )

#         logger.info("No existing vector store found – building from data/ …")
#         return self._build_vectorstore()

#     def _build_vectorstore(self) -> Chroma:
#         documents: List[Document] = []

#         # ── built-in seed knowledge (always present) ─────────
#         seed_docs = self._get_seed_medical_docs()
#         documents.extend(seed_docs)

#         # ── load user PDFs / TXTs from data/ ─────────────────
#         pdf_files = list(DATA_DIR.glob("*.pdf"))
#         txt_files = list(DATA_DIR.glob("*.txt"))

#         for pdf in pdf_files:
#             try:
#                 loader = PyPDFLoader(str(pdf))
#                 documents.extend(loader.load())
#                 logger.info(f"Loaded PDF: {pdf.name}")
#             except Exception as e:
#                 logger.warning(f"Could not load {pdf.name}: {e}")

#         for txt in txt_files:
#             try:
#                 loader = TextLoader(str(txt), encoding="utf-8")
#                 documents.extend(loader.load())
#                 logger.info(f"Loaded TXT: {txt.name}")
#             except Exception as e:
#                 logger.warning(f"Could not load {txt.name}: {e}")

#         # ── chunk & embed ──────────────────────────────────────
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=CHUNK_SIZE,
#             chunk_overlap=CHUNK_OVERLAP,
#             separators=["\n\n", "\n", ".", "!", "?", " "],
#         )
#         chunks = splitter.split_documents(documents)
#         logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents.")

#         vs = Chroma.from_documents(
#             documents=chunks,
#             embedding=self.embeddings,
#             persist_directory=str(VECTORSTORE_DIR),
#             collection_name="medical_kb",
#         )
#         logger.info("Vector store built and persisted.")
#         return vs

#     def _get_seed_medical_docs(self) -> List[Document]:
#         """Hardcoded seed medical knowledge so the bot works out of the box."""
#         entries = [
#             ("Common Cold", "The common cold is a viral infection of the upper respiratory tract. Symptoms include runny nose, sore throat, cough, congestion, mild headache, sneezing, and low-grade fever. Most colds resolve within 7-10 days. Treatment is supportive: rest, hydration, and OTC medications for symptom relief. Antibiotics are NOT effective against viral infections. Seek medical attention if fever exceeds 103°F (39.4°C), symptoms worsen after 10 days, or you experience difficulty breathing."),
#             ("Hypertension", "Hypertension (high blood pressure) is defined as systolic BP ≥130 mmHg or diastolic BP ≥80 mmHg (AHA 2017 guidelines). It is a major risk factor for heart disease, stroke, and kidney disease. Lifestyle modifications: DASH diet, limit sodium to <2300 mg/day, regular aerobic exercise (150 min/week), limit alcohol, stop smoking, manage stress. First-line medications include ACE inhibitors, ARBs, thiazide diuretics, and calcium channel blockers. Regular BP monitoring is essential."),
#             ("Type 2 Diabetes", "Type 2 diabetes is characterised by insulin resistance and relative insulin deficiency. Symptoms: polyuria, polydipsia, polyphagia, fatigue, blurred vision, slow wound healing. Diagnosis: fasting glucose ≥126 mg/dL, HbA1c ≥6.5%, or 2-hour OGTT ≥200 mg/dL. Management: diet (low glycaemic index, calorie control), regular exercise, weight loss, metformin (first-line), other oral agents or insulin as needed. Monitor HbA1c every 3 months until stable, then every 6 months."),
#             ("Chest Pain – Emergency", "EMERGENCY: Chest pain can indicate a life-threatening condition. Call 911 / emergency services immediately if chest pain is: crushing, squeezing, or pressure-like; radiates to arm, jaw, neck, or back; accompanied by shortness of breath, sweating, nausea, or dizziness. These are signs of a potential heart attack (myocardial infarction). Do NOT drive yourself. Chew aspirin 325 mg if not allergic while waiting for help. Other causes of chest pain include angina, pulmonary embolism, aortic dissection, and pneumothorax – all require urgent evaluation."),
#             ("Migraine", "Migraines are recurrent headaches causing moderate to severe pulsating pain, usually on one side of the head. Accompanied by nausea, vomiting, and sensitivity to light/sound. May be preceded by aura (visual disturbances, numbness). Triggers: stress, hormonal changes, certain foods (tyramine, caffeine, alcohol), sleep disruption, bright lights. Acute treatment: triptans (e.g. sumatriptan), NSAIDs, antiemetics. Preventive medications: beta-blockers, topiramate, amitriptyline, CGRP inhibitors. Keep a headache diary to identify triggers."),
#             ("Asthma", "Asthma is a chronic inflammatory airway disease causing reversible airflow obstruction. Symptoms: wheezing, shortness of breath, chest tightness, cough (especially at night). Triggers: allergens, exercise, cold air, respiratory infections, smoke, strong odours. Diagnosis: spirometry showing reversible obstruction, peak flow monitoring. Treatment: reliever inhalers (SABA – salbutamol/albuterol) for acute symptoms; preventer inhalers (ICS – beclometasone, fluticasone) for long-term control. Severe asthma attack: use reliever inhaler, call emergency services if no improvement."),
#             ("Depression", "Major depressive disorder (MDD) involves persistent depressed mood or loss of interest for ≥2 weeks, plus additional symptoms: changes in weight/appetite, sleep disturbances, fatigue, feelings of worthlessness, difficulty concentrating, and recurrent thoughts of death or suicide. Treatment: psychotherapy (CBT, IPT), antidepressants (SSRIs first-line – sertraline, fluoxetine), or combination. If you or someone you know is experiencing suicidal thoughts, contact the 988 Suicide & Crisis Lifeline (US) or local emergency services immediately."),
#             ("Vaccination Schedule Adults", "Key adult vaccines (CDC recommendations): Influenza – annually; Tdap – once, then Td booster every 10 years; COVID-19 – stay up to date with current recommendations; Pneumococcal – PCV15 or PCV20 at age 65+; Shingles (Zoster) – Shingrix 2 doses at age 50+; MMR – 2 doses if not previously vaccinated; Hepatitis B – 3-dose series if not vaccinated; HPV – through age 26, shared decision up to 45. Consult your healthcare provider for personalised schedule."),
#             ("Drug Interactions Warning", "Common dangerous drug interactions: Warfarin + NSAIDs (increased bleeding risk); MAOIs + SSRIs (serotonin syndrome); ACE inhibitors + potassium-sparing diuretics (hyperkalaemia); Statins + certain antibiotics/antifungals (myopathy risk); Metformin + contrast dye (lactic acidosis risk – hold 48h before/after); Opioids + benzodiazepines (respiratory depression). Always inform all your healthcare providers and pharmacist about every medication, supplement, and herbal remedy you take."),
#             ("First Aid – Burns", "Burn treatment by degree – 1st degree (redness, pain): cool running water 10-20 min, do NOT use ice or butter, apply aloe vera, OTC pain reliever. 2nd degree (blisters): same cooling, do NOT pop blisters, cover loosely with sterile gauze, seek medical care if >3 inches or on face/hands/feet/genitals. 3rd degree (charring, white/black): Call 911, do NOT remove burned clothing, do NOT apply water to large areas (hypothermia risk), cover loosely with clean cloth. Chemical burns: remove clothing, brush off dry chemicals, flush with water 20+ min, call Poison Control."),
#         ]

#         docs = []
#         for title, content in entries:
#             docs.append(Document(
#                 page_content=f"[MEDICAL KNOWLEDGE: {title}]\n\n{content}",
#                 metadata={"source": "MedicalRAG_Seed", "topic": title},
#             ))
#         return docs

#     # ── Memory ────────────────────────────────────────────────
#     def _get_memory(self, session_id: str) -> ConversationBufferWindowMemory:
#         if session_id not in self._memories:
#             self._memories[session_id] = ConversationBufferWindowMemory(
#                 k=8,
#                 memory_key="chat_history",
#                 return_messages=True,
#                 output_key="answer",
#             )
#         return self._memories[session_id]

#     def clear_session(self, session_id: str) -> None:
#         self._memories.pop(session_id, None)

#     # ── Tavily web search ─────────────────────────────────────
#     def _web_search(self, query: str) -> List[Document]:
#         if not self.tavily:
#             return []
#         try:
#             results = self.tavily.search(
#                 query=f"medical health {query}",
#                 search_depth="advanced",
#                 max_results=TOP_K_WEB,
#                 include_domains=[
#                     "mayoclinic.org", "webmd.com", "nih.gov",
#                     "medlineplus.gov", "healthline.com", "who.int",
#                     "cdc.gov", "nhs.uk", "pubmed.ncbi.nlm.nih.gov",
#                 ],
#             )
#             docs = []
#             for r in results.get("results", []):
#                 docs.append(Document(
#                     page_content=f"[WEB SOURCE: {r.get('url', '')}]\n{r.get('content', '')}",
#                     metadata={"source": r.get("url", "web"), "title": r.get("title", "")},
#                 ))
#             return docs
#         except Exception as e:
#             logger.warning(f"Tavily search failed: {e}")
#             return []

#     # ── Main chat method ──────────────────────────────────────
#     def chat(self, question: str, session_id: str = "default") -> dict:
#         try:
#             memory = self._get_memory(session_id)
#             retriever = self.vectorstore.as_retriever(
#                 search_type="mmr",
#                 search_kwargs={"k": TOP_K_LOCAL, "fetch_k": 20},
#             )

#             # Gather context
#             local_docs  = retriever.invoke(question)
#             web_docs    = self._web_search(question)
#             all_docs    = local_docs + web_docs

#             context_str = "\n\n---\n\n".join(d.page_content for d in all_docs)

#             # Build prompt
#             chat_history_str = ""
#             for msg in memory.chat_memory.messages:
#                 role = "Patient" if msg.type == "human" else "MediAssist"
#                 chat_history_str += f"{role}: {msg.content}\n"

#             prompt_value = MEDICAL_PROMPT.format(
#                 context=context_str or "No relevant context retrieved.",
#                 chat_history=chat_history_str,
#                 question=question,
#             )

#             response = self.llm.invoke(prompt_value)
#             answer   = response.content

#             # Save to memory
#             memory.chat_memory.add_user_message(question)
#             memory.chat_memory.add_ai_message(answer)

#             # Detect emergency
#             emergency_keywords = [
#                 "chest pain", "difficulty breathing", "stroke", "unconscious",
#                 "severe bleeding", "overdose", "anaphylaxis", "heart attack",
#                 "call 911", "emergency services",
#             ]
#             is_emergency = any(kw in answer.lower() or kw in question.lower()
#                                for kw in emergency_keywords)

#             sources = []
#             for d in all_docs[:6]:
#                 src = d.metadata.get("source", "Knowledge Base")
#                 if src not in sources:
#                     sources.append(src)

#             return {
#                 "answer": answer,
#                 "sources": sources,
#                 "is_emergency": is_emergency,
#                 "web_search_used": len(web_docs) > 0,
#                 "session_id": session_id,
#             }

#         except Exception as e:
#             logger.error(f"Chat error: {e}", exc_info=True)
#             return {
#                 "answer": f"I encountered an error processing your question: {str(e)}. Please try again.",
#                 "sources": [],
#                 "is_emergency": False,
#                 "web_search_used": False,
#                 "session_id": session_id,
#             }

#     # ── Document ingestion ────────────────────────────────────
#     def add_documents(self, file_paths: List[str]) -> dict:
#         documents = []
#         for fp in file_paths:
#             path = Path(fp)
#             try:
#                 if path.suffix.lower() == ".pdf":
#                     loader = PyPDFLoader(str(path))
#                 else:
#                     loader = TextLoader(str(path), encoding="utf-8")
#                 documents.extend(loader.load())
#                 logger.info(f"Loaded: {path.name}")
#             except Exception as e:
#                 logger.warning(f"Failed to load {path}: {e}")

#         if not documents:
#             return {"status": "error", "message": "No documents could be loaded."}

#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=CHUNK_SIZE,
#             chunk_overlap=CHUNK_OVERLAP,
#         )
#         chunks = splitter.split_documents(documents)
#         self.vectorstore.add_documents(chunks)
#         return {"status": "success", "chunks_added": len(chunks)}
