# 🧠 EV Assistant

A Retrieval-Augmented Generation (RAG) chatbot for EV (electric vehicle) diagnostics, built with **Streamlit**, **Qdrant**, **LangChain**, and **Groq**. Upload technical PDFs, ask diagnostic questions, and get grounded answers backed only by your knowledge base — with a self-improving memory that learns from corrections over time.

Live demo: [evdiagnosis.streamlit.app](https://evdiagnosis.streamlit.app/)

---

## ✨ Features

- **Document‑grounded Q&A** — answers are generated only from uploaded PDF content and past corrections, with an explicit "insufficient information" fallback to avoid hallucination.
- **HyDE retrieval (optional)** — generates a hypothetical answer passage to improve semantic search recall before hitting the vector store.
- **Cross‑encoder reranking** — retrieved chunks are reranked with `cross-encoder/ms-marco-MiniLM-L-6-v2` for higher‑precision context selection.
- **Self‑learning memory** — 👍/👎 feedback on answers lets admins save corrections into a dedicated Qdrant collection (`ev_memory`), which is retrieved alongside document context on future queries.
- **Password‑protected admin controls** — separate gates for PDF uploads and feedback/correction management, backed by an admin Groq key.
- **Bring‑your‑own‑key mode** — non‑admin users can supply their own Groq API key to chat without needing upload/feedback access.
- **Dynamic model selection** — fetches and lists all Groq models available to the active API key.
- **Automatic Qdrant collection creation** — creates required collections (`ev_docs` and `ev_memory`) on first use if they don't already exist.

---

## 🏗️ How It Works

1. **Ingestion** — Admins upload PDFs via the sidebar. Each document is split into chunks (`RecursiveCharacterTextSplitter`), embedded with a HuggingFace sentence‑transformer model, and upserted into a Qdrant `ev_docs` collection with UUID‑based IDs (Qdrant requires UUIDs or integers, so we generate valid UUIDs for each chunk).
2. **Retrieval** — On each user question, relevant chunks are retrieved from Qdrant (optionally via a HyDE‑generated hypothetical passage), then reranked with a cross‑encoder to surface the most relevant results.
3. **Memory recall** — Past user‑confirmed answers and corrections are retrieved from a separate `ev_memory` collection and passed to the LLM as additional context.
4. **Generation** — A Groq‑hosted LLM (default: `llama3‑70b‑8192`) generates an answer strictly from the retrieved context and memory, following a diagnostic‑style prompt (diagnosis → evidence → next checks → facts vs. recommendations).
5. **Feedback loop** — When feedback mode is enabled, 👍 saves the Q&A pair to memory as‑is; 👎 prompts the admin for the correct answer, which is then saved to memory for future retrieval.

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|------------|
| UI | [Streamlit](https://streamlit.io/) |
| Orchestration | [LangChain](https://www.langchain.com/) |
| Vector database | [Qdrant Cloud](https://cloud.qdrant.io/) (managed, free tier available) |
| Embeddings | HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2` by default) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM inference | [Groq](https://groq.com/) |
| PDF parsing | `pypdf` via `PyPDFLoader` |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- A [Qdrant Cloud](https://cloud.qdrant.io/) account (free tier, no credit card required) – you'll get a cluster URL and an API key.
- A [Groq](https://console.groq.com/) account and API key.

### Installation

```bash
git clone https://github.com/<your-username>/ragai.git
cd ragai
pip install -r requirements.txt
