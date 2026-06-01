# Financial Deep Research Agent 🤖📊

> A multi-agent AI system that orchestrates a streamlined LangGraph pipeline — from combined query routing & planning to investor-grade financial report generation — powered by Gemini 2.5 Flash, Tavily, and yfinance.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Overview

The **Financial Deep Research Agent** is a full-stack AI application that accepts a plain-English financial query (e.g., *"Analyze HDFC Bank's competitive position in Indian private banking"*) and returns a structured, Wall Street–style research report with 6 canonical sections. It rejects non-financial queries at the router stage, auto-classifies the sector, extracts ticker symbols (with smart Indian exchange suffix detection), runs 4 parallelized web searches, pulls live market data from Yahoo Finance, cross-references qualitative and quantitative findings, and writes the final report — all in a single API call.

The system solves the core problem of free-tier LLM rate limits by implementing a `GeminiLLMClientRotator` that manages up to 6 Gemini API keys, instantly rotating to the next key on a `429 RESOURCE_EXHAUSTED` error instead of waiting. It also exposes a **fast research path** (`stream_fast_research`) that completes in exactly 2 Gemini calls for production-grade speed. It is aimed at retail investors, financial analysts, and developers building AI-native fintech tools.

---

## ✨ Features

### Combined Query Routing & Planning
- **`router_planner_node`** — a single merged LangGraph node that uses the `RouterPlannerOutput` Pydantic schema to (1) determine if the query is finance-related, (2) classify the sector (IT, Banking, Pharma, FMCG, Auto, Energy, etc.), (3) extract the stock ticker with exchange suffix (`.NS` for NSE, `.BO` for BSE, plain for US), and (4) generate exactly 4 specific search queries — all in one Gemini call; non-finance queries are rejected immediately with a polite message

### Research & Analysis (Single Merged Node)
- **`researcher_analyst_node`** — combines web search, RAG retrieval, and LLM synthesis into a single node: fires 4 concurrent Tavily searches via `asyncio.gather`, retrieves any uploaded PDF context from ChromaDB, then synthesizes all data into a 4-heading structured analyst memo (Market Overview, Financial Performance, Competitive Position, Key Risks & Opportunities) in a single Gemini call

### Quantitative Finance Layer
- **`finance_node`** — deterministically fetches live market data using the ticker identified by `router_planner_node`; includes smart Indian exchange detection logic that tries `TICKER.NS` first, then `.BO`, before falling back to the raw symbol
- **Fetched Metrics** — market cap, trailing PE, forward PE, profit margins, 52-week high/low, dividend yield, and revenue growth pulled directly from `yfinance` `info` dict
- **Supporting Tools** — `calculations.py` computes 50-day and 200-day SMAs; `ratios.py` extracts PEG Ratio, Price-to-Book, Debt-to-Equity, ROE, and EBITDA; `market_data.py` handles income statements, balance sheet, and cash flow

### Reporting
- **`reporter_node`** — synthesizes all research context into a strict 6-section Markdown report (Executive Summary → Sources) following a Wall Street analyst system prompt capped at 800 words; explicitly instructs the LLM to write "Data not available" rather than hallucinate numbers

### Fast Research Path
- **`stream_fast_research()`** — the production path used by `/api/v1/research/`; makes exactly 2 Gemini calls (brainstorm via `BrainstormResult` schema + master analyst synthesis) and 4 concurrent Tavily searches, yielding NDJSON progress updates (`update`, `report_chunk`, `error`) for streaming-friendly frontends, bypassing the full LangGraph graph for speed

### RAG / Document Intelligence
- **PDF Ingestion** — users can upload 10-Ks and Annual Reports via the Streamlit sidebar; `pdf_parser.py` chunks the document and `vectorstore.py` stores embeddings in a local ChromaDB instance under `chroma_db/`; a **Reset Vector DB** button clears all stored embeddings via `/api/v1/documents/reset`
- **Local Embeddings** — uses `fastembed` alongside `all-MiniLM-L6-v2` via `HuggingFaceEmbeddings` as a singleton to avoid repeated model loads; runs entirely locally with no additional API calls
- **RAG in Researcher Node** — `retrieve_financial_context()` is called at the start of `researcher_analyst_node` and its output is fed directly into the analyst synthesis prompt alongside live web data

### API Key Rotation
- **`GeminiLLMClientRotator`** — a smart singleton LLM client that accepts up to 6 `GOOGLE_API_KEY_*` values, tries them in priority order (Key 1 always preferred), marks a key as exhausted for 60 seconds on a 429 error, and picks up from the exact retry-delay hint embedded in the error message via regex

### Frontend
- **Streamlit Chat UI** (`frontend/app.py`) — a wide-layout chat interface with a sidebar for optional ticker input, a PDF upload + ChromaDB ingestion button with chunk count and time-taken feedback, a vector DB reset button, a backend health indicator, and a persistent `st.session_state` chat history

---

## 🛠️ Tech Stack

### Backend

| Technology | Purpose |
|---|---|
| FastAPI 0.115+ | REST API framework; hosts `/api/v1/research/`, `/api/v1/documents/upload`, and `/api/v1/documents/reset` |
| Uvicorn | ASGI server to run the FastAPI app |
| LangGraph 0.2+ | Directed graph runtime for 4-node multi-agent state orchestration |
| LangChain Core 0.3+ | Message types, structured output, and tool interfaces |
| `langchain-google-genai` | Gemini 2.5 Flash client with structured output support |
| `tavily-python` | Web search API with per-result title, URL, and content |
| yfinance 0.2.37 | Live and historical market data from Yahoo Finance |
| Pydantic 2.9+ | Typed state schemas, structured LLM output validation |
| `pydantic-settings` | `.env` file loading into the `Settings` config object |
| Tenacity | Retry logic utilities |
| PyMuPDF / pypdf | PDF parsing for RAG ingestion |

### Frontend

| Technology | Purpose |
|---|---|
| Streamlit 1.32 | Chat UI with sidebar controls, PDF upload, and `st.session_state` history |
| Requests | HTTP client to call the FastAPI backend |

### AI & Embeddings

| Technology | Purpose |
|---|---|
| Gemini 2.5 Flash (`gemini-2.5-flash`) | All LLM inference — routing, planning, analysis, reporting |
| `all-MiniLM-L6-v2` (HuggingFace) + `fastembed` | Local sentence embeddings for RAG document chunks |
| ChromaDB | Local vector store for PDF-ingested financial documents |

### Dev Tooling

| Technology | Purpose |
|---|---|
| python-dotenv | Local `.env` loading for development |
| logging | Structured INFO/ERROR logs throughout all agent nodes |
| asyncio | Concurrent search execution inside `researcher_analyst_node` and `stream_fast_research` |

---

## 🏗️ Architecture / How It Works

```
User Query (via Streamlit or REST)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                  FastAPI Backend                         │
│  POST /api/v1/research/  →  stream_fast_research()       │
│         OR                                               │
│  LangGraph Graph  →  create_research_graph()             │
└───────────────────────────┬─────────────────────────────┘
                            │
          ┌─────────────────▼──────────────────┐
          │       LangGraph StateGraph          │
          │  (ResearchState typed dict)          │
          └─────────────────────────────────────┘
                            │
               ┌────────────▼────────────┐
               │  1. router_planner_node  │  ← Gemini: RouterPlannerOutput
               │  (Rejected → END)        │    sector + ticker + 4 queries
               └────────────┬────────────┘
                            │ finance query
               ┌────────────▼────────────┐
               │  2. finance_node         │  ← yfinance: price, ratios,
               │  (Indian .NS/.BO logic)  │    52wk hi/lo, margins
               └────────────┬────────────┘
                            │
               ┌────────────▼────────────┐
               │  3. researcher_analyst   │  ← RAG retrieval + 4 Tavily searches
               │     _node               │    + Gemini: 4-heading analyst memo
               └────────────┬────────────┘
                            │
               ┌────────────▼────────────┐
               │  4. reporter_node        │  ← Gemini: 6-section Markdown report
               └────────────┬────────────┘
                            │
                      final_report string
                            │
               ┌────────────▼────────────┐
               │  ResearchResponse JSON   │
               │  → Streamlit renders     │
               │    Markdown in chat      │
               └─────────────────────────┘
```

**Fast path:** `stream_fast_research()` in `services/fast_research.py` is the production path used by the `/api/v1/research/` route. It makes exactly 2 Gemini calls (brainstorm via `BrainstormResult` + master analyst) and 4 concurrent Tavily searches, yielding NDJSON progress events, bypassing the full LangGraph graph for speed.

**Key rotation flow:** Every `ainvoke` or `ainvoke_structured` call on `gemini_client` goes through `GeminiLLMClientRotator._get_next_client()`, which scans the in-memory `_exhausted_until` dict and instantly picks the first available key. On a `429`, the key is marked exhausted using the delay extracted from the error message itself (via regex on `retry_in: Xs`).

---

## 📂 Folder Structure

```
Financial-Deep-Research-Agent-main/
│
├── backend/
│   ├── .env.example                  # Template for all required environment variables
│   ├── requirements.txt              # All Python dependencies
│   ├── frontend.py                   # Alternate Streamlit launcher from within backend/
│   ├── check_models.py               # Utility: list available Gemini models
│   ├── list_models.py                # Utility: enumerate API-accessible models
│   ├── ping.py                       # Utility: health-check script
│   │
│   └── app/
│       ├── main.py                   # FastAPI app factory; registers all routers
│       ├── core/
│       │   └── config.py             # Pydantic Settings — loads .env.dev into settings object
│       │
│       ├── models/
│       │   └── schemas.py            # ResearchRequest, ResearchResponse, DocumentUploadResponse
│       │
│       ├── api/
│       │   └── routes/
│       │       ├── health.py         # GET /health/ — liveness probe
│       │       ├── research.py       # POST /api/v1/research/ — main research trigger (streaming)
│       │       └── documents.py      # POST /api/v1/documents/upload + /reset — PDF ingestion
│       │
│       ├── graph/
│       │   ├── workflow.py           # LangGraph StateGraph: 4-node linear graph definition
│       │   └── nodes.py              # Re-exports all 4 agent node callables
│       │
│       ├── state/
│       │   ├── __init__.py
│       │   └── research_state.py     # ResearchState TypedDict — single source of truth for graph state
│       │
│       ├── agents/
│       │   ├── shared/
│       │   │   └── llm.py            # GeminiLLMClientRotator singleton (gemini_client)
│       │   ├── nodes/
│       │   │   ├── router_planner.py # router_planner_node — combined sector router + planner (1 Gemini call)
│       │   │   ├── researcher_analyst.py # researcher_analyst_node — RAG + web search + analyst memo
│       │   │   └── finance.py        # finance_node — yfinance data pull with Indian .NS/.BO detection
│       │   └── reporter/
│       │       └── reporter.py       # reporter_node — 6-section final report writer
│       │
│       ├── services/
│       │   ├── fast_research.py      # 2-call streaming fast pipeline (production path for /research/)
│       │   ├── finance.py            # Helper finance service utilities
│       │   ├── gemini.py             # Standalone Gemini service wrapper
│       │   ├── tavily_search.py      # Tavily service wrapper
│       │   └── vectordb.py           # Vector DB service helper
│       │
│       └── tools/
│           ├── __init__.py
│           ├── finance/
│           │   ├── market_data.py    # get_company_info, get_historical_prices, get_financial_statements
│           │   ├── calculations.py   # calculate_moving_averages (SMA50/200), calculate_yoy_growth
│           │   └── ratios.py         # extract_key_ratios (PE, PEG, ROE, margins, EBITDA)
│           ├── rag/
│           │   ├── pdf_parser.py     # Parses + chunks PDF files for ingestion
│           │   ├── embeddings.py     # Singleton local embeddings (all-MiniLM-L6-v2 / fastembed)
│           │   ├── vectorstore.py    # ChromaDB singleton — persists to ./chroma_db/
│           │   └── retriever.py      # retrieve_financial_context() — RAG retrieval called in researcher node
│           └── search/
│               ├── __init__.py
│               └── tavily_search.py  # TavilySearchTool wrapper with async .search() method
│
└── frontend/
    └── app.py                        # Streamlit chat UI — connects to FastAPI backend
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.11+
- A [Google AI Studio](https://aistudio.google.com/) account (free tier works; up to 6 API keys supported)
- A [Tavily](https://tavily.com/) API key (free tier: 1,000 searches/month)

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/Financial-Deep-Research-Agent.git
cd Financial-Deep-Research-Agent
```

### 3. Install Backend Dependencies
```bash
cd backend
pip install -r requirements.txt
```

> **Note:** `yfinance` is pinned at `0.2.37` in `requirements.txt`. If you encounter Pydantic version conflicts, install with `pip install pydantic==2.9.0 --force-reinstall`.

### 4. Configure Environment Variables
```bash
cp .env.example .env.dev
```

Open `.env.dev` and fill in your keys (see [Environment Variables](#-environment-variables) below).

### 5. Run the Backend
```bash
# From within the backend/ directory
uvicorn app.main:app --reload --port 8001
```

The API will be available at `http://127.0.0.1:8001`. Visit `http://127.0.0.1:8001/docs` for the interactive Swagger UI.

### 6. Run the Frontend
```bash
# In a separate terminal, from the project root
streamlit run frontend/app.py
```

The Streamlit UI will open at `http://localhost:8501`.

### 7. Verify the Connection
The Streamlit sidebar displays a green **✅ Backend: Online** badge when the FastAPI health endpoint responds successfully.

---

## 🔑 Environment Variables

All variables are loaded from `backend/.env.dev` via `pydantic-settings`.

| Variable | Description | Example |
|---|---|---|
| `GOOGLE_API_KEY` | Primary Gemini API key (Key 1 — always tried first) | `AIzaSy...` |
| `GOOGLE_API_KEY_2` | Fallback Gemini key — used when Key 1 hits quota | `AIzaSy...` |
| `GOOGLE_API_KEY_3` | Third Gemini key in the rotation pool | `AIzaSy...` |
| `GOOGLE_API_KEY_4` | Fourth Gemini key | `AIzaSy...` |
| `GOOGLE_API_KEY_5` | Fifth Gemini key | `AIzaSy...` |
| `GOOGLE_API_KEY_6` | Sixth Gemini key (max rotation depth) | `AIzaSy...` |
| `TAVILY_API_KEY` | Tavily web search API key | `tvly-...` |
| `ENVIRONMENT` | App environment flag | `development` |
| `LOG_LEVEL` | Python logging verbosity | `info` |

> You only need `GOOGLE_API_KEY` and `TAVILY_API_KEY` to get started. Additional `GOOGLE_API_KEY_*` entries increase throughput on free-tier quotas.

---

## 🧪 Usage

### Via the Streamlit Chat UI

1. Open `http://localhost:8501`
2. (Optional) Enter a stock ticker in the sidebar (e.g., `AAPL`, `HDFCBANK.NS`). Leave blank to let the AI auto-detect it from your query.
3. Type a financial query in the chat input, for example:
   - `"Provide a comprehensive analysis of Infosys's position in the IT services market"`
   - `"What is the investment outlook for Tesla in 2025?"`
   - `"Analyze HDFC Bank's competitive position in Indian private banking"`
4. Wait 20–60 seconds while the pipeline runs. A status indicator shows progress updates from the streaming endpoint.
5. The 6-section Markdown report renders inline in the chat.

### Via the RAG Knowledge Base (PDF Upload)

1. In the sidebar, upload a PDF Annual Report or 10-K.
2. Enter the stock ticker in the ticker field (required for metadata tagging in ChromaDB).
3. Click **⚡ Ingest into Local Vector DB** — the system chunks the PDF, embeds it locally, and stores it in `backend/chroma_db/`. The UI confirms the number of chunks processed and time taken.
4. Subsequent research queries automatically retrieve relevant passages via `retrieve_financial_context()` inside `researcher_analyst_node`.
5. Use **🗑️ Reset Vector DB** in the sidebar to clear all stored embeddings and start fresh.

### Via the REST API
```bash
curl -X POST "http://127.0.0.1:8001/api/v1/research/" \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze Apple competitive moat", "company_symbol": "AAPL"}'
```

Response:
```json
{
  "status": "completed",
  "final_report": "# Apple Inc. - Comprehensive Research Report\n\n## 1. Executive Summary\n..."
}
```

---

## 📸 Screenshots / Demo

| View | Screenshot |
|---|---|
| Streamlit Chat Interface | *(add `screenshots/chat_ui.png`)* |
| Research Report Output | *(add `screenshots/report_output.png`)* |
| Sidebar – PDF Upload | *(add `screenshots/sidebar_upload.png`)* |
| FastAPI Swagger Docs | *(add `screenshots/swagger_docs.png`)* |

> To capture: run the app, ask a query like "Analyze TSLA", and screenshot the rendered Markdown report.

---

## 🚧 Challenges & Learnings

**1. Free-Tier Rate Limits on Gemini Flash**
The system makes 2 Gemini calls per fast research run and 3 calls per full LangGraph pipeline run. On free tier, `429 RESOURCE_EXHAUSTED` errors happen within 2–3 consecutive queries. The solution was `GeminiLLMClientRotator` — it reads the `retry_in: Xs` value embedded in the error string via regex and sets the exhaustion timer to exactly that duration plus a 5-second buffer, rather than a naive flat 60-second backoff. This eliminated most unnecessary wait time.

**2. Merging Router + Planner into One Node**
The original design had a dedicated `router_node` (sector classification) and a separate `planner_node` (research plan generation) as two sequential Gemini calls. Merging them into `router_planner_node` with the `RouterPlannerOutput` schema — is_finance_query, sector, ticker, and 4 search queries in one structured call — halved the LLM call count for the LangGraph path and dramatically reduced quota consumption.

**3. Gemini Structured Output Content Blocks**
`ChatGoogleGenerativeAI.ainvoke()` returns a `BaseMessage` whose `.content` field can be either a plain string or a list of content dicts (e.g., `[{"type": "text", "text": "..."}]`). The `reporter_node` includes an explicit list-flattening guard to handle both formats, preventing silent `AttributeError` crashes in production.

**4. yfinance on Non-US Tickers**
Indian exchange tickers (e.g., `HDFCBANK.NS`) require the `.NS` suffix for NSE or `.BO` for BSE. `finance_node` now implements a detection heuristic: it checks for Indian keywords in the query and known Indian sectors, then tries `TICKER.NS` first, then `TICKER.BO`, before falling back to the raw ticker. The `get_company_info` function logs errors but returns empty dicts rather than raising, so the pipeline gracefully degrades to qualitative-only reports.

**5. ChromaDB Persistence Path Resolution**
`VectorStoreManager` resolves `CHROMA_PERSIST_DIR` using `os.getcwd()` at import time, which means the resolved path depends on which directory Uvicorn is launched from. If Uvicorn is started from the project root rather than `backend/`, the `chroma_db/` directory ends up in the wrong location. This is a subtle operational footgun that surfaces only when re-ingesting documents across restarts.

---

## 🔮 Future Improvements

**1. Streaming Token-by-Token Report Output**
The `stream_fast_research()` function already yields NDJSON `update` and `report_chunk` events. Connecting the Streamlit frontend to consume these events via `requests.Session().get()` with `stream=True` would allow the report to render word-by-word, dramatically improving perceived latency on slow API tiers.

**2. Full RAG Integration into LangGraph Graph**
`ResearchState` already includes a `retrieved_documents` field annotated for accumulation and `researcher_analyst_node` already calls `retrieve_financial_context()`. The remaining work is ensuring the ChromaDB retriever is fully wired when no PDFs have been uploaded (currently returns "None uploaded"), and surfacing retrieved chunk scores in the analyst memo.

**3. LangGraph Persistence / Checkpointing**
LangGraph supports checkpointing via `SqliteSaver` or `PostgresSaver`. Adding a checkpointer to `create_research_graph()` would allow interrupted pipelines to resume from the last completed node rather than restarting from scratch — useful for multi-step queries on slow API tiers.

**4. OpenAI / Other LLM Backends**
`langchain-openai` is already in `requirements.txt` and `ResearchState` in `research_state.py` includes a generic `messages` field. Abstracting `gemini_client` behind a `BaseLLMClient` interface would enable swapping to GPT-4o or Claude with a single config change.

**5. Structured Report Export**
The `final_report` is currently a raw Markdown string. Adding a `/api/v1/research/export` endpoint that converts the Markdown to PDF using `weasyprint` or `reportlab` would produce shareable investor-grade documents without copy-pasting from the chat UI.

**6. Automated Test Suite**
The utility scripts in `backend/` (`check_models.py`, `ping.py`, `list_models.py`) are standalone scripts. Migrating them to a `pytest-asyncio` suite with shared fixtures for `gemini_client` mocking and FastAPI's `TestClient` would enable CI integration and faster local development feedback.

---

## 🤝 Contributing

1. Fork the repository and create a branch: `git checkout -b feat/your-feature`
2. Make your changes; follow existing module patterns (each agent node is an `async def node(state: ResearchState) -> Dict[str, Any]`)
3. Commit with conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`
4. Open a pull request with a description of what the change does and which agent node it affects

**Code style requirements:**
- Python 3.11+ type hints throughout; Pydantic models for all LLM structured outputs
- All LLM calls must go through `gemini_client` (the `GeminiLLMClientRotator` singleton) — never instantiate `ChatGoogleGenerativeAI` directly in agent nodes
- Use `logger = logging.getLogger(__name__)` in every module; log at INFO for pipeline milestones, ERROR for exceptions
- Agent node functions must be `async def` and return a partial `ResearchState` dict

**Good first issues:**
- Surface RAG retrieval scores in the `researcher_analyst_node` synthesis prompt for better source attribution
- Fix `CHROMA_PERSIST_DIR` to use an absolute path relative to `__file__` instead of `os.getcwd()`
- Add a `pytest` test for `GeminiLLMClientRotator` key rotation logic using mocked `ChatGoogleGenerativeAI` clients
- Extend `router_planner_node` to detect Gulf/Asian exchange tickers (`.AX`, `.HK`, `.SA`) alongside the existing `.NS`/`.BO` logic

---

## 📜 License

MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
