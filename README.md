# Customer Support System v2 (LangGraph Edition)

An AI-powered customer support system built using a parallel LangGraph pipeline, Redis caching, ChromaDB RAG, and full observability. This system replaces the legacy CrewAI implementation to provide a more robust, stateful, and observable support agent architecture.

## 🌟 Key Features

*   **LangGraph Agent Pipeline:** Utilizes stateful, parallel agents for efficient query handling, escalation, and resolution.
*   **Mistral LLM Integration:** Powered by Mistral via `langchain_mistralai`.
*   **Vector Database (RAG):** Uses **ChromaDB** for semantic search over company policies and a knowledge base.
*   **Caching Layer:** **Redis** caching for quick retrieval of common queries and trending topics.
*   **FastAPI Backend:** A robust API with built-in CORS, GZip compression, and observability traces.
*   **React + Vite Frontend:** 
    *   **Customer Portal:** A chat interface for users to interact with the AI support agent and upload images for complaints.
    *   **Admin Dashboard:** A secured portal for human agents to review escalated tickets, approve/reject pending matters, and monitor AI analytics.
*   **AI Observability:** Integrated guardrails, fallback mechanisms, and evaluation engines.

## 📁 Project Structure

*   `backend/`: Contains the FastAPI application, LangGraph agents, observability engines, and backend services.
*   `frontend/`: The React + Vite application (Customer Portal & Admin Dashboard).
*   `data/`: Raw data files (e.g., `knowledge_base.json`, `company_policies.csv`) used to populate the vector database.
*   `vector_db/`: Local ChromaDB storage for RAG.
*   `uploads/`: Directory where user complaint images are stored.

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.9+ and Node.js installed. 

### 2. Environment Setup
Create a `.env` file in the root directory (you can copy from an existing template if available) and add your API keys:
```env
MISTRAL_API_KEY=your_mistral_api_key
REDIS_URL=redis://localhost:6379 # Optional: Falls back to in-memory if not provided
LANGSMITH_API_KEY=your_langsmith_key # Optional: For tracing
ADMIN_EMAIL=admin@yourcompany.com # Required for Admin Dashboard OTP login
SMTP_USER=your_gmail_address # Required for sending OTP emails
SMTP_APP_PASSWORD=your_gmail_app_password
```

### 3. Backend Setup
Create and activate a virtual environment, then install the dependencies:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # On Windows
pip install -r requirements.txt
```

### 4. Initialize the Knowledge Base
Run the ingestion script to populate the ChromaDB vector store with your knowledge base and policies:
```powershell
python ingest_kb.py
```

### 5. Running the Backend Server
Start the FastAPI server (it runs on port 8000 by default):
```powershell
python run_server.py --reload
```

### 6. Running the Frontend
In a new terminal window, navigate to the `frontend` directory, install Node packages, and start the Vite dev server:
```powershell
cd frontend
npm install
npm run dev
```

## 🛠️ Usage

*   **Customer Portal:** Navigate to `http://localhost:5173/` to interact with the AI agent.
*   **Admin Dashboard:** Navigate to `http://localhost:5173/admin` to view analytics and pending approvals. You will be prompted to enter your admin email to receive an OTP.
*   **API Documentation:** Once the backend is running, you can view the Swagger UI at `http://localhost:8000/docs`.

## 📜 Architecture

*   **API:** FastAPI (`backend/api/main.py`)
*   **Graph/Agents:** LangGraph (`backend/langgraph_system/`)
*   **Services:** `KnowledgeBaseService`, `PolicyService`, `CacheService`
*   **Frontend:** React, React Router, Vite
*   **Vector DB:** ChromaDB
*   **LLM:** MistralAI
