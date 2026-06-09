"""
FastAPI Application — Customer Support System v2 (LangGraph)
Replaces the old CrewAI main.py.

Features:
  - CORS configured for frontend dev server
  - Lifespan startup: initializes all services once
  - Redis + ChromaDB + LangGraph graph mounted on app.state
  - All microservices injected via app.state (no global singletons)
"""
from __future__ import annotations

import base64
import logging
import os
import shutil
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ─── Lifespan: boot all services once ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialize all services. Shutdown: cleanup."""
    logger.info("🚀 Starting Customer Support System v2 (LangGraph)")

    # ── Paths ────────────────────────────────────────────────────────────────
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    src_dir     = os.path.abspath(os.path.join(base_dir, ".."))
    root_dir    = os.path.abspath(os.path.join(src_dir, "..", ".."))
    data_dir    = os.path.join(root_dir, "data")
    vdb_dir     = os.path.join(root_dir, "vector_db")
    uploads_dir = os.path.join(root_dir, "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    app.state.uploads_dir = uploads_dir

    # ── Cache (Redis) ────────────────────────────────────────────────────────
    from ..services.cache_service import CacheService
    app.state.cache_service = CacheService()
    logger.info(f"Cache: {app.state.cache_service.health()['status']}")

    # ── Knowledge Base (ChromaDB) ────────────────────────────────────────────
    from ..services.kb_service import KnowledgeBaseService
    app.state.kb_service = KnowledgeBaseService(vector_db_path=vdb_dir)
    logger.info(f"KB: {app.state.kb_service.health()}")

    # ── Policy Service ───────────────────────────────────────────────────────
    from ..services.policy_service import PolicyService
    app.state.policy_service = PolicyService(data_dir=data_dir, vector_db_path=vdb_dir)
    logger.info(f"Policy: {app.state.policy_service.health()['policies_loaded']} policies loaded")

    # ── LangGraph Graph ──────────────────────────────────────────────────────
    from ..langgraph_system.graph import SupportGraph
    app.state.support_graph = SupportGraph(
        kb_service=app.state.kb_service,
        policy_service=app.state.policy_service,
        cache_service=app.state.cache_service,
    )
    logger.info("LangGraph: compiled ✅")

    # ── Observability ────────────────────────────────────────────────────────
    from ..observability.guardrails import GuardrailsEngine
    from ..observability.fallbacks import FallbackEngine
    from ..observability.evaluation import EvaluationEngine

    app.state.guardrails      = GuardrailsEngine()
    app.state.fallback_engine = FallbackEngine()
    app.state.evaluator       = EvaluationEngine()
    logger.info("Observability: guardrails + fallbacks + evaluation ✅")

    logger.info("=" * 60)
    logger.info("✅ System ready! All services initialized.")
    logger.info("=" * 60)

    yield  # ← app is running

    logger.info("🛑 Shutting down Customer Support System")


# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Support System — LangGraph Edition",
    description=(
        "AI-powered customer support with parallel LangGraph pipeline, "
        "Redis caching, ChromaDB RAG, and full observability."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS — allow frontend dev server + production ───────────────────────────
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
    "http://127.0.0.1:5500",   # VS Code Live Server
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    # Add production domain here:
    # "https://support.yourcompany.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Trace-ID", "X-Processing-Time"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# ─── Custom headers middleware ────────────────────────────────────────────────
@app.middleware("http")
async def add_trace_headers(request: Request, call_next):
    import uuid, time
    start = time.time()
    trace_id = str(uuid.uuid4())[:8]
    response = await call_next(request)
    response.headers["X-Trace-ID"] = trace_id
    response.headers["X-Processing-Time"] = f"{(time.time() - start)*1000:.1f}ms"
    return response

# ─── Global exception handler ────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )

# ─── Register routes ──────────────────────────────────────────────────────────
from .routes.health    import router as health_router
from .routes.chat      import router as chat_router
from .routes.analytics import analytics_router, sessions_router
from .routes.admin     import router as admin_router

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(analytics_router)
app.include_router(sessions_router)
app.include_router(admin_router)

# ─── Serve uploaded images ───────────────────────────────────────────────────
_uploads_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "uploads")
os.makedirs(_uploads_dir, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=_uploads_dir), name="uploads")

# ─── Image Upload Endpoint ────────────────────────────────────────────────────
@app.post("/chat/upload-image", tags=["chat"])
async def upload_complaint_image(file: UploadFile = File(...)):
    """
    Upload a product image with a complaint.
    Returns the stored URL + base64 for LLM vision context.
    """
    allowed = {"image/jpeg", "image/png", "image/webp", "image/gif"}
    if file.content_type not in allowed:
        return JSONResponse(status_code=400, content={"error": f"File type {file.content_type} not allowed. Use JPEG, PNG, or WebP."})

    ext      = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "jpg"
    filename = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(_uploads_dir, filename)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Read back as base64 for LLM vision context
    with open(save_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    return {
        "filename": filename,
        "url": f"/uploads/{filename}",
        "content_type": file.content_type,
        "base64_preview": f"data:{file.content_type};base64,{b64[:100]}...",  # truncated for API
        "full_base64": b64,  # used by agent for vision context
        "size_bytes": os.path.getsize(save_path),
    }

# ─── Serve React frontend build ───────────────────────────────────────────────
_react_build = os.path.join(os.path.dirname(__file__), "..", "..", "..", "frontend", "dist")
if os.path.exists(_react_build):
    app.mount("/app", StaticFiles(directory=_react_build, html=True), name="frontend")

# ─── Root endpoint ────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": "Customer Support System — LangGraph Edition",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/docs",
        "frontend": "/app",
        "health": "/health",
        "graph": "/health/graph",
    }


# ─── Dev entrypoint ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.support_system.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
