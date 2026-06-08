#!/usr/bin/env python
"""
run_server.py — Startup script for the Customer Support System v2 (LangGraph).
Replaces the old main.py (CrewAI).

Usage:
  python run_server.py              # production mode
  python run_server.py --reload     # dev mode with hot reload
  python run_server.py --port 9000  # custom port
"""
import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def check_env():
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        print("❌ MISTRAL_API_KEY not set in .env file")
        print("   Add: MISTRAL_API_KEY=your_key_here")
        sys.exit(1)
    print("✅ MISTRAL_API_KEY found")

    if not os.getenv("REDIS_URL") and not os.getenv("REDIS_HOST"):
        print("⚠️  Redis not configured — using in-memory fallback")
        print("   Optional: add REDIS_URL=redis://localhost:6379 to .env")

    if os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY"):
        print("✅ LangSmith tracing enabled")
    else:
        print("ℹ️  LangSmith not configured — custom tracing active")
        print("   Optional: add LANGSMITH_API_KEY=your_key to .env")


def main():
    parser = argparse.ArgumentParser(description="Customer Support System v2")
    parser.add_argument("--host",   default="0.0.0.0",   help="Host (default: 0.0.0.0)")
    parser.add_argument("--port",   default=8000, type=int, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true",  help="Enable hot reload (dev mode)")
    args = parser.parse_args()

    print("=" * 60)
    print("  🤖 Customer Support System v2 — LangGraph Edition")
    print("  ⚡ Parallel agents · Redis cache · ChromaDB RAG")
    print("=" * 60)

    check_env()

    print(f"\n🚀 Starting server on http://{args.host}:{args.port}")
    print(f"   📖 API docs:  http://localhost:{args.port}/docs")
    print(f"   🖥️  Frontend:  http://localhost:{args.port}/app")
    print(f"   ❤️  Health:    http://localhost:{args.port}/health")
    print(f"   📊 Traces:    http://localhost:{args.port}/analytics/traces")
    print()

    import uvicorn

    # Add src/ to path
    src_path = os.path.join(os.path.dirname(__file__), "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    uvicorn.run(
        "support_system.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
