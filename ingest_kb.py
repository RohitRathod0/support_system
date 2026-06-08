"""
KB Ingestion Script — loads knowledge_base.json + company_policies.csv into ChromaDB.
Run this ONCE after setup, or after updating either data file.

Usage:
  python ingest_kb.py
"""
import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
load_dotenv()

BASE_DIR   = os.path.dirname(__file__)
DATA_DIR   = os.path.join(BASE_DIR, "data")
VECTOR_DIR = os.path.join(BASE_DIR, "vector_db")

def load_knowledge_base():
    """Convert knowledge_base.json into KB service documents."""
    kb_path = os.path.join(DATA_DIR, "knowledge_base.json")
    with open(kb_path, encoding="utf-8") as f:
        kb = json.load(f)

    docs = []

    # FAQs
    for key, faq in kb.get("faqs", {}).items():
        docs.append({
            "id": f"faq_{key}",
            "content": f"Q: {faq['question']}\nA: {faq['answer']}",
            "metadata": {
                "type": "faq",
                "category": faq.get("category", "general"),
                "tags": ", ".join(faq.get("tags", [])),
                "source": "knowledge_base.json",
            }
        })

    # Procedures
    for key, proc in kb.get("procedures", {}).items():
        steps = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(proc.get("steps", [])))
        docs.append({
            "id": f"proc_{key}",
            "content": f"PROCEDURE: {proc['title']}\n{steps}",
            "metadata": {
                "type": "procedure",
                "category": "general",
                "source": "knowledge_base.json",
            }
        })

    # Policy quick reference
    qr = kb.get("policy_quick_reference", {})
    if qr:
        content = "POLICY QUICK REFERENCE:\n"
        for section, rules in qr.items():
            content += f"\n{section.upper()}:\n"
            if isinstance(rules, dict):
                for k, v in rules.items():
                    content += f"  - {k}: {v}\n"
        docs.append({
            "id": "policy_quick_reference",
            "content": content,
            "metadata": {"type": "policy_reference", "category": "general", "source": "knowledge_base.json"}
        })

    return docs


def main():
    from support_system.services.kb_service import KnowledgeBaseService
    from support_system.services.policy_service import PolicyService

    print("=" * 55)
    print("  KB Ingestion — Customer Support System v2")
    print("=" * 55)

    # ── Knowledge Base ────────────────────────────────────────
    print("\n[1/2] Ingesting knowledge_base.json into ChromaDB...")
    kb_svc = KnowledgeBaseService(vector_db_path=VECTOR_DIR)
    docs = load_knowledge_base()
    count = kb_svc.ingest_documents(docs, source="knowledge_base")
    print(f"      Ingested {count} documents")
    health = kb_svc.health()
    print(f"      KB docs total: {health.get('knowledge_docs', '?')}")

    # ── Policies ──────────────────────────────────────────────
    print("\n[2/2] Re-indexing company_policies.csv into ChromaDB...")
    pol_svc = PolicyService(data_dir=DATA_DIR, vector_db_path=VECTOR_DIR)
    health = pol_svc.health()
    print(f"      Policies loaded: {health['policies_loaded']}")
    print(f"      Vector store: {health['vector_store']}")
    print(f"      Categories: {health['categories']}")

    print("\nDone! ChromaDB is now populated.")
    print(f"Vector DB path: {VECTOR_DIR}")
    print("\nRestart the server to pick up the new embeddings:")
    print("  python run_server.py")


if __name__ == "__main__":
    main()
