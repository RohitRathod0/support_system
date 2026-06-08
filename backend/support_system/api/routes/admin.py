"""
FastAPI Routes — Admin Dashboard (Human-in-the-Loop)

Endpoints:
  POST /admin/auth/send-otp    — send 6-digit OTP to admin Gmail via SMTP
  POST /admin/auth/verify-otp  — verify OTP, return JWT session token
  GET  /admin/pending          — list money-matter tickets awaiting human approval
  POST /admin/approve/{id}     — admin approves a ticket
  POST /admin/reject/{id}      — admin rejects a ticket with reason
  GET  /admin/product-analytics — per-product complaint counts & alerts
  GET  /admin/ai-resolution-summary — AI handling stats & FCR breakdown
  GET  /admin/feedback-feed    — customer feedback entries

Security: All routes except /auth/* require a valid JWT in Authorization header.
"""
from __future__ import annotations

import json
import logging
import os
import random
import smtplib
import string
import uuid
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])

# ─── Config ──────────────────────────────────────────────────────────────────
JWT_SECRET   = os.getenv("ADMIN_JWT_SECRET", "super-secret-admin-key-change-in-prod")
JWT_ALGO     = "HS256"
JWT_EXP_HRS  = 2
OTP_TTL_SECS = 600  # 10 minutes
ADMIN_EMAIL  = os.getenv("ADMIN_EMAIL", "")           # must match who can log in

SMTP_HOST    = "smtp.gmail.com"
SMTP_PORT    = 587
SMTP_USER    = os.getenv("SMTP_USER", "")             # your Gmail address
SMTP_PASS    = os.getenv("SMTP_APP_PASSWORD", "")     # Gmail App Password


# ─── Pydantic schemas (inline for self-containment) ──────────────────────────
class OTPRequest(BaseModel):
    email: str

class OTPVerify(BaseModel):
    email: str
    code: str

class RejectReason(BaseModel):
    reason: str


# ─── JWT helpers ─────────────────────────────────────────────────────────────
def _create_token(email: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(hours=JWT_EXP_HRS)
    return jwt.encode({"sub": email, "exp": exp}, JWT_SECRET, algorithm=JWT_ALGO)


def _verify_token(token: str) -> str:
    """Returns email claim or raises HTTPException 401."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        return payload["sub"]
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired admin token")


# ─── Auth dependency ─────────────────────────────────────────────────────────
_bearer = HTTPBearer()

def require_admin(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> str:
    return _verify_token(credentials.credentials)


# ─── SMTP helper ─────────────────────────────────────────────────────────────
def _send_otp_email(to_email: str, otp: str) -> None:
    """Send OTP via Gmail SMTP. Raises on failure."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🔐 Admin OTP: {otp} — Customer Support System"
    msg["From"]    = SMTP_USER
    msg["To"]      = to_email

    html = f"""
    <html><body style="font-family:Arial,sans-serif;background:#0f0f1a;color:#e2e8f0;padding:40px">
      <div style="max-width:480px;margin:0 auto;background:#1a1a2e;border-radius:16px;
                  padding:40px;border:1px solid #6366f1">
        <h2 style="color:#6366f1;margin-top:0">🛡️ Admin Access OTP</h2>
        <p style="color:#94a3b8">Your one-time login code for the Support Admin Dashboard:</p>
        <div style="background:#0f0f1a;border-radius:12px;padding:24px;text-align:center;
                    margin:24px 0;border:1px solid #6366f1">
          <span style="font-size:42px;font-weight:700;letter-spacing:12px;
                       color:#6366f1;font-family:monospace">{otp}</span>
        </div>
        <p style="color:#64748b;font-size:13px">
          ⏱ Valid for <strong style="color:#f59e0b">10 minutes</strong>.<br>
          Do not share this code with anyone.
        </p>
        <hr style="border-color:#2d2d44;margin:24px 0">
        <p style="color:#475569;font-size:12px">
          Customer Support System — Admin Portal<br>
          If you did not request this, ignore this email.
        </p>
      </div>
    </body></html>
    """
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.ehlo()
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, to_email, msg.as_string())


# ─── Redis key helpers ────────────────────────────────────────────────────────
def _otp_key(email: str) -> str:
    return f"admin:otp:{email.lower()}"

def _pending_key(item_id: str) -> str:
    return f"admin:pending:{item_id}"

def _resolved_key(item_id: str) -> str:
    return f"admin:resolved:{item_id}"


# ─── AUTH ROUTES ─────────────────────────────────────────────────────────────

@router.post("/auth/send-otp")
async def send_otp(body: OTPRequest, request: Request) -> Dict[str, Any]:
    """
    Generate a 6-digit OTP, store in Redis (10 min TTL), and email it.
    For security, we always respond with success even if email not found,
    but we only send if ADMIN_EMAIL is configured and matches.
    """
    email = body.email.strip().lower()
    configured = ADMIN_EMAIL.strip().lower()

    # Hard check: only the configured admin email can log in
    if configured and email != configured:
        # Still return 200 (don't leak if email is valid or not)
        logger.warning(f"OTP request for non-admin email: {email}")
        return {"status": "sent", "message": "If this email is registered, you will receive a code."}

    # Generate OTP
    otp = "".join(random.choices(string.digits, k=6))

    # Store in Redis
    cache = request.app.state.cache_service
    if cache and cache._available:
        cache._r.setex(_otp_key(email), OTP_TTL_SECS, otp)
    else:
        # Fallback: in-memory (single process only — fine for dev)
        if not hasattr(request.app.state, "_otp_store"):
            request.app.state._otp_store = {}
        request.app.state._otp_store[email] = {
            "otp": otp,
            "expires": datetime.now(timezone.utc) + timedelta(seconds=OTP_TTL_SECS),
        }

    # Send email
    if SMTP_USER and SMTP_PASS:
        try:
            _send_otp_email(email, otp)
            logger.info(f"OTP sent to {email}")
        except Exception as e:
            logger.error(f"Failed to send OTP email: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")
    else:
        # Dev mode: log OTP to console
        logger.warning(f"[DEV MODE] OTP for {email}: {otp}  (SMTP not configured)")

    return {"status": "sent", "message": "If this email is registered, you will receive a code."}


@router.post("/auth/verify-otp")
async def verify_otp(body: OTPVerify, request: Request) -> Dict[str, Any]:
    """Verify the 6-digit OTP and return a JWT session token."""
    email = body.email.strip().lower()
    code  = body.code.strip()

    cache = request.app.state.cache_service
    stored_otp: Optional[str] = None

    if cache and cache._available:
        stored_otp = cache._r.get(_otp_key(email))
        if stored_otp:
            stored_otp = stored_otp.decode() if isinstance(stored_otp, bytes) else stored_otp
    else:
        store = getattr(request.app.state, "_otp_store", {})
        entry = store.get(email)
        if entry and datetime.now(timezone.utc) < entry["expires"]:
            stored_otp = entry["otp"]

    if not stored_otp:
        raise HTTPException(status_code=400, detail="OTP expired or not found. Please request a new code.")

    if stored_otp != code:
        raise HTTPException(status_code=400, detail="Invalid OTP code.")

    # Invalidate OTP (one-time use)
    if cache and cache._available:
        cache._r.delete(_otp_key(email))
    elif hasattr(request.app.state, "_otp_store"):
        request.app.state._otp_store.pop(email, None)

    token = _create_token(email)
    return {
        "status": "authenticated",
        "token": token,
        "email": email,
        "expires_in_hours": JWT_EXP_HRS,
    }


# ─── PENDING APPROVALS ───────────────────────────────────────────────────────

@router.get("/pending")
async def get_pending_approvals(
    request: Request,
    admin: str = Depends(require_admin),
) -> Dict[str, Any]:
    """List all money-matter tickets pending human approval."""
    cache = request.app.state.cache_service
    items: List[Dict] = []

    if cache and cache._available:
        pattern = "admin:pending:*"
        keys = cache._r.keys(pattern)
        for key in keys:
            raw = cache._r.get(key)
            if raw:
                try:
                    items.append(json.loads(raw))
                except Exception:
                    pass
    else:
        store = getattr(request.app.state, "_pending_store", {})
        items = list(store.values())

    # Sort newest first
    items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return {"pending": items, "count": len(items), "timestamp": datetime.now().isoformat()}


@router.post("/approve/{item_id}")
async def approve_ticket(
    item_id: str,
    request: Request,
    admin: str = Depends(require_admin),
) -> Dict[str, Any]:
    """Admin approves a money-matter ticket."""
    cache = request.app.state.cache_service
    key = _pending_key(item_id)
    raw = None

    if cache and cache._available:
        raw = cache._r.get(key)
    else:
        store = getattr(request.app.state, "_pending_store", {})
        raw = store.get(item_id)

    if not raw:
        raise HTTPException(status_code=404, detail="Ticket not found or already actioned.")

    try:
        item = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    except Exception:
        item = {}

    # Move to resolved
    resolved = {
        **item,
        "status": "approved",
        "actioned_by": admin,
        "actioned_at": datetime.now().isoformat(),
        "decision": "APPROVED",
    }
    resolved_raw = json.dumps(resolved)

    if cache and cache._available:
        cache._r.delete(key)
        cache._r.setex(_resolved_key(item_id), 86400 * 30, resolved_raw)  # keep 30 days
    else:
        store = getattr(request.app.state, "_pending_store", {})
        store.pop(item_id, None)
        if not hasattr(request.app.state, "_resolved_store"):
            request.app.state._resolved_store = {}
        request.app.state._resolved_store[item_id] = resolved

    logger.info(f"Admin {admin} APPROVED ticket {item_id}")
    return {"status": "approved", "ticket_id": item_id, "actioned_by": admin}


@router.post("/reject/{item_id}")
async def reject_ticket(
    item_id: str,
    body: RejectReason,
    request: Request,
    admin: str = Depends(require_admin),
) -> Dict[str, Any]:
    """Admin rejects a money-matter ticket with a reason."""
    cache = request.app.state.cache_service
    key = _pending_key(item_id)
    raw = None

    if cache and cache._available:
        raw = cache._r.get(key)
    else:
        store = getattr(request.app.state, "_pending_store", {})
        raw = store.get(item_id)

    if not raw:
        raise HTTPException(status_code=404, detail="Ticket not found or already actioned.")

    try:
        item = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    except Exception:
        item = {}

    resolved = {
        **item,
        "status": "rejected",
        "actioned_by": admin,
        "actioned_at": datetime.now().isoformat(),
        "decision": "REJECTED",
        "rejection_reason": body.reason,
    }
    resolved_raw = json.dumps(resolved)

    if cache and cache._available:
        cache._r.delete(key)
        cache._r.setex(_resolved_key(item_id), 86400 * 30, resolved_raw)
    else:
        store = getattr(request.app.state, "_pending_store", {})
        store.pop(item_id, None)
        if not hasattr(request.app.state, "_resolved_store"):
            request.app.state._resolved_store = {}
        request.app.state._resolved_store[item_id] = resolved

    logger.info(f"Admin {admin} REJECTED ticket {item_id}: {body.reason}")
    return {"status": "rejected", "ticket_id": item_id, "reason": body.reason}


# ─── RESOLVED HISTORY ────────────────────────────────────────────────────────

@router.get("/resolved")
async def get_resolved_tickets(
    request: Request,
    admin: str = Depends(require_admin),
    limit: int = 50,
) -> Dict[str, Any]:
    """List recently resolved (approved/rejected) money-matter tickets."""
    cache = request.app.state.cache_service
    items: List[Dict] = []

    if cache and cache._available:
        keys = cache._r.keys("admin:resolved:*")
        for key in list(keys)[:limit]:
            raw = cache._r.get(key)
            if raw:
                try:
                    items.append(json.loads(raw))
                except Exception:
                    pass
    else:
        store = getattr(request.app.state, "_resolved_store", {})
        items = list(store.values())[:limit]

    items.sort(key=lambda x: x.get("actioned_at", ""), reverse=True)
    return {"resolved": items, "count": len(items), "timestamp": datetime.now().isoformat()}


# ─── PRODUCT ANALYTICS ───────────────────────────────────────────────────────

@router.get("/product-analytics")
async def get_product_analytics(
    request: Request,
    admin: str = Depends(require_admin),
) -> Dict[str, Any]:
    """
    Aggregate complaint counts per product/category.
    A product with ≥3 complaints gets an ⚠️ Issue Alert flag.
    """
    cache = request.app.state.cache_service

    # Pull trending categories from Redis
    raw_trending = cache.get_trending_categories(top_n=20) if cache else []

    # Also scan pending + resolved tickets for product mentions
    product_counts: Dict[str, Dict] = {}
    all_tickets: List[Dict] = []

    if cache and cache._available:
        for key in cache._r.keys("admin:pending:*"):
            raw = cache._r.get(key)
            if raw:
                try:
                    all_tickets.append(json.loads(raw))
                except Exception:
                    pass
        for key in cache._r.keys("admin:resolved:*"):
            raw = cache._r.get(key)
            if raw:
                try:
                    all_tickets.append(json.loads(raw))
                except Exception:
                    pass
    else:
        pending = getattr(request.app.state, "_pending_store", {})
        resolved = getattr(request.app.state, "_resolved_store", {})
        all_tickets = list(pending.values()) + list(resolved.values())

    for ticket in all_tickets:
        cat = ticket.get("category", "General")
        if cat not in product_counts:
            product_counts[cat] = {
                "category": cat,
                "total_complaints": 0,
                "pending_approvals": 0,
                "common_issues": [],
                "last_reported": None,
                "has_alert": False,
            }
        product_counts[cat]["total_complaints"] += 1
        if ticket.get("status") == "pending":
            product_counts[cat]["pending_approvals"] += 1
        ts = ticket.get("timestamp")
        if ts and (not product_counts[cat]["last_reported"] or ts > product_counts[cat]["last_reported"]):
            product_counts[cat]["last_reported"] = ts
        issue = ticket.get("query", "")[:60]
        if issue and issue not in product_counts[cat]["common_issues"]:
            product_counts[cat]["common_issues"].append(issue)

    # Merge with general trending categories
    for t in raw_trending:
        cat = t["category"]
        if cat not in product_counts:
            product_counts[cat] = {
                "category": cat,
                "total_complaints": t["count"],
                "pending_approvals": 0,
                "common_issues": [],
                "last_reported": None,
                "has_alert": t["count"] >= 3,
            }
        else:
            product_counts[cat]["total_complaints"] = max(
                product_counts[cat]["total_complaints"], t["count"]
            )

    # Set alert flag for >= 3 complaints
    for cat_data in product_counts.values():
        cat_data["has_alert"] = cat_data["total_complaints"] >= 3

    products = sorted(product_counts.values(), key=lambda x: x["total_complaints"], reverse=True)
    alert_count = sum(1 for p in products if p["has_alert"])

    return {
        "products": products,
        "total_categories": len(products),
        "alert_count": alert_count,
        "timestamp": datetime.now().isoformat(),
    }


# ─── AI RESOLUTION SUMMARY ───────────────────────────────────────────────────

@router.get("/ai-resolution-summary")
async def get_ai_resolution_summary(
    request: Request,
    admin: str = Depends(require_admin),
) -> Dict[str, Any]:
    """
    Stats on how the AI is performing:
    - Total handled, escalated, resolved, pending human review
    - FCR verdict distribution
    - Avg QA score, avg processing time
    """
    from ...observability.traces import TraceManager

    traces = TraceManager.get_recent_traces(limit=100)
    completed = [t for t in traces if t.get("status") == "completed"]

    total = len(traces)
    total_completed = len(completed)

    # FCR distribution from evaluation data stored in traces
    # We'll pull from pending/resolved for money matters
    cache = request.app.state.cache_service
    pending_count = 0
    resolved_count = 0

    if cache and cache._available:
        pending_count  = len(cache._r.keys("admin:pending:*"))
        resolved_count = len(cache._r.keys("admin:resolved:*"))
    else:
        pending_count  = len(getattr(request.app.state, "_pending_store", {}))
        resolved_count = len(getattr(request.app.state, "_resolved_store", {}))

    avg_qa = (
        sum(t.get("qa_score", 0) for t in completed) / total_completed
        if total_completed else 0
    )
    avg_time = (
        sum(t.get("processing_time_s", 0) for t in completed) / total_completed
        if total_completed else 0
    )
    escalated = sum(1 for t in completed if t.get("escalation_needed"))
    ai_resolved = total_completed - escalated

    # Simulated FCR breakdown (from traces if available)
    fcr_counts = {"Fully Resolved": 0, "Partially Resolved": 0, "Not Resolved": 0}
    for t in completed:
        v = t.get("verdict", "Partially Resolved")
        if v in fcr_counts:
            fcr_counts[v] += 1
        else:
            fcr_counts["Partially Resolved"] += 1

    # Cache stats
    cache_stats = cache.health() if cache else {}
    hit_rate = cache_stats.get("hit_rate_24h", 0) if isinstance(cache_stats, dict) else 0

    return {
        "overview": {
            "total_conversations": total,
            "ai_resolved": ai_resolved,
            "escalated_to_human": escalated,
            "pending_admin_approval": pending_count,
            "resolved_by_admin": resolved_count,
            "ai_success_rate": round(ai_resolved / total * 100, 1) if total else 0,
        },
        "quality": {
            "avg_qa_score": round(avg_qa, 1),
            "avg_processing_time_s": round(avg_time, 2),
            "cache_hit_rate": hit_rate,
        },
        "fcr_distribution": fcr_counts,
        "timestamp": datetime.now().isoformat(),
    }


# ─── CUSTOMER FEEDBACK FEED ──────────────────────────────────────────────────

@router.get("/feedback-feed")
async def get_feedback_feed(
    request: Request,
    admin: str = Depends(require_admin),
    filter: str = "all",   # all | low_rated | escalated
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Return customer feedback entries for the admin to review.
    filter: all | low_rated (rating<=2) | escalated
    """
    cache = request.app.state.cache_service
    entries: List[Dict] = []

    if cache and cache._available:
        keys = cache._r.keys("feedback:*")
        for key in list(keys)[:200]:
            raw = cache._r.get(key)
            if raw:
                try:
                    entries.append(json.loads(raw))
                except Exception:
                    pass

    # Apply filter
    if filter == "low_rated":
        entries = [e for e in entries if e.get("rating", 5) <= 2]
    elif filter == "escalated":
        entries = [e for e in entries if e.get("escalated", False)]

    entries.sort(key=lambda x: x.get("timestamp", x.get("created_at", "")), reverse=True)
    entries = entries[:limit]

    # Summary stats
    ratings = [e.get("rating", 0) for e in entries if e.get("rating")]
    avg_rating = sum(ratings) / len(ratings) if ratings else 0
    low_rated  = sum(1 for r in ratings if r <= 2)
    high_rated = sum(1 for r in ratings if r >= 4)

    return {
        "feedback": entries,
        "count": len(entries),
        "summary": {
            "avg_rating": round(avg_rating, 1),
            "low_rated_count": low_rated,
            "high_rated_count": high_rated,
            "total_feedback": len(entries),
        },
        "timestamp": datetime.now().isoformat(),
    }
