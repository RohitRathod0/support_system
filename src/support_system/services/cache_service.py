"""
Redis Cache Service — Microservice for complaint pattern detection and response caching.

Key features:
  - Cache full LangGraph responses for repeated complaint categories (1h TTL)
  - Track complaint counts per category for trending issue detection  
  - Distributed rate limiting per user
  - Cache invalidation API
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis

logger = logging.getLogger(__name__)


class CacheService:
    """Redis-backed caching for complaint responses and pattern tracking."""

    # TTLs (seconds)
    RESPONSE_TTL = 3600          # 1 hour for cached responses
    PATTERN_TTL  = 86400         # 24 hours for pattern counters
    SESSION_TTL  = 86400         # 24 hours for session data
    RATE_TTL     = 60            # 1-minute window for rate limiting

    # Thresholds
    TRENDING_THRESHOLD = 5       # 5+ same-category complaints → trending
    RATE_LIMIT = 20              # 20 requests per minute per user

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
    ):
        redis_url = os.getenv("REDIS_URL")
        try:
            if redis_url:
                self._r = redis.from_url(redis_url, decode_responses=True)
            else:
                self._r = redis.Redis(
                    host=host, port=port, db=db,
                    password=password, decode_responses=True,
                    socket_connect_timeout=2,
                )
            self._r.ping()
            self._available = True
            logger.info("✅ Redis connected")
        except Exception as e:
            logger.warning(f"⚠️  Redis unavailable ({e}), using in-memory fallback")
            self._available = False
            self._fallback: Dict[str, Any] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # RESPONSE CACHING
    # ─────────────────────────────────────────────────────────────────────────

    def make_cache_key(self, category: str, urgency: str, query_hash: str) -> str:
        """Build a deterministic cache key from complaint fingerprint."""
        raw = f"{category.lower()}:{urgency.lower()}:{query_hash}"
        return f"response:{hashlib.sha256(raw.encode()).hexdigest()[:16]}"

    def query_fingerprint(self, query: str) -> str:
        """Normalize query to a stable fingerprint (strip whitespace/case)."""
        normalized = " ".join(query.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()[:12]

    def get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Return cached response dict or None on miss."""
        try:
            if self._available:
                raw = self._r.get(cache_key)
                if raw:
                    logger.info(f"🎯 Cache HIT: {cache_key}")
                    data = json.loads(raw)
                    data["cache_hit"] = True
                    return data
            else:
                raw = self._fallback.get(cache_key)
                if raw:
                    data = json.loads(raw)
                    data["cache_hit"] = True
                    return data
        except Exception as e:
            logger.warning(f"Cache GET failed: {e}")
        return None

    def cache_response(
        self,
        cache_key: str,
        response: Dict[str, Any],
        ttl: int = RESPONSE_TTL,
    ) -> bool:
        """Store response in cache with TTL."""
        try:
            payload = json.dumps(response, default=str)
            if self._available:
                self._r.setex(cache_key, ttl, payload)
            else:
                self._fallback[cache_key] = payload
            logger.info(f"💾 Cached response: {cache_key} (TTL={ttl}s)")
            return True
        except Exception as e:
            logger.warning(f"Cache SET failed: {e}")
            return False

    def invalidate(self, cache_key: str) -> bool:
        """Delete a specific cache entry."""
        try:
            if self._available:
                self._r.delete(cache_key)
            else:
                self._fallback.pop(cache_key, None)
            return True
        except Exception:
            return False

    def invalidate_category(self, category: str) -> int:
        """Invalidate all cached responses for a category prefix."""
        count = 0
        try:
            if self._available:
                pattern = f"response:*{category.lower()}*"
                keys = list(self._r.scan_iter(pattern))
                if keys:
                    count = self._r.delete(*keys)
        except Exception as e:
            logger.warning(f"Bulk invalidation failed: {e}")
        return count

    # ─────────────────────────────────────────────────────────────────────────
    # COMPLAINT PATTERN TRACKING
    # ─────────────────────────────────────────────────────────────────────────

    def track_complaint(self, category: str, urgency: str = "medium") -> Dict[str, Any]:
        """
        Increment complaint counter for a category.
        Returns pattern info: count, is_trending, top_categories.
        """
        key = f"complaints:category:{category.lower()}"
        hourly_key = f"complaints:hourly:{datetime.now().strftime('%Y%m%d%H')}:{category.lower()}"
        try:
            if self._available:
                pipe = self._r.pipeline()
                pipe.incr(key)
                pipe.expire(key, self.PATTERN_TTL)
                pipe.incr(hourly_key)
                pipe.expire(hourly_key, 3600)
                results = pipe.execute()
                count = results[0]
            else:
                count = self._fallback.get(key, 0) + 1
                self._fallback[key] = count
        except Exception:
            count = 1

        return {
            "category": category,
            "count_24h": count,
            "is_trending": count >= self.TRENDING_THRESHOLD,
            "urgency": urgency,
        }

    def get_trending_categories(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Return top N most-complained-about categories in last 24h."""
        try:
            if not self._available:
                return []
            keys = list(self._r.scan_iter("complaints:category:*"))
            counts = []
            pipe = self._r.pipeline()
            for k in keys:
                pipe.get(k)
            values = pipe.execute()
            for k, v in zip(keys, values):
                category = k.split("complaints:category:")[-1]
                counts.append({"category": category, "count": int(v or 0)})
            counts.sort(key=lambda x: x["count"], reverse=True)
            return counts[:top_n]
        except Exception as e:
            logger.warning(f"Trending fetch failed: {e}")
            return []

    def get_hourly_stats(self) -> Dict[str, int]:
        """Return per-category complaint counts for the current hour."""
        try:
            if not self._available:
                return {}
            hour = datetime.now().strftime("%Y%m%d%H")
            keys = list(self._r.scan_iter(f"complaints:hourly:{hour}:*"))
            stats: Dict[str, int] = {}
            if keys:
                values = self._r.mget(keys)
                for k, v in zip(keys, values):
                    cat = k.split(f"complaints:hourly:{hour}:")[-1]
                    stats[cat] = int(v or 0)
            return stats
        except Exception:
            return {}

    # ─────────────────────────────────────────────────────────────────────────
    # RATE LIMITING
    # ─────────────────────────────────────────────────────────────────────────

    def check_rate_limit(self, user_id: str) -> Dict[str, Any]:
        """
        Sliding-window rate limit: 20 req/min per user.
        Returns {"allowed": bool, "remaining": int, "reset_in": int}.
        """
        key = f"ratelimit:{user_id}"
        try:
            if self._available:
                pipe = self._r.pipeline()
                pipe.incr(key)
                pipe.ttl(key)
                count, ttl = pipe.execute()
                if count == 1:
                    self._r.expire(key, self.RATE_TTL)
                    ttl = self.RATE_TTL
            else:
                count = self._fallback.get(key, 0) + 1
                self._fallback[key] = count
                ttl = self.RATE_TTL

            allowed = count <= self.RATE_LIMIT
            return {
                "allowed": allowed,
                "remaining": max(0, self.RATE_LIMIT - count),
                "reset_in": ttl,
                "count": count,
            }
        except Exception:
            return {"allowed": True, "remaining": self.RATE_LIMIT, "reset_in": 60, "count": 0}

    # ─────────────────────────────────────────────────────────────────────────
    # SESSION CACHING
    # ─────────────────────────────────────────────────────────────────────────

    def cache_session_context(self, session_id: str, context: Dict[str, Any]) -> None:
        """Cache lightweight session context for fast retrieval."""
        key = f"session:{session_id}"
        try:
            payload = json.dumps(context, default=str)
            if self._available:
                self._r.setex(key, self.SESSION_TTL, payload)
            else:
                self._fallback[key] = payload
        except Exception as e:
            logger.warning(f"Session cache failed: {e}")

    def get_session_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached session context."""
        key = f"session:{session_id}"
        try:
            if self._available:
                raw = self._r.get(key)
            else:
                raw = self._fallback.get(key)
            return json.loads(raw) if raw else None
        except Exception:
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # HEALTH
    # ─────────────────────────────────────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        """Return Redis health status."""
        try:
            if self._available:
                info = self._r.info("server")
                return {
                    "status": "connected",
                    "version": info.get("redis_version"),
                    "uptime_seconds": info.get("uptime_in_seconds"),
                }
        except Exception:
            pass
        return {"status": "fallback_memory"}
