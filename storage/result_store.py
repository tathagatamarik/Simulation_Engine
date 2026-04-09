"""
Result Store — Redis-backed with in-memory fallback.

In production: uses Redis for distributed access by multiple API workers.
In development/testing: uses a simple in-memory dict (no Redis required).

Configuration:
    Set REDIS_URL environment variable to enable Redis backend.
    If not set, falls back to in-memory store (single-process only).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# In-Memory Store (default, single-process)
# ---------------------------------------------------------------------------

_MEMORY_STORE: Dict[str, str] = {}


class ResultStore:
    """
    Async-compatible result store.

    Auto-selects backend:
      - REDIS_URL set → Redis
      - Otherwise    → in-memory dict
    """

    RESULT_TTL = 3600      # Seconds before a result expires (1 hour)
    _redis = None          # Lazy-initialized Redis client

    @classmethod
    async def _get_redis(cls):
        """Lazy initialize Redis client."""
        if cls._redis is None:
            redis_url = os.getenv("REDIS_URL")
            if redis_url:
                try:
                    import redis.asyncio as aioredis
                    cls._redis = aioredis.from_url(redis_url, decode_responses=True)
                except ImportError:
                    cls._redis = None
        return cls._redis

    @classmethod
    async def save(cls, run_id: str, data: Dict[str, Any]) -> None:
        """Persist simulation result data for a run_id."""
        serialized = json.dumps(data, default=str)
        r = await cls._get_redis()
        if r:
            await r.setex(run_id, cls.RESULT_TTL, serialized)
        else:
            _MEMORY_STORE[run_id] = serialized

    @classmethod
    async def load(cls, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve result data for a run_id. Returns None if not found."""
        r = await cls._get_redis()
        if r:
            raw = await r.get(run_id)
        else:
            raw = _MEMORY_STORE.get(run_id)

        if raw is None:
            return None
        return json.loads(raw)

    @classmethod
    async def delete(cls, run_id: str) -> None:
        """Remove a result from the store."""
        r = await cls._get_redis()
        if r:
            await r.delete(run_id)
        else:
            _MEMORY_STORE.pop(run_id, None)

    @classmethod
    async def exists(cls, run_id: str) -> bool:
        """Check if a run_id exists in the store."""
        return await cls.load(run_id) is not None
