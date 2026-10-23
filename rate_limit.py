"""
Rate limits for /ask, the endpoint that costs something (a Gemini call).

Two limits, because they guard against different things:

    per client  RATE_LIMIT_PER_MINUTE (5) requests in any 60 s, so one caller
                can't take the whole quota by themselves
    global      ASK_LIMIT_PER_DAY (20) requests in any 24 h, whoever is calling.
                A per-client limit alone is beaten by switching IP addresses;
                this one caps the spend, and keeps a public demo inside the
                Gemini free tier's daily quota.

The counters live in memory. That is exact with a single instance (Cloud Run
max-instances=1); with several, each would count on its own, and the counters
would have to move to a shared store such as Postgres or Redis.
"""

import os
import threading
import time
from collections import defaultdict, deque

PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "5"))
PER_DAY = int(os.environ.get("ASK_LIMIT_PER_DAY", "20"))


class RateLimiter:
    """Sliding-window counters: a request is allowed if fewer than `limit` were allowed in the window."""

    def __init__(self, per_minute: int = PER_MINUTE, per_day: int = PER_DAY, clock=time.monotonic):
        self.per_minute = per_minute
        self.per_day = per_day
        self.clock = clock
        self._by_client: dict[str, deque] = defaultdict(deque)
        self._all: deque = deque()
        self._lock = threading.Lock()

    @staticmethod
    def _expire(window: deque, now: float, seconds: float) -> None:
        while window and window[0] <= now - seconds:
            window.popleft()

    def check(self, client: str) -> tuple[bool, int]:
        """(allowed, seconds to wait if not). An allowed request is counted."""
        with self._lock:
            now = self.clock()
            mine = self._by_client[client]
            self._expire(mine, now, 60)
            self._expire(self._all, now, 86_400)
            if len(self._all) >= self.per_day:
                return False, int(self._all[0] + 86_400 - now) + 1
            if len(mine) >= self.per_minute:
                return False, int(mine[0] + 60 - now) + 1
            mine.append(now)
            self._all.append(now)
            if len(self._by_client) > 10_000:  # forget idle clients so the dict can't grow without bound
                for key in [k for k, v in self._by_client.items() if not v]:
                    del self._by_client[key]
            return True, 0
