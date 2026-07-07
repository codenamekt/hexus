# Forked from andreab67/hermes-memory-pgvector (BSD-3-Clause)
import threading
from typing import Dict, Optional


class CCRCache:
    """Thread-safe in-memory cache mapping memory_id (int) to compressed text (str)."""

    def __init__(self, maxsize: int = 1000):
        self._maxsize = maxsize
        self._cache: Dict[int, str] = {}
        self._lock = threading.Lock()

    def get(self, memory_id: int) -> Optional[str]:
        with self._lock:
            if memory_id in self._cache:
                val = self._cache.pop(memory_id)
                self._cache[memory_id] = val
                return val
            return None

    def set(self, memory_id: int, compressed: str) -> None:
        with self._lock:
            if memory_id in self._cache:
                self._cache.pop(memory_id)
            elif len(self._cache) >= self._maxsize:
                # Evict first key (Least Recently Used)
                first_key = next(iter(self._cache))
                self._cache.pop(first_key, None)
            self._cache[memory_id] = compressed

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
