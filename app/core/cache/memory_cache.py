"""In-memory caching with LRU eviction and TTL support"""
import time
from typing import Any, Optional, Dict
from collections import OrderedDict
from threading import RLock
from app.utils.logger import logger


class MemoryCache:
    """
    Thread-safe in-memory cache with LRU eviction and TTL.

    Features:
    - LRU (Least Recently Used) eviction policy
    - TTL (Time To Live) support per entry
    - Thread-safe operations
    - Performance metrics tracking
    - Automatic cleanup of expired entries
    """

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        Initialize memory cache.

        Args:
            max_size: Maximum number of entries
            ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = ttl
        self._cache: OrderedDict = OrderedDict()
        self._lock = RLock()

        # Metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

        logger.info(f"MemoryCache initialized (max_size={max_size}, ttl={ttl}s)")

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            value, expiry_time = self._cache[key]

            # Check if expired
            if expiry_time < time.time():
                del self._cache[key]
                self._expirations += 1
                self._misses += 1
                return None

            # Move to end (mark as recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional custom TTL for this entry
        """
        with self._lock:
            expiry_time = time.time() + (ttl if ttl is not None else self.default_ttl)

            # Update existing key
            if key in self._cache:
                self._cache[key] = (value, expiry_time)
                self._cache.move_to_end(key)
                return

            # Add new key
            self._cache[key] = (value, expiry_time)
            self._cache.move_to_end(key)

            # Evict oldest entry if cache is full
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)  # Remove oldest (first) item
                self._evictions += 1

    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")

    def get_stats(self) -> Dict:
        """Get cache performance statistics"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
                "expirations": self._expirations
            }

    def cleanup_expired(self) -> int:
        """
        Cleanup expired entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, (_, expiry) in self._cache.items()
                if expiry < current_time
            ]

            for key in expired_keys:
                del self._cache[key]

            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

            return len(expired_keys)