import time
from typing import Dict, Optional


def normalize_key(query: str) -> str:
    """Normalize query for use as a cache key."""
    return " ".join(query.lower().split())

class ExactMatchCache:
    """A cache for exact query matches with time-to-live functionality"""

    def __init__(self, ttl_seconds: int = 86400):  # Default 24 hours
        self.cache: Dict[str, str] = {}
        self.timestamps: Dict[str, float] = {}
        self.ttl = ttl_seconds

    def get(self, query: str) -> Optional[str]:
        """Get cached response for exact query match if not expired"""
        key = normalize_key(query) # Use consistent normalization
        if key in self.cache and time.time() - self.timestamps[key] < self.ttl:
            # print(f"[ExactCache] GET HIT: {key}") # DEBUG
            return self.cache[key]
        # print(f"[ExactCache] GET MISS: {key}") # DEBUG
        return None

    def add(self, query: str, response: str) -> None:
            """Add response to exact match cache with current timestamp"""
            key = normalize_key(query) # Use consistent normalization
            print(f"[ExactCache] Attempting ADD: Key='{key}'") # DEBUG
            self.cache[key] = response
            self.timestamps[key] = time.time()
            print(f"[ExactCache] ADD successful. Key='{key}'. Cache size now: {len(self.cache)}") # DEBUG

    def clear_expired(self) -> int:
        """Clear expired entries and return count of removed items"""
        now = time.time()
        # Ensure keys being checked exist in timestamps before comparing
        expired_keys = [k for k, t in self.timestamps.items() if now - t >= self.ttl]

        removed_count = 0
        for key in expired_keys:
            if key in self.cache:
                del self.cache[key]
                removed_count += 1
            # Always remove from timestamps if expired
            if key in self.timestamps:
                 del self.timestamps[key]

        # Recalculate count based on actual removals from cache dict
        return len(expired_keys) # Return count based on timestamp expiry check

    def remove(self, query: str) -> bool:
        """Remove a specific query from the cache."""
        key = normalize_key(query)
        removed = False
        if key in self.cache:
            del self.cache[key]
            removed = True
        if key in self.timestamps:
            del self.timestamps[key]
            removed = True # Ensure it returns True even if only timestamp existed
        return removed