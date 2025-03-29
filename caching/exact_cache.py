import time
from typing import Dict, Optional

class ExactMatchCache:
    """A cache for exact query matches with time-to-live functionality"""
    
    def __init__(self, ttl_seconds: int = 86400):  # Default 24 hours
        self.cache: Dict[str, str] = {}
        self.timestamps: Dict[str, float] = {}
        self.ttl = ttl_seconds
    
    def get(self, query: str) -> Optional[str]:
        """Get cached response for exact query match if not expired"""
        key = query.lower().strip()
        if key in self.cache and time.time() - self.timestamps[key] < self.ttl:
            return self.cache[key]
        return None
    
    def add(self, query: str, response: str) -> None:
        """Add response to exact match cache with current timestamp"""
        key = query.lower().strip()
        self.cache[key] = response
        self.timestamps[key] = time.time()
    
    def clear_expired(self) -> int:
        """Clear expired entries and return count of removed items"""
        now = time.time()
        expired_keys = [k for k, t in self.timestamps.items() if now - t >= self.ttl]
        
        for key in expired_keys:
            del self.cache[key]
            del self.timestamps[key]
        
        return len(expired_keys)