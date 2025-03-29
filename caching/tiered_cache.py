import time
from typing import List, Tuple, Dict, Any, Optional
from langchain.schema import Document
from .exact_cache import ExactMatchCache
from .vector_cache import VectorCache
from processing.similarity import calculate_similarity

class TieredCacheManager:
    """Multi-level cache manager with tiered storage strategy"""
    
    def __init__(self, vector_store, exact_ttl: int = 3600, frequent_ttl: int = 86400):
        # L1: Very fast, exact matches (in-memory)
        self.exact_cache = ExactMatchCache(ttl_seconds=exact_ttl)  # 1 hour TTL
        
        # L2: Semantic matches with high confidence (in-memory)
        self.frequent_cache: Dict[str, str] = {}
        self.frequent_timestamps: Dict[str, float] = {}
        self.frequent_ttl = frequent_ttl
        self.access_counts: Dict[str, int] = {}
        
        # L3: Vector store (persistent)
        self.vector_cache = VectorCache(vector_store)
        
        # Cache hit counters for analytics
        self.hits = {"L1": 0, "L2": 0, "L3": 0, "miss": 0}
    
    def check_cache(self, query: str) -> Tuple[List[Document], bool, str]:
        """Check all cache levels and return matching documents"""
        # L1: Check exact match cache first (fastest)
        exact_match = self.exact_cache.get(query)
        if exact_match:
            self.hits["L1"] += 1
            return [Document(page_content=exact_match, metadata={"query": query, "source": "L1_exact"})], True, "L1_exact"
        
        # L2: Check frequent queries cache
        l2_match = self._check_frequent_cache(query)
        if l2_match:
            self.hits["L2"] += 1
            return [Document(page_content=l2_match[0], metadata={"query": l2_match[1], "source": "L2_frequent"})], True, "L2_frequent"
        
        # L3: Vector search for semantic matches
        threshold = self._get_dynamic_threshold(query)
        vector_results = self.vector_cache.search(query, threshold=threshold)
        
        if vector_results:
            self.hits["L3"] += 1
            docs = [doc for doc, _ in vector_results]
            return docs, True, "L3_vector"
        
        self.hits["miss"] += 1
        return [], False, "miss"
    
    def add_to_cache(self, query: str, response: str) -> None:
        """Store in all cache levels"""
        # L1: Add to exact match - ensure normalized
        normalized_query = query.lower().strip()
        self.exact_cache.add(normalized_query, response)
        
        # L2: Update frequent cache tracking
        self._update_access_count(normalized_query)
        if self.access_counts.get(normalized_query, 0) >= 3:  # Threshold for "frequent"
            self._add_to_frequent_cache(normalized_query, response)
        
        # L3: Add to vector store with original query as metadata
        self.vector_cache.add(query, response, {"original_query": query})
    
    def _check_frequent_cache(self, query: str) -> Optional[Tuple[str, str]]:
        """Check if query matches anything in frequent cache. Returns (response, original_query)"""
        now = time.time()
        # Clean expired entries first
        expired = [k for k, t in self.frequent_timestamps.items() 
                  if now - t >= self.frequent_ttl]
        for k in expired:
            if k in self.frequent_cache:
                del self.frequent_cache[k]
                del self.frequent_timestamps[k]
        
        # Check for similar queries
        for cached_query, response in self.frequent_cache.items():
            similarity = calculate_similarity(query, cached_query)
            if similarity > 0.85:  # Slightly lower than before for more matches
                self._update_access_count(cached_query)
                return (response, cached_query)  # Return both response and original query
        return None
    
    def _update_access_count(self, query: str) -> None:
        """Track query frequency"""
        normalized_query = query.lower().strip()
        self.access_counts[normalized_query] = self.access_counts.get(normalized_query, 0) + 1
    
    def _add_to_frequent_cache(self, query: str, response: str) -> None:
        """Add to frequent cache with timestamp"""
        normalized_query = query.lower().strip()
        self.frequent_cache[normalized_query] = response
        self.frequent_timestamps[normalized_query] = time.time()
    
    def _get_dynamic_threshold(self, query: str) -> float:
        """Calculate dynamic similarity threshold based on query properties"""
        base = 0.65  # Slightly increased base threshold
        # Adjust based on query complexity
        words = len(query.split())
        if words <= 3:
            return base - 0.1  # More lenient for short queries
        elif words >= 10:
            return base + 0.1   # Stricter for long queries
        return base
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_hits = sum(v for k, v in self.hits.items() if k != "miss")
        total_queries = total_hits + self.hits["miss"]
        
        return {
            "hits": self.hits,
            "hit_ratio": total_hits / max(1, total_queries),
            "l1_size": len(self.exact_cache.cache),
            "l2_size": len(self.frequent_cache),
            # L3 size would require a call to the vector store
        }