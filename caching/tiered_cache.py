import time
from typing import List, Tuple, Dict, Any, Optional
from langchain.schema import Document

# Assuming these are in the correct relative paths based on the structure
from .exact_cache import ExactMatchCache, normalize_key # Import consistent normalization
from .vector_cache import VectorCache
from processing.similarity import calculate_similarity # Needs to be importable

class TieredCacheManager:
    """Multi-level cache manager with tiered storage strategy"""

    def __init__(self, vector_store_instance, exact_ttl: int = 3600, frequent_ttl: int = 86400, l2_similarity_threshold: float = 0.85, l2_access_threshold: int = 3):
        # L1: Very fast, exact matches (in-memory)
        self.exact_cache = ExactMatchCache(ttl_seconds=exact_ttl)  # 1 hour TTL

        # L2: Semantic matches for frequently accessed items (in-memory)
        # Stores mapping: normalized_query -> response
        self.frequent_cache: Dict[str, str] = {}
        self.frequent_timestamps: Dict[str, float] = {}
        self.frequent_ttl = frequent_ttl
        self.access_counts: Dict[str, int] = {}
        self.l2_similarity_threshold = l2_similarity_threshold
        self.l2_access_threshold = l2_access_threshold

        # L3: Vector store (persistent)
        self.vector_cache = VectorCache(vector_store_instance) # Pass the actual vector store instance

        # Cache hit counters for analytics
        self.hits = {"L1": 0, "L2": 0, "L3": 0, "miss": 0}

    def check_cache(self, query: str) -> Tuple[List[Document], bool, str, Optional[float]]:
        """
        Check all cache levels and return matching documents, hit status, source, and score (if applicable).
        Score is typically relevant for L3 hits.
        """
        # Use consistently normalized query for lookups
        normalized_query = normalize_key(query)

        # L1: Check exact match cache first (fastest)
        exact_match_response = self.exact_cache.get(normalized_query) # Use normalized query
        if exact_match_response is not None:
            self.hits["L1"] += 1
            print(f"[TieredCache] L1 HIT: {normalized_query}") # DEBUG
             # For L1, score is effectively 1.0 (perfect match)
            return [Document(page_content=exact_match_response, metadata={"normalized_query": normalized_query, "source": "L1_exact"})], True, "L1_exact", 1.0

        # L2: Check frequent/semantic cache (optional enhancement)
        l2_match = self._check_frequent_cache(normalized_query) # Use normalized query
        if l2_match:
            self.hits["L2"] += 1
            response, original_cached_query, similarity_score = l2_match
            # print(f"[TieredCache] L2 HIT: Input '{normalized_query}' matched '{original_cached_query}' with score {similarity_score:.2f}") # DEBUG
            # Return score from L2 similarity check
            return [Document(page_content=response, metadata={"matched_query": original_cached_query, "input_query": normalized_query, "source": "L2_frequent", "similarity": similarity_score})], True, "L2_frequent", similarity_score

        # L3: Vector search for semantic matches
    # L3: Vector search for semantic matches
        threshold = self._get_dynamic_threshold(normalized_query)
        vector_results = self.vector_cache.search(query, threshold=threshold) # Use original query

        # --- ADD THIS DEBUG BLOCK ---
        print(f"[TieredCache] L3 Search Results for '{query}':")
        if vector_results:
            for i, (doc, score) in enumerate(vector_results):
                print(f"  - Hit {i+1}: Score={score:.4f}, OrigQuery='{doc.metadata.get('original_query', 'N/A')}'")
        else:
            print("  - No hits found meeting threshold.")
        # --- END DEBUG BLOCK ---

        if vector_results:
            self.hits["L3"] += 1
            # Sort by score descending (most relevant first)
            vector_results.sort(key=lambda item: item[1], reverse=True)
            docs = [doc for doc, _ in vector_results]
            top_score = vector_results[0][1] # Score of the best match
            # Add source info to metadata for clarity downstream
            for doc, score in vector_results:
                doc.metadata["source"] = "L3_vector"
                doc.metadata["score"] = score
            # print(f"[TieredCache] L3 HIT: {normalized_query} - Found {len(docs)} results, Top Score: {top_score:.2f}") # DEBUG
            return docs, True, "L3_vector", top_score

        # print(f"[TieredCache] MISS: {normalized_query}") # DEBUG
        self.hits["miss"] += 1
        return [], False, "miss", None

    def add_to_cache(self, query: str, response: str) -> None:
        """Store in appropriate cache levels. Uses normalized query for L1/L2 keys."""
        normalized_query = normalize_key(query)

        # L1: Add to exact match cache
        self.exact_cache.add(normalized_query, response)

        # L2: Update access count using normalized query. Promote if threshold met.
        self._update_access_count(normalized_query)
        if self.access_counts.get(normalized_query, 0) >= self.l2_access_threshold:
            self._add_to_frequent_cache(normalized_query, response)

        # L3: Add to vector store. Use original query in metadata for better context retrieval later.
        # Pass the *original* non-normalized query here for metadata.
        self.vector_cache.add(query, response, {"original_query": query})

    def _check_frequent_cache(self, normalized_query: str) -> Optional[Tuple[str, str, float]]:
        """Check if normalized query semantically matches anything in frequent cache."""
        self._clear_expired_frequent() # Clean expired L2 entries first

        best_match = None
        highest_similarity = -1.0

        # Check for similar queries in the L2 cache
        for cached_normalized_query, response in self.frequent_cache.items():
            similarity = calculate_similarity(normalized_query, cached_normalized_query)
            print(f"[TieredCache] L2 Check: Input='{normalized_query}', Cached='{cached_normalized_query}', Similarity={similarity:.4f} (Threshold: {self.l2_similarity_threshold})") # DEBUG

            if similarity > self.l2_similarity_threshold and similarity > highest_similarity:
                 highest_similarity = similarity
                 best_match = (response, cached_normalized_query, similarity) # Return response, the query it matched, and score

        if best_match:
            # Update access count for the *matched* frequent query
            self._update_access_count(best_match[1])
            # Update timestamp for the matched item to keep it fresh
            self.frequent_timestamps[best_match[1]] = time.time()
            return best_match

        return None

    def _update_access_count(self, normalized_query: str) -> None:
        """Track query frequency using normalized keys."""
        self.access_counts[normalized_query] = self.access_counts.get(normalized_query, 0) + 1
        # print(f"[TieredCache] Access count for '{normalized_query}': {self.access_counts[normalized_query]}") # DEBUG

    def _add_to_frequent_cache(self, normalized_query: str, response: str) -> None:
        """Add to frequent cache (L2) with timestamp using normalized keys."""
        if normalized_query not in self.frequent_cache:
             # print(f"[TieredCache] Promoting to L2: {normalized_query}") # DEBUG
             self.frequent_cache[normalized_query] = response
             self.frequent_timestamps[normalized_query] = time.time()
        else:
             # Optionally update the response if it's already there? Or just refresh timestamp?
             # Let's just refresh the timestamp to keep it alive
             self.frequent_timestamps[normalized_query] = time.time()


    def _clear_expired_frequent(self) -> int:
        """Clear expired L2 entries."""
        now = time.time()
        expired_keys = [k for k, t in self.frequent_timestamps.items()
                       if now - t >= self.frequent_ttl]
        count = 0
        for k in expired_keys:
            if k in self.frequent_cache:
                del self.frequent_cache[k]
                del self.frequent_timestamps[k]
                # Also remove from access counts if desired, or let it decay naturally
                if k in self.access_counts:
                    del self.access_counts[k]
                count += 1
        # if count > 0: print(f"[TieredCache] Cleared {count} expired L2 entries.") # DEBUG
        return count

    def _get_dynamic_threshold(self, query: str) -> float:
        """Calculate dynamic similarity threshold based on query properties"""
        # Base threshold for vector search relevance
        base_threshold = 0.55

        # Example: Adjust based on query length (longer queries might need higher precision)
        word_count = len(query.split())
        if word_count <= 3:
            # Slightly more lenient for very short, potentially ambiguous queries
            return max(0.50, base_threshold - 0.1)
        elif word_count >= 10:
             # Slightly stricter for longer, more specific queries
             return min(0.75, base_threshold + 0.05)

        return base_threshold

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_hits = sum(v for k, v in self.hits.items() if k != "miss")
        total_lookups = total_hits + self.hits["miss"]

        l3_size = -1 # Indicate unknown or expensive to calculate
        if hasattr(self.vector_cache.vector_store, '_collection'):
             try:
                 l3_size = self.vector_cache.vector_store._collection.count()
             except Exception:
                 pass # Ignore if count fails

        return {
            "hits_ L1": self.hits["L1"],
            "hits_L2": self.hits["L2"],
            "hits_L3": self.hits["L3"],
            "misses": self.hits["miss"],
            "total_lookups": total_lookups,
            "overall_hit_ratio": total_hits / max(1, total_lookups),
            "l1_size": len(self.exact_cache.cache),
            "l2_size": len(self.frequent_cache),
            "l3_approx_size": l3_size, # Provide L3 size if easily available
        }

    def remove_from_cache(self, query: str) -> bool:
        """Remove a query and its associated response from all relevant cache levels."""
        normalized_query = normalize_key(query)
        removed = False

        # Remove from L1
        if self.exact_cache.remove(normalized_query):
            removed = True

        # Remove from L2
        if normalized_query in self.frequent_cache:
            del self.frequent_cache[normalized_query]
            if normalized_query in self.frequent_timestamps:
                del self.frequent_timestamps[normalized_query]
            # Optionally remove from access counts too
            if normalized_query in self.access_counts:
                del self.access_counts[normalized_query]
            removed = True

        # Remove from L3 (using original query stored in metadata)
        # We need the original query that *generated* the response we want to remove.
        # This is tricky if we only have the potentially *different* query used for lookup.
        # Assuming the 'query' passed here is the one whose response should be removed.
        # We rely on VectorCache's ability to remove based on 'original_query' metadata.
        if self.vector_cache.remove_by_metadata(query) > 0: # Use original query
             removed = True

        # print(f"[TieredCache] REMOVE attempt for '{normalized_query}': Success={removed}") # DEBUG
        return removed