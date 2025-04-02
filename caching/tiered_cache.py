import time
from typing import Dict, Optional, Tuple, List, Any
from langchain.schema import Document
from .exact_cache import ExactMatchCache, normalize_key
from .vector_cache import VectorCache
# from processing.similarity import calculate_similarity
from sklearn.metrics.pairwise import cosine_similarity 



class TieredCacheManager:
    """Multi-level cache manager with tiered storage strategy"""

    
    def __init__(self, vector_store_instance, embedding_model, exact_ttl: int = 3600, frequent_ttl: int = 86400, l2_similarity_threshold: float = 0.80, l2_access_threshold: int = 2):
        # L1: Very fast, exact matches (in-memory)
        self.exact_cache = ExactMatchCache(ttl_seconds=exact_ttl)

        # L2: Semantic matches for frequently accessed items (in-memory)
        self.frequent_cache: Dict[str, str] = {}
        self.frequent_timestamps: Dict[str, float] = {}
        self.frequent_ttl = frequent_ttl
        self.access_counts: Dict[str, int] = {}
        self.l2_similarity_threshold = l2_similarity_threshold # Use the init value
        self.l2_access_threshold = l2_access_threshold         # Use the init value
        self.embedding_model = embedding_model                 

        # L3: Vector store (persistent)
        self.vector_cache = VectorCache(vector_store_instance)

        # Cache hit counters for analytics
        self.hits = {"L1": 0, "L2": 0, "L3": 0, "miss": 0}

    
    def check_cache(self, query: str) -> Tuple[List[Document], bool, str, Optional[float]]:
        """
        Check all cache levels and return matching documents, hit status, source, and score (if applicable).
        Score is typically relevant for L2/L3 hits. Updates access count.
        """
        normalized_query = normalize_key(query)

        # L1: Check exact match cache first (fastest)
        exact_match_response = self.exact_cache.get(normalized_query)
        if exact_match_response is not None:
            self.hits["L1"] += 1
            print(f"[TieredCache] L1 HIT: {normalized_query}")
            self._update_access_count(normalized_query)
            # Promote to L2 if threshold met
            if self.access_counts.get(normalized_query, 0) >= self.l2_access_threshold:
                 # Check if already promoted to avoid redundant prints/timestamp updates
                 if normalized_query not in self.frequent_cache:
                     self._add_to_frequent_cache(normalized_query, exact_match_response)
            
            return [Document(page_content=exact_match_response, metadata={"normalized_query": normalized_query, "source": "L1_exact"})], True, "L1_exact", 1.0

        # L2: Check frequent/semantic cache (uses embeddings now)
        l2_match = self._check_frequent_cache(normalized_query) 
        if l2_match:
            self.hits["L2"] += 1
            response, original_cached_query, similarity_score = l2_match
            #print(f"[TieredCache] L2 HIT: Input '{normalized_query}' matched '{original_cached_query}' with Cosine score {similarity_score:.4f}")
        
            return [Document(page_content=response, metadata={"matched_query": original_cached_query, "input_query": normalized_query, "source": "L2_frequent", "similarity": similarity_score})], True, "L2_frequent", similarity_score

        # L3: Vector search for semantic matches
        threshold = self._get_dynamic_threshold(normalized_query) # Use dynamic threshold
        vector_results = self.vector_cache.search(query, threshold=threshold) # Use original query for search

        print(f"[TieredCache] L3 Search Results for '{query}':") # Keep L3 debug print
        if vector_results:
            for i, (doc, score) in enumerate(vector_results):
                print(f"  - Hit {i+1}: Score={score:.4f}, OrigQuery='{doc.metadata.get('original_query', 'N/A')}'")
        else:
            print("  - No hits found meeting threshold.")

        if vector_results:
            self.hits["L3"] += 1

            self._update_access_count(normalized_query)
            vector_results.sort(key=lambda item: item[1], reverse=True)
            docs = [doc for doc, _ in vector_results]
            top_score = vector_results[0][1]
            for doc, score in vector_results:
                doc.metadata["source"] = "L3_vector"
                doc.metadata["score"] = score
            return docs, True, "L3_vector", top_score

        # Count the miss for the input query as well
        self._update_access_count(normalized_query)
        # Promote to L2 if threshold met *after* a miss led to generation (handled in orchestrator's add_to_cache call now)
        # -----------------------------------------
        #print(f"[TieredCache] MISS: {normalized_query}")
        self.hits["miss"] += 1
        return [], False, "miss", None

    def add_to_cache(self, query: str, response: str) -> None:
        """Store in appropriate cache levels. L1/L3 always, L2 if frequency threshold met."""
        normalized_query = normalize_key(query)

        # L1: Add to exact match cache
        self.exact_cache.add(normalized_query, response)

        # L2: Check access count (already updated by check_cache on miss/hit)
        # Promote if threshold met. This handles promotion after generation.
        if self.access_counts.get(normalized_query, 0) >= self.l2_access_threshold:
             # Check if already promoted to avoid redundant prints/timestamp updates
             if normalized_query not in self.frequent_cache:
                 self._add_to_frequent_cache(normalized_query, response)

        # L3: Add to vector store. Pass original query for metadata.
        self.vector_cache.add(query, response, {"original_query": query})

    
    def _check_frequent_cache(self, normalized_query: str) -> Optional[Tuple[str, str, float]]:
        """Check if normalized query semantically matches anything in frequent cache using embeddings."""
        self._clear_expired_frequent() # Clean expired L2 entries first

        best_match = None
        highest_similarity = -1.0 # Cosine similarity ranges -1 to 1

        # Check only if cache has items and embedding model is available
        if not self.frequent_cache or not hasattr(self, 'embedding_model') or self.embedding_model is None:
             return None

        try:
            # Embed the input query once
            query_embedding = self.embedding_model.embed_query(normalized_query)

            cached_keys = list(self.frequent_cache.keys())
            if not cached_keys:
                 return None 

            # Embed all cached keys
            # Consider batch embedding if embed_documents supports it well and L2 cache grows large
            cached_embeddings = self.embedding_model.embed_documents(cached_keys)

        
            # similarities will be a numpy array of shape (1, num_cached_keys)
            similarities = cosine_similarity([query_embedding], cached_embeddings)[0] # Extract the 1D array of scores

            # Find the best match above the threshold
            for i, cached_key in enumerate(cached_keys):
                similarity = similarities[i]
                print(f"[TieredCache] L2 Check: Input='{normalized_query}', Cached='{cached_key}', CosineSimilarity={similarity:.4f} (Threshold: {self.l2_similarity_threshold})") # DEBUG

                # Check if this similarity score is above threshold and better than the current best
                if similarity >= self.l2_similarity_threshold and similarity > highest_similarity:
                     highest_similarity = similarity
                     # Ensure the key still exists in the cache (could expire between getting keys and checking)
                     if cached_key in self.frequent_cache:
                         response = self.frequent_cache[cached_key]
                         best_match = (response, cached_key, similarity) # Store response, matched key, and score
                     else:
                          # This case is unlikely but handles potential race conditions/timing issues
                          print(f"[TieredCache] Warning: Key '{cached_key}' found during L2 similarity check but missing from frequent_cache dict.")

        except Exception as e:
            print(f"[TieredCache] Error during L2 embedding/similarity check: {e}")
            return None # Return None on error to avoid crashing

        # If a best match was found
        if best_match:
            matched_cached_query_key = best_match[1]
            # Update access count for the *matched* frequent query key to keep it relevant
            self._update_access_count(matched_cached_query_key)
            # Update timestamp for the matched item to keep it fresh in L2
            if matched_cached_query_key in self.frequent_timestamps:
                self.frequent_timestamps[matched_cached_query_key] = time.time()
            return best_match # Return tuple: (response, matched_key, score)

        return None # No suitable match found in L2


    def _update_access_count(self, normalized_query: str) -> None:
        """Track query frequency using normalized keys."""
        self.access_counts[normalized_query] = self.access_counts.get(normalized_query, 0) + 1
        print(f"[TieredCache] Access count for '{normalized_query}': {self.access_counts[normalized_query]}") # DEBUG

    def _add_to_frequent_cache(self, normalized_query: str, response: str) -> None:
        """Add to frequent cache (L2) with timestamp using normalized keys."""
        if normalized_query not in self.frequent_cache:
             print(f"[TieredCache] Promoting to L2: '{normalized_query}' (Access Count: {self.access_counts.get(normalized_query, 'N/A')})") # DEBUG
             self.frequent_cache[normalized_query] = response
             self.frequent_timestamps[normalized_query] = time.time()
        else:
             # If already in L2 (e.g. promoted via L1 hit, then generated again later?), just refresh timestamp
             print(f"[TieredCache] Refreshing L2 timestamp for existing key: '{normalized_query}'") # DEBUG
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
                
                if k in self.access_counts:
                    del self.access_counts[k]
                count += 1
        # if count > 0: print(f"[TieredCache] Cleared {count} expired L2 entries.") # DEBUG
        return count

    def _get_dynamic_threshold(self, query: str) -> float:
        """Calculate dynamic similarity threshold based on query properties"""
        # Base threshold for vector search relevance
        base_threshold = 0.55 # Keep this lower for initial L3 search

        # Example: Adjust based on query length
        word_count = len(query.split())
        if word_count <= 3:
            return max(0.50, base_threshold - 0.1)
        elif word_count >= 10:
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
            "hits_L1": self.hits["L1"],
            "hits_L2": self.hits["L2"],
            "hits_L3": self.hits["L3"],
            "misses": self.hits["miss"],
            "total_lookups": total_lookups,
            "overall_hit_ratio": total_hits / max(1, total_lookups),
            "l1_size": len(self.exact_cache.cache),
            "l2_size": len(self.frequent_cache),
            "l3_approx_size": l3_size,
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
            if normalized_query in self.access_counts:
                del self.access_counts[normalized_query]
            removed = True

        # Remove from L3
        if hasattr(self.vector_cache, 'remove_by_metadata'):
            # Assumes remove_by_metadata uses the 'original_query' passed here effectively
            if self.vector_cache.remove_by_metadata(query) > 0:
                 removed = True

        print(f"[TieredCache] REMOVE attempt for '{normalized_query}': Success={removed}") # DEBUG
        return removed