import time
from typing import Dict, Any, List, Optional
from langchain.schema import Document

# Assuming correct relative paths
from caching.tiered_cache import TieredCacheManager
from processing.response_generator import ResponseGenerator
from processing.query_preprocessor import QueryPreprocessor
from monitoring.telemetry import TelemetrySystem
from monitoring.feedback import FeedbackSystem
# Keep calculate_similarity only if used elsewhere, removed from L3 check
# from processing.similarity import calculate_similarity

# --- UPDATED CLASS ---
class ImprovedCAGOrchestrator:
    """Main orchestrator for the cache-augmented generation system"""

    # --- UPDATED __init__ ---
    def __init__(self, cache_manager: TieredCacheManager, response_generator: ResponseGenerator, quality_threshold: float = 0.67, l2_quality_threshold: float = 0.80): # Default L2 threshold matches TieredCacheManager
        self.cache_manager = cache_manager
        self.response_generator = response_generator
        self.telemetry = TelemetrySystem()
        self.feedback_system = FeedbackSystem(cache_manager)
        self.query_preprocessor = QueryPreprocessor()
        self.query_log = []
        self.quality_threshold = quality_threshold       # Threshold for L3 vector score
        self.l2_quality_threshold = l2_quality_threshold # Threshold for L2 semantic score
        print(f"[Orchestrator Init] L3 quality_threshold={self.quality_threshold}, L2 quality_threshold={self.l2_quality_threshold}") # DEBUG Init values

    # --- UPDATED METHOD (Refined Logic) ---
    def ask(self, query: str) -> str:
        start_time = time.time()
        original_query = query
        normalized_query = self.query_preprocessor.normalize(original_query)

        # Initial Cache Check
        relevant_docs, initial_hit, initial_source, initial_score = self.cache_manager.check_cache(original_query)

        final_response = ""
        final_source = "miss" # Source for telemetry/logging
        response_served_from_cache = False # Flag for telemetry

        if initial_hit:
            # Check if the found hit is usable
            is_high_quality = self._is_high_quality_match(initial_source, initial_score, relevant_docs, normalized_query)
            if is_high_quality:
                 # Use the cached response directly
                 final_response = self.response_generator.retrieve_cached_response(relevant_docs)
                 final_source = f"cache_{initial_source}" # e.g., cache_L1_exact, cache_L2_frequent
                 response_served_from_cache = True
                 print(f"[Orchestrator] Using high-quality CACHED response from {initial_source}")
            else:
                 # Hit found, but low quality. Log it, but proceed to generate.
                 print(f"[Orchestrator] LOW QUALITY HIT from {initial_source} (Score: {initial_score:.4f}). Will generate new response.")
                 final_source = "miss_low_quality_hit"
                 # Pass relevant_docs as context for generation
        else:
            # Initial lookup was a complete miss
            print(f"[Orchestrator] Cache MISS for: {original_query}")
            final_source = "miss"
            relevant_docs = [] # No context for generation on a clean miss

        # If no high-quality cached response was selected, generate a new one
        if not final_response:
            print(f"[Orchestrator] Generating NEW response...")
            # Pass potential low-quality docs as context if initial_hit was True but !is_high_quality
            # Otherwise pass empty list if it was a clean miss
            context_docs = relevant_docs if initial_hit else []
            final_response = self.response_generator.generate_new_response(original_query, context_docs)
            final_source = "generated" # Overwrite source if generated
            response_served_from_cache = False

            # Cache the newly generated response if applicable
            if self._should_cache_response(original_query, final_response):
                print(f"[Orchestrator] Caching newly generated response for: '{original_query}'")
                # Let add_to_cache handle L1/L3 and L2 promotion check
                self.cache_manager.add_to_cache(original_query, final_response)
                # Optional: Add post-add check logs here if needed
            else:
                print(f"[Orchestrator] Decided NOT to cache newly generated response for: '{original_query}'")


        # Record telemetry using final determination
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        # Log initial source for analysis, but use response_served_from_cache for hit ratio
        self.telemetry.record_query(original_query, response_served_from_cache, initial_source if initial_hit else "miss", response_time_ms)

        # Add to internal query log
        self.query_log.append({
            "timestamp": time.time(),
            "query": original_query,
            "normalized_query": normalized_query,
            "is_hit_initial": initial_hit,
            "cache_source": initial_source if initial_hit else "miss",
            "score": initial_score,
            "response_source": final_source, # Where the final response came from
            "response_time_ms": response_time_ms,
            "response": final_response[:100] + "..." if len(final_response)>100 else final_response
        })
        if len(self.query_log) > 100: self.query_log.pop(0)

        return final_response

    def provide_feedback(self, query: str, response: str, rating: int) -> None:
        """Record user feedback (e.g., 1-5 rating) and potentially update cache."""
        normalized_query = self.query_preprocessor.normalize(query)
        self.feedback_system.record_feedback(normalized_query, response, rating, original_query=query)


    # --- UPDATED METHOD ---
    def _is_high_quality_match(self, source: str, score: Optional[float], docs: List[Document], normalized_query: str) -> bool:
        """Determine if the cached match is high quality enough to use directly."""

        # Handle L1_exact FIRST and return immediately
        if source == "L1_exact":
            print("[Orchestrator] Quality Check: L1 source is always high quality. Returning True.") # DEBUG L1
            return True

        # If not L1, proceed with other checks
        if not docs:
            print("[Orchestrator] Quality Check: No documents provided (and not L1). Returning False.") # DEBUG
            return False

        # L2 frequent/semantic matches
        if source == "L2_frequent":
            # Use the specific L2 quality threshold defined in the orchestrator
            print(f"[Orchestrator] L2 Quality Check: Score={score:.4f} (Threshold: {self.l2_quality_threshold})") # DEBUG L2
            is_high_quality = score is not None and score >= self.l2_quality_threshold
            print(f"[Orchestrator] L2 Quality Check Result: {is_high_quality}") # DEBUG L2
            return is_high_quality

        # L3 vector matches - REMOVED SEMANTIC CHECK
        if source == "L3_vector":
            # Use self.quality_threshold for vector score check
            print(f"[Orchestrator] L3 Quality Check Input: InputQuery='{normalized_query}', VectorScore={score:.4f} (Threshold: {self.quality_threshold})")

            vector_check_passed = score is not None and score >= self.quality_threshold

            # Decision now relies ONLY on the vector score check
            if vector_check_passed:
                 print(f"[Orchestrator] L3 Quality Check Result (Vector Only Passed): True")
                 return True
            else:
                 print(f"[Orchestrator] L3 Quality Check Result (Vector Failed): False")
                 return False

        # Default to False if source is unknown or other conditions not met
        print(f"[Orchestrator] Quality Check: Unknown source '{source}' or fallthrough. Returning False.") # DEBUG
        return False

    # --- Existing methods (_should_cache_response, get_stats, cleanup) remain the same ---
    # Ensure _should_cache_response has a reasonable length check (e.g., >= 5)
    def _should_cache_response(self, query: str, response: str) -> bool:
        """Determine if a newly generated response should be cached."""
        min_length = 5 # Allow short answers like "Paris."
        if len(response) < min_length:
            print(f"[Orchestrator] Not caching response - too short ({len(response)} < {min_length}): '{response}'") #DEBUG
            return False

        if "password" in query.lower() or "credit card" in query.lower():
             print(f"[Orchestrator] Not caching response - sensitive query detected.") #DEBUG
             return False

        # Adjusted check for generic failures
        generic_failures = ["cannot answer", "don't have information", "don't know", "unable to provide"]
        if any(phrase in response.lower() for phrase in generic_failures):
             print(f"[Orchestrator] Not caching response - generic failure message.") #DEBUG
             return False

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics about the system performance."""
        cache_stats = self.cache_manager.get_stats()
        telemetry_metrics = self.telemetry.get_metrics()
        feedback_metrics = self.feedback_system.get_response_quality_metrics()

        # Calculate actual response source distribution from the log
        response_sources = {
            "cache_L1": 0, "cache_L2": 0, "cache_L3": 0, "generated": 0, "miss_low_quality": 0
        }
        total_resp = len(self.query_log)
        if total_resp > 0:
            for entry in self.query_log:
                src = entry.get("response_source", "unknown")
                if src == "cache_L1_exact": response_sources["cache_L1"] += 1
                elif src == "cache_L2_frequent": response_sources["cache_L2"] += 1
                elif src == "cache_L3_vector": response_sources["cache_L3"] += 1
                elif src == "generated": response_sources["generated"] += 1
                elif src == "miss_low_quality_hit": response_sources["miss_low_quality"] += 1


        actual_cache_responses = response_sources["cache_L1"] + response_sources["cache_L2"] + response_sources["cache_L3"]
        actual_hit_ratio = actual_cache_responses / max(1, total_resp)

        return {
            "overall_performance": {
                "queries_processed": telemetry_metrics["query_count"],
                "avg_response_time_ms": telemetry_metrics["avg_response_time_ms"], # Correct key
                "actual_response_hit_ratio": actual_hit_ratio,
            },
            "cache_performance": cache_stats, # Raw lookup stats from TieredCacheManager
            "response_source_distribution": response_sources, # Based on orchestrator log
            "telemetry_details": telemetry_metrics,
            "feedback_summary": feedback_metrics,
        }

    def cleanup(self):
        """Perform cleanup tasks, e.g., persisting vector store."""
        if hasattr(self.cache_manager.vector_cache.vector_store, 'persist'):
            print("[Orchestrator] Persisting vector store...")
            try:
                self.cache_manager.vector_cache.vector_store.persist()
                print("[Orchestrator] Vector store persisted.")
            except Exception as e:
                print(f"[Orchestrator] Error persisting vector store: {e}")
        # Add any other cleanup needed