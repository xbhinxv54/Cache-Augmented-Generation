import time
from typing import Dict, Any, List, Optional
from langchain.schema import Document

# Assuming correct relative paths
from caching.tiered_cache import TieredCacheManager
from processing.response_generator import ResponseGenerator
from processing.query_preprocessor import QueryPreprocessor
from monitoring.telemetry import TelemetrySystem
from monitoring.feedback import FeedbackSystem
# Assuming similarity is needed here directly, or handled within cache manager/feedback
from processing.similarity import calculate_similarity



class ImprovedCAGOrchestrator:
    """Main orchestrator for the cache-augmented generation system"""

    def __init__(self, cache_manager: TieredCacheManager, response_generator: ResponseGenerator, quality_threshold: float = 0.67, semantic_threshold: float = 0.62):
        self.response_generator = response_generator
        self.cache_manager=cache_manager
        self.telemetry = TelemetrySystem()
        self.feedback_system = FeedbackSystem(cache_manager)
        self.query_preprocessor = QueryPreprocessor()
        self.query_log = []
        self.quality_threshold = quality_threshold # Threshold for vector score
        self.semantic_threshold = semantic_threshold # NEW: Threshold for semantic check score
        print(f"[Orchestrator Init] quality_threshold={self.quality_threshold}, semantic_threshold={self.semantic_threshold}") # DEBUG Init values

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
                 final_source = f"cache_{initial_source}"
                 response_served_from_cache = True
                 print(f"[Orchestrator] Using high-quality CACHED response from {initial_source}")
            else:
                 # Hit found, but low quality. Log it, but proceed to generate.
                 print(f"[Orchestrator] LOW QUALITY HIT from {initial_source} (Score: {initial_score:.2f}). Will generate new response.")
                 final_source = "miss_low_quality_hit"
                 # Decide if relevant_docs should be used as context for generation
                 # Option: Keep docs: pass relevant_docs
                 # Option: Ignore docs: pass []
                 # Let's keep them for now, generator might use them
        else:
            # Initial lookup was a complete miss
            print(f"[Orchestrator] Cache MISS for: {original_query}")
            final_source = "miss"
            relevant_docs = [] # No context for generation on a clean miss

        # If no high-quality cached response was selected, generate a new one
        if not final_response:
            print(f"[Orchestrator] Generating NEW response...")
            final_response = self.response_generator.generate_new_response(original_query, relevant_docs) # Pass potential low-qual docs as context
            final_source = "generated" # Overwrite source if generated
            response_served_from_cache = False

            # Cache the newly generated response if applicable
            if self._should_cache_response(original_query, final_response):
                print(f"[Orchestrator] Caching newly generated response for: '{original_query}'")
                self.cache_manager.add_to_cache(original_query, final_response)
                # ... post-add check ...
            else:
                print(f"[Orchestrator] Decided NOT to cache newly generated response for: '{original_query}'")


        # Record telemetry using final determination
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        # Use the initial_source for cache breakdown, but response_served_from_cache for hit ratio
        self.telemetry.record_query(original_query, response_served_from_cache, initial_source if initial_hit else "miss", response_time_ms)

        # ... query logging ...
        self.query_log.append({
            "timestamp": time.time(),
            "query": original_query,
            "normalized_query": normalized_query,
            "is_hit_initial": initial_hit, # Whether cache returned *anything*
            "cache_source": initial_source if initial_hit else "miss", # L1/L2/L3/miss from initial check
            "score": initial_score,
            "response_source": final_source, # cache_L1/cache_L2/cache_L3/generated/miss_low_quality
            "response_time_ms": response_time_ms,
            "response": final_response[:100] + "..." if len(final_response)>100 else final_response
        })
        if len(self.query_log) > 100: self.query_log.pop(0)


        return final_response
    def provide_feedback(self, query: str, response: str, rating: int) -> None:
        """Record user feedback (e.g., 1-5 rating) and potentially update cache."""
        # Normalize query to potentially find related feedback entries
        normalized_query = self.query_preprocessor.normalize(query)
        self.feedback_system.record_feedback(normalized_query, response, rating, original_query=query)

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

    def _should_cache_response(self, query: str, response: str) -> bool:
        """Determine if a newly generated response should be cached."""
        # Basic heuristics:
        # Don't cache very short responses (might be errors like "I don't know")
        if len(response) < 5:
            # print(f"[Orchestrator] Not caching response - too short: {len(response)} chars") #DEBUG
            return False

        # Don't cache responses to potentially volatile or sensitive queries (example)
        if "password" in query.lower() or "credit card" in query.lower():
             # print(f"[Orchestrator] Not caching response - sensitive query detected.") #DEBUG
             return False

        # Don't cache generic failure messages
        if "cannot answer" in response.lower() or "don't have information" in response.lower():
             # print(f"[Orchestrator] Not caching response - generic failure message.") #DEBUG
             return False

        # Cache most other generated responses
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics about the system performance."""
        cache_stats = self.cache_manager.get_stats()
        telemetry_metrics = self.telemetry.get_metrics()
        feedback_metrics = self.feedback_system.get_response_quality_metrics()

        # Calculate actual response source distribution from the log for finer detail
        response_sources = {
            "cache_L1": 0, "cache_L2": 0, "cache_L3": 0, "generated": 0, "miss_low_quality": 0
        }
        total_resp = len(self.query_log)
        if total_resp > 0:
            for entry in self.query_log:
                src = entry.get("response_source", "unknown")
                if src.startswith("cache_L1"): response_sources["cache_L1"] += 1
                elif src.startswith("cache_L2"): response_sources["cache_L2"] += 1
                elif src.startswith("cache_L3"): response_sources["cache_L3"] += 1
                elif src == "generated": response_sources["generated"] += 1
                # Check for low quality hits that resulted in generation
                if entry.get("cache_source") == "miss_low_quality_hit":
                    response_sources["miss_low_quality"] += 1


        actual_cache_responses = response_sources["cache_L1"] + response_sources["cache_L2"] + response_sources["cache_L3"]
        actual_hit_ratio = actual_cache_responses / max(1, total_resp)

        return {
            "overall_performance": {
                "queries_processed": telemetry_metrics["query_count"],
                "avg_response_time_ms": telemetry_metrics["avg_response_time_ms"],
                "actual_response_hit_ratio": actual_hit_ratio, # Ratio of responses served *directly* from cache
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
            self.cache_manager.vector_cache.vector_store.persist()
        # Add any other cleanup needed