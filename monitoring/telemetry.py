import time
from typing import List, Dict, Any
from collections import deque
import statistics # For more robust avg calculation if needed

class TelemetrySystem:
    """System for tracking and analyzing performance metrics"""

    def __init__(self, max_history: int = 1000, response_time_window: int = 100):
        self.metrics = {
            "query_count": 0,
            # Raw cache lookup stats
            "cache_lookup_hits": 0,
            "cache_lookup_misses": 0,
            "cache_lookup_hit_ratio": 0.0,
            # Stats based on whether response was served from cache
            "responses_from_cache": 0,
            "responses_generated": 0,
            "actual_response_hit_ratio": 0.0,
            # Performance
            "response_times_ms": deque(maxlen=response_time_window), # Store recent times
            "avg_response_time_ms": 0.0,
            # Cache source breakdown (based on initial lookup)
            "cache_source_distribution": {"L1_exact": 0, "L2_frequent": 0, "L3_vector": 0, "miss": 0, "miss_low_quality_hit": 0}
        }
        self.max_history = max_history
        self.response_time_window = response_time_window
        # Keep detailed history for potential analysis/debugging
        self.history = deque(maxlen=self.max_history)


    def record_query(self, query: str, response_served_from_cache: bool, cache_source: str, response_time_ms: float):
        """Record telemetry for a completed query."""

        # --- Update Counters ---
        self.metrics["query_count"] += 1

        # Track if the initial cache lookup was a hit or miss
        is_lookup_hit = cache_source not in ["miss", "miss_low_quality_hit"]
        if is_lookup_hit:
            self.metrics["cache_lookup_hits"] += 1
        else:
            self.metrics["cache_lookup_misses"] += 1

        # Track if the final response came from cache or was generated
        if response_served_from_cache:
             self.metrics["responses_from_cache"] += 1
        else:
             self.metrics["responses_generated"] += 1

        # Track the source reported by the cache manager lookup
        if cache_source in self.metrics["cache_source_distribution"]:
            self.metrics["cache_source_distribution"][cache_source] += 1
        else:
            # Handle unexpected source strings gracefully
            self.metrics["cache_source_distribution"]["unknown"] = self.metrics["cache_source_distribution"].get("unknown", 0) + 1


        # --- Update Performance Metrics ---
        self.metrics["response_times_ms"].append(response_time_ms)


        # --- Add to History ---
        self.history.append({
            "timestamp": time.time(),
            "query": query,
            "response_from_cache": response_served_from_cache,
            "cache_lookup_source": cache_source, # Source from initial check
            "response_time_ms": response_time_ms
        })

        # --- Update Derived Metrics ---
        self._update_derived_metrics()


    def _update_derived_metrics(self):
        """Update metrics calculated from raw counters."""
        total_queries = self.metrics["query_count"]
        if total_queries > 0:
            # Raw lookup hit ratio
            self.metrics["cache_lookup_hit_ratio"] = self.metrics["cache_lookup_hits"] / total_queries
            # Actual hit ratio based on served responses
            self.metrics["actual_response_hit_ratio"] = self.metrics["responses_from_cache"] / total_queries

        # Average response time
        times = list(self.metrics["response_times_ms"]) # Convert deque to list for calculation
        if times:
            # Use statistics.mean for robustness against potential non-numeric entries if any errors occur upstream
            try:
                self.metrics["avg_response_time_ms"] = statistics.mean(times)
            except TypeError:
                 # Fallback if list contains non-numeric data somehow
                 self.metrics["avg_response_time_ms"] = sum(t for t in times if isinstance(t, (int, float))) / max(1, len(times))

        else:
            self.metrics["avg_response_time_ms"] = 0.0


    def get_metrics(self) -> Dict[str, Any]:
        """Get a copy of the current telemetry metrics."""
        # Return a copy to prevent external modification
        metrics_copy = self.metrics.copy()
        # Convert deque to list for easier consumption if needed by consumers
        metrics_copy["response_times_ms"] = list(metrics_copy["response_times_ms"])
        return metrics_copy

    def get_recent_history(self, count: int = 20) -> List[Dict[str, Any]]:
        """Get the N most recent query history entries."""
        return list(self.history)[-count:]

    def generate_report(self) -> str:
        """Generate a simple text report of key metrics."""
        m = self.get_metrics() # Get current calculated metrics
        report = f"""
        === Telemetry Report ===
        Total Queries Processed: {m['query_count']}

        --- Response Source ---
        Responses Served from Cache: {m['responses_from_cache']} ({m['actual_response_hit_ratio']:.2%})
        Responses Generated Anew:    {m['responses_generated']}

        --- Cache Lookup Performance ---
        Cache Lookups Hit:  {m['cache_lookup_hits']} ({m['cache_lookup_hit_ratio']:.2%})
        Cache Lookups Miss: {m['cache_lookup_misses']}
        Lookup Source Distribution: {m['cache_source_distribution']}

        --- Timing ---
        Average Response Time: {m['avg_response_time_ms']:.2f} ms (based on last {len(m['response_times_ms'])} queries)
        """
        return report.strip()