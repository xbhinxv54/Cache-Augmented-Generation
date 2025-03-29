import time
from typing import List, Dict, Any
from collections import deque

class TelemetrySystem:
    """System for tracking and analyzing performance metrics"""
    
    def __init__(self, max_history: int = 1000):
        self.metrics = {
            "query_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "response_times": [],
            "cache_hit_ratio": 0,
            "avg_response_time": 0,
            "cache_source_distribution": {"exact": 0, "vector": 0, "none": 0}
        }
        
        # Keep limited history for time-series analysis
        self.history = deque(maxlen=max_history)
    
    def record_query(self, query: str, is_hit: bool, source: str, response_time: float):
        """Record telemetry for a query"""
        # Update counters
        self.metrics["query_count"] += 1
        
        if is_hit:
            self.metrics["cache_hits"] += 1
            if source.startswith("L1"):
                self.metrics["cache_source_distribution"]["exact"] += 1
            else:
                self.metrics["cache_source_distribution"]["vector"] += 1
        else:
            self.metrics["cache_misses"] += 1
            self.metrics["cache_source_distribution"]["none"] += 1
            
        self.metrics["response_times"].append(response_time)
        
        # Keep only the last 100 response times to avoid memory issues
        if len(self.metrics["response_times"]) > 100:
            self.metrics["response_times"] = self.metrics["response_times"][-100:]
        
        # Add to history
        self.history.append({
            "timestamp": time.time(),
            "query": query,
            "is_hit": is_hit,
            "source": source,
            "response_time": response_time
        })
        
        # Update derived metrics
        self._update_derived_metrics()
    
    def _update_derived_metrics(self):
        """Update metrics that are calculated from other metrics"""
        total = self.metrics["query_count"]
        if total > 0:
            self.metrics["cache_hit_ratio"] = self.metrics["cache_hits"] / total
            
        times = self.metrics["response_times"]
        if times:
            self.metrics["avg_response_time"] = sum(times) / len(times)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current telemetry metrics"""
        return self.metrics
    
    def get_recent_history(self, count: int = 20) -> List[Dict[str, Any]]:
        """Get recent query history"""
        return list(self.history)[-count:]
    
    def generate_report(self) -> str:
        """Generate a text report of key metrics"""
        return f"""
        === Cache Performance Report ===
        Total Queries: {self.metrics['query_count']}
        Cache Hit Ratio: {self.metrics['cache_hit_ratio']:.2%}
        Average Response Time: {self.metrics['avg_response_time']:.2f}ms
        
        Cache Source Distribution:
        - Exact Match: {self.metrics['cache_source_distribution']['exact']}
        - Vector Match: {self.metrics['cache_source_distribution']['vector']}
        - No Match: {self.metrics['cache_source_distribution']['none']}
        """