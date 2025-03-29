import time
from typing import Dict, Any, List
from caching.tiered_cache import TieredCacheManager
from processing.response_generator import ResponseGenerator
from processing.query_preprocessor import QueryPreprocessor
from monitoring.telemetry import TelemetrySystem
from monitoring.feedback import FeedbackSystem

class ImprovedCAGOrchestrator:
    """Main orchestrator for the cache-augmented generation system"""
    
    def __init__(self, cache_manager: TieredCacheManager, response_generator: ResponseGenerator):
        self.cache_manager = cache_manager
        self.response_generator = response_generator
        self.telemetry = TelemetrySystem()
        self.feedback_system = FeedbackSystem(cache_manager)
        self.query_preprocessor = QueryPreprocessor()
        self.query_log = []
    
    def ask(self, query: str) -> str:
        """Process a query and return the response"""
        # Start timing
        start_time = time.time()
        
        # Store original query for response generation
        original_query = query
        
        # Preprocess query for cache lookup only
        processed_query = self.query_preprocessor.normalize(query)  # Only normalize, don't remove stopwords
        
        # Check cache with improved logic
        relevant_docs, is_hit, source = self.cache_manager.check_cache(processed_query)
        
        # Generate response
        if is_hit and self._is_high_quality_match(relevant_docs, processed_query):
            response = self.response_generator.retrieve_cached_response(relevant_docs)
            response_source = "cache"
        else:
            # For a cache miss or low-quality hit, use original query to preserve meaning
            response = self.response_generator.generate_new_response(original_query, 
                                                                    [] if not is_hit else relevant_docs)
            response_source = "generated"
            
            # Only add high-quality responses to cache
            if self._should_cache_response(original_query, response):
                self.cache_manager.add_to_cache(processed_query, response)  # Use processed query for cache key
        
        # Record telemetry
        end_time = time.time()
        response_time = (end_time - start_time) * 1000  # Convert to ms
        self.telemetry.record_query(original_query, is_hit, source, response_time)
        
        # Add to query log
        self.query_log.append({
            "timestamp": time.time(),
            "query": original_query,
            "processed_query": processed_query,
            "is_hit": is_hit,
            "source": source,
            "response_source": response_source,
            "response_time_ms": response_time
        })
        
        return response
    
    def provide_feedback(self, query: str, response: str, rating: int) -> None:
        """Record user feedback (1-5 rating)"""
        self.feedback_system.record_feedback(query, response, rating)
    
    def _is_high_quality_match(self, docs: List, query: str) -> bool:
        """Determine if the cached match is high quality enough to use"""
        if not docs:
            return False
            
        # Simple implementation - could be more sophisticated
        top_doc = docs[0]
        
        # Check if it's from exact cache (highest quality)
        if hasattr(top_doc, 'metadata') and top_doc.metadata.get('source', '').startswith('L1'):
            return True
            
        # For vector matches, implement additional quality check based on metadata
        if hasattr(top_doc, 'metadata'):
            # If this is a vector match (L3), check the metadata for the original query
            original_query = top_doc.metadata.get('query', '')
            if original_query:
                # Use similarity calculation to ensure high relevance
                from processing.similarity import calculate_similarity
                similarity = calculate_similarity(query, original_query)
                return similarity >= 0.8  # Stricter threshold for quality check
        
        return False  # Default to not using low-confidence matches
    
    def _should_cache_response(self, query: str, response: str) -> bool:
        """Determine if a response should be cached"""
        # Don't cache very short responses (might be error messages)
        if len(response) < 20:
            return False
            
        # Don't cache responses to very specific queries unlikely to be repeated
        if len(query.split()) > 15 and any(char in query for char in ['?', '!']):
            return False
            
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the system"""
        # Calculate actual cache hit ratio from query log
        if not self.query_log:
            actual_hit_ratio = 0
        else:
            hits = sum(1 for entry in self.query_log if entry["response_source"] == "cache")
            actual_hit_ratio = hits / len(self.query_log)
        
        return {
            "cache_stats": {
                **self.cache_manager.get_stats(),
                "actual_hit_ratio": actual_hit_ratio
            },
            "telemetry": self.telemetry.get_metrics(),
            "feedback": self.feedback_system.get_response_quality_metrics(),
            "queries_processed": len(self.query_log),
            "actual_hits": sum(1 for entry in self.query_log if entry["response_source"] == "cache")
        }