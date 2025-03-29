import time
from typing import Dict, List, Any, Optional

class FeedbackSystem:
    """System for collecting and analyzing user feedback"""
    
    def __init__(self, cache_manager):
        self.cache_manager = cache_manager
        self.feedback_log: Dict[str, List[Dict[str, Any]]] = {}
        self.promotion_threshold = 4  # Rating threshold for promotion
        self.demotion_threshold = 2   # Rating threshold for demotion
    
    def record_feedback(self, query: str, response: str, rating: int) -> None:
        """Record user feedback (1-5 rating)"""
        if query not in self.feedback_log:
            self.feedback_log[query] = []
        
        self.feedback_log[query].append({
            "response": response,
            "rating": rating,
            "timestamp": time.time()
        })
        
        self._update_cache_based_on_feedback(query, response, rating)
    
    def _update_cache_based_on_feedback(self, query: str, response: str, rating: int) -> None:
        """Update or remove cache entries based on feedback"""
        if rating >= self.promotion_threshold:
            # High rating - prioritize this response
            self.promote_response(query, response)
        elif rating <= self.demotion_threshold:
            # Poor rating - consider removing from cache
            self.demote_response(query, response)
    
    def promote_response(self, query: str, response: str) -> None:
        """Promote a response by ensuring it's in all cache levels"""
        # This is a simplified implementation
        self.cache_manager.add_to_cache(query, response)
    
    def demote_response(self, query: str, response: str) -> None:
        """Demote a response (could remove from cache in a real implementation)"""
        # This would require additional methods in the cache manager
        # For now, we just log it
        print(f"[FEEDBACK] Response demoted for query: '{query[:30]}...'")
    
    def get_response_quality_metrics(self) -> Dict[str, Any]:
        """Get aggregate metrics on response quality"""
        if not self.feedback_log:
            return {"average_rating": 0, "total_feedback": 0}
            
        all_ratings = [item["rating"] for entries in self.feedback_log.values() 
                      for item in entries]
        
        return {
            "average_rating": sum(all_ratings) / len(all_ratings) if all_ratings else 0,
            "total_feedback": len(all_ratings),
            "rating_distribution": {
                rating: all_ratings.count(rating) for rating in range(1, 6)
            }
        }
    
    def get_feedback_for_query(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """Get all feedback for a specific query"""
        return self.feedback_log.get(query)