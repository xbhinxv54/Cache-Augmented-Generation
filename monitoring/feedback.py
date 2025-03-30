import time
from typing import Dict, List, Any, Optional

# Requires TieredCacheManager for demotion
from caching.tiered_cache import TieredCacheManager

class FeedbackSystem:
    """System for collecting and analyzing user feedback"""

    def __init__(self, cache_manager: TieredCacheManager, promotion_threshold: int = 4, demotion_threshold: int = 2):
        self.cache_manager = cache_manager
        # Store feedback keyed by normalized query for grouping
        self.feedback_log: Dict[str, List[Dict[str, Any]]] = {}
        self.promotion_threshold = promotion_threshold  # Rating >= threshold promotes
        self.demotion_threshold = demotion_threshold    # Rating <= threshold demotes

    def record_feedback(self, normalized_query: str, response: str, rating: int, original_query: Optional[str] = None) -> None:
        """Record user feedback (e.g., 1-5 rating) using normalized query as key."""
        if not (1 <= rating <= 5):
            print(f"[Feedback] Invalid rating received: {rating}. Ignoring.")
            return

        if normalized_query not in self.feedback_log:
            self.feedback_log[normalized_query] = []

        feedback_entry = {
            "response": response,
            "rating": rating,
            "timestamp": time.time(),
            "original_query": original_query or normalized_query # Store original if provided
        }
        self.feedback_log[normalized_query].append(feedback_entry)
        # print(f"[Feedback] Recorded rating {rating} for query '{normalized_query}'") # DEBUG

        # Update cache based on this feedback
        self._update_cache_based_on_feedback(normalized_query, response, rating, original_query)

    def _update_cache_based_on_feedback(self, normalized_query: str, response: str, rating: int, original_query: Optional[str]) -> None:
        """Update or remove cache entries based on feedback."""
        query_for_cache_ops = original_query or normalized_query # Use original query if available for L3 matching

        if rating >= self.promotion_threshold:
            # High rating - ensure this query/response pair is strongly cached
            # print(f"[Feedback] Promoting response for query '{normalized_query}' due to rating {rating}.") # DEBUG
            # Use original query for adding, as add_to_cache handles normalization internally
            self.cache_manager.add_to_cache(query_for_cache_ops, response)
        elif rating <= self.demotion_threshold:
            # Poor rating - remove this specific query/response from caches
            # print(f"[Feedback] Demoting response for query '{normalized_query}' due to rating {rating}.") # DEBUG
            # Use the query associated with the feedback for removal attempt
            removed = self.cache_manager.remove_from_cache(query_for_cache_ops)
            # if removed: print(f"[Feedback] Successfully removed from cache levels.") # DEBUG
            # else: print(f"[Feedback] Entry not found or already removed from cache.") # DEBUG


    def get_response_quality_metrics(self) -> Dict[str, Any]:
        """Get aggregate metrics on response quality based on recorded feedback."""
        all_ratings = []
        for entries in self.feedback_log.values():
            for item in entries:
                all_ratings.append(item["rating"])

        total_feedback = len(all_ratings)
        if total_feedback == 0:
            return {
                "average_rating": 0,
                "total_feedback": 0,
                "rating_distribution": {r: 0 for r in range(1, 6)}
            }

        avg_rating = sum(all_ratings) / total_feedback
        rating_dist = {rating: all_ratings.count(rating) for rating in range(1, 6)}

        return {
            "average_rating": avg_rating,
            "total_feedback": total_feedback,
            "rating_distribution": rating_dist
        }

    def get_feedback_for_query(self, normalized_query: str) -> Optional[List[Dict[str, Any]]]:
        """Get all feedback entries for a specific normalized query."""
        return self.feedback_log.get(normalized_query)

    def get_low_rated_queries(self, threshold: int = 2, min_feedback: int = 1) -> List[Dict[str, Any]]:
        """Get queries that consistently receive low ratings."""
        low_rated = []
        for query, entries in self.feedback_log.items():
            if len(entries) >= min_feedback:
                avg_rating = sum(item['rating'] for item in entries) / len(entries)
                if avg_rating <= threshold:
                    low_rated.append({
                        "normalized_query": query,
                        "average_rating": avg_rating,
                        "feedback_count": len(entries),
                        "last_original_query": entries[-1].get("original_query", query) # Get an example original query
                    })
        return sorted(low_rated, key=lambda x: x['average_rating']) # Sort by lowest rating first