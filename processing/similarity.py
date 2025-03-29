from typing import Set

def calculate_similarity(str1: str, str2: str) -> float:
    """Calculate similarity between two strings using Jaccard similarity"""
    # Convert to sets of words
    words1: Set[str] = set(str1.lower().split())
    words2: Set[str] = set(str2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0