from typing import List, Set

class QueryPreprocessor:
    """Preprocessor for query normalization and analysis"""
    
    def __init__(self, custom_stop_words: List[str] = None):
        default_stop_words = ['the', 'is', 'and', 'of', 'to', 'a', 'in', 'that', 'for']
        self.stop_words: Set[str] = set(custom_stop_words or default_stop_words)
    
    def normalize(self, query: str) -> str:
        """Normalize query by lowercasing, removing extra spaces"""
        return " ".join(query.lower().split())
    
    def remove_stopwords(self, query: str) -> str:
        """Remove common stopwords"""
        return " ".join([word for word in query.split() if word not in self.stop_words])
    
    def extract_entities(self, query: str) -> List[str]:
        """Extract key entities from query (could use spaCy or similar)"""
        # Placeholder for entity extraction
        # In a real implementation, you would use NER here
        return query.split()
    
    def preprocess(self, query: str) -> str:
        """Apply all preprocessing steps"""
        normalized = self.normalize(query)
        filtered = self.remove_stopwords(normalized)
        return filtered