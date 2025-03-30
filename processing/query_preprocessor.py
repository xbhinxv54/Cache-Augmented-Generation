from typing import List, Set

class QueryPreprocessor:
    """Preprocessor for query normalization and analysis"""

    def __init__(self, custom_stop_words: List[str] = None):
        # Example stop words, customize as needed
        default_stop_words = [
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
            'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
            'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
            'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
            'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
            'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
            'with', 'about', 'against', 'between', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
            'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
            'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll',
            'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn',
            'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn',
            'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn', 'tell', 'me' # Added 'tell', 'me'
            ]
        self.stop_words: Set[str] = set(custom_stop_words if custom_stop_words is not None else default_stop_words)


    def normalize(self, query: str) -> str:
        """Normalize query by lowercasing and removing extra spaces. Consistent with cache keys."""
        return " ".join(query.lower().split())

    def remove_stopwords(self, query: str) -> str:
        """Normalize and remove common stopwords."""
        # Always normalize first
        normalized = self.normalize(query)
        return " ".join([word for word in normalized.split() if word not in self.stop_words])

    def extract_entities(self, query: str) -> List[str]:
        """Extract key entities from query (Placeholder)."""
        # Placeholder: In a real scenario, use NER library like spaCy or NLTK
        # Example:
        # import spacy
        # nlp = spacy.load("en_core_web_sm") # Load a spaCy model
        # doc = nlp(query)
        # return [ent.text for ent in doc.ents]
        # For now, just return significant words (not stopwords, normalized)
        normalized = self.normalize(query)
        return [word for word in normalized.split() if word not in self.stop_words and len(word) > 1]


    def preprocess_for_search(self, query: str) -> str:
        """Apply preprocessing steps suitable for semantic search (e.g., vector search)."""
        # Example: Normalize and remove stopwords for potentially better semantic matching
        return self.remove_stopwords(query)

    def preprocess_for_exact_match(self, query: str) -> str:
        """Apply preprocessing steps suitable for exact matching (L1 cache)."""
        # Just normalize for exact matching
        return self.normalize(query)