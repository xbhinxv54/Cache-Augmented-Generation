from typing import List, Tuple, Any, Dict
from langchain.schema import Document

class VectorCache:
    """Wrapper for vector store with additional functionality"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    def search(self, query: str, k: int = 3, threshold: float = 0.6) -> List[Tuple[Document, float]]:
        """Search vector store with threshold filtering"""
        results = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
        return [(doc, score) for doc, score in results if score >= threshold]
    
    def add(self, query: str, response: str, metadata: Dict[str, Any] = None) -> None:
        """Add document to vector store"""
        if metadata is None:
            metadata = {}
        
        metadata["query"] = query
        metadata["type"] = "response"
        
        self.vector_store.add_documents([
            Document(page_content=response, metadata=metadata)
        ])
    
    def get_all(self) -> List[Document]:
        """Get all documents in the vector store"""
        return self.vector_store.get()