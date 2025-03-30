from typing import List, Tuple, Any, Dict
from langchain.schema import Document
import time
class VectorCache:
    """Wrapper for vector store with additional functionality"""

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def search(self, query: str, k: int = 3, threshold: float = 0.6) -> List[Tuple[Document, float]]:
        """Search vector store with threshold filtering, returning documents and scores."""
        try:
            results_with_scores = self.vector_store.similarity_search_with_relevance_scores(query, k=k)

            # Filter based on the threshold
            filtered_results = [(doc, score) for doc, score in results_with_scores if score >= threshold]
            # print(f"[VectorCache] Search for '{query[:30]}...' (k={k}, th={threshold:.2f}): Found {len(results_with_scores)}, Filtered to {len(filtered_results)}") # DEBUG
            return filtered_results
        except Exception as e:
            print(f"[VectorCache] Error during search: {e}")
            return []


    def add(self, query: str, response: str, metadata: Dict[str, Any] = None) -> None:
        """Add document to vector store"""
        if metadata is None:
            metadata = {}

        # Ensure essential metadata is present
        original_query = metadata.get("original_query", query) # Store the original query used for generation
        metadata["original_query"] = original_query
        metadata["response_added_timestamp"] = time.time()
        metadata["type"] = "response" # Indicate this is a cached response

        print(f"[VectorCache] Attempting ADD: Original Query='{original_query[:50]}...'") # DEBUG
        doc_to_add = Document(page_content=response, metadata=metadata)
        try:
            added_ids = self.vector_store.add_documents([doc_to_add])
            print(f"[VectorCache] ADD successful. Doc ID(s): {added_ids}. Original Query='{original_query[:50]}...'") # DEBUG

            # --- Add immediate count check ---
            if hasattr(self.vector_store, '_collection'):
                try:
                    count = self.vector_store._collection.count()
                    print(f"[VectorCache] Collection count immediately after add: {count}") # DEBUG
                except Exception as ce:
                    print(f"[VectorCache] Error getting count after add: {ce}")
            # --- End count check ---

        except Exception as e:
            print(f"[VectorCache] Error during add for query '{original_query[:50]}...': {e}")


    def get_all(self) -> List[Document]:
        """Get all documents in the vector store (potentially inefficient)"""
        try:
            if hasattr(self.vector_store, '_collection'):
                 results = self.vector_store._collection.get()
                 # Reconstruct Document objects if needed (adjust based on Chroma's get() format)
                 docs = []
                 if results and results.get('documents'):
                     for i, content in enumerate(results['documents']):
                         meta = results['metadatas'][i] if results.get('metadatas') and i < len(results['metadatas']) else {}
                         docs.append(Document(page_content=content, metadata=meta))
                 return docs
            else:
                 print("[VectorCache] Warning: Cannot get all documents directly.")
                 return []

        except Exception as e:
            print(f"[VectorCache] Error during get_all: {e}")
            return []


    def remove_by_metadata(self, query_to_remove: str) -> int:
        """Remove documents matching a specific original query in metadata."""

        count = 0
        try:
            if hasattr(self.vector_store, '_collection'):
                # Find IDs matching the metadata filter
                ids_to_delete = self.vector_store._collection.get(
                    where={"original_query": query_to_remove},
                    include=[] # Don't need embeddings, docs, metadata here
                )['ids']

                if ids_to_delete:
                    # print(f"[VectorCache] REMOVE: Found {len(ids_to_delete)} docs for query '{query_to_remove[:30]}...'") # DEBUG
                    self.vector_store._collection.delete(ids=ids_to_delete)
                    count = len(ids_to_delete)
            else:
                print("[VectorCache] Warning: Removal by metadata not supported or implemented for this store.")
        except Exception as e:
            print(f"[VectorCache] Error during remove_by_metadata for query '{query_to_remove[:30]}...': {e}")
        return count