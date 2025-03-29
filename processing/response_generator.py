from typing import List, Dict, Any
from langchain.schema import Document
from langchain.chains import LLMChain

class ResponseGenerator:
    """Generates responses using LLM chains"""
    
    def __init__(self, llm, base_prompt, augmented_prompt, memory):
        self.llm = llm
        self.base_prompt = base_prompt
        self.augmented_prompt = augmented_prompt
        self.memory = memory
        
        # Initialize chains
        self.base_chain = LLMChain(
            llm=self.llm,
            prompt=self.base_prompt,
            memory=self.memory,
            verbose=True
        )
        
        self.augmented_chain = LLMChain(
            llm=self.llm,
            prompt=self.augmented_prompt,
            memory=self.memory,
            verbose=True
        )

    def retrieve_cached_response(self, documents: List[Document]) -> str:
        """Return the cached response directly"""
        #print("Replying with Cache")
        if documents:
            # Return the content of the most relevant document
            return documents[0].page_content
        return "No cached response found."

    def generate_new_response(self, query: str, context: List[Document] = None) -> str:
        """Generate a new response using the appropriate LLM chain"""
        #print("Generating new response")
        if context:
            # We have relevant documents - use them as context
            context_str = "\n".join([doc.page_content for doc in context])
            return self.augmented_chain.invoke({
                "input": query,
                "context": context_str
            })["text"]
        else:
            # No relevant context - use base prompt
            return self.base_chain.invoke({
                "input": query
            })["text"]