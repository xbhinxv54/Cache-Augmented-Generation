import os
from dotenv import load_dotenv
import warnings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

# Import our modules
from caching.tiered_cache import TieredCacheManager
from processing.response_generator import ResponseGenerator
from orchestration.orchestrator import ImprovedCAGOrchestrator
from monitoring.dashboard import run_dashboard

# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY')

def initialize_system():
    """Initialize the complete system"""
    # Initialize memory
    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        input_key="input"
    )
    
    # Create prompts
    base_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer concisely using conversation history if needed:"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    augmented_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using context and history:"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("system", "Relevant Context:\n{context}"),
        ("human", "{input}")
    ])
    
    # Initialize embedding model
    embed_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )
    
    # Create vector store
    vector_store = Chroma(
        embedding_function=embed_model,
        persist_directory="./chroma_db"
    )
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        api_key=google_api_key,
        model="gemini-1.5-pro"
    )
    
    # Initialize our components
    cache_manager = TieredCacheManager(vector_store)
    
    response_generator = ResponseGenerator(
        llm=llm,
        base_prompt=base_prompt,
        augmented_prompt=augmented_prompt,
        memory=memory
    )
    
    # Create the orchestrator
    orchestrator = ImprovedCAGOrchestrator(
        cache_manager=cache_manager,
        response_generator=response_generator
    )
    
    return orchestrator, memory

def run_tests(orchestrator, memory):
    """Run some test queries"""
    # Clear memory before testing
    memory.clear()
    
    # Test 1: First query (will be a miss)
    print("=" * 50)
    print("--- Test 1 ---")
    response1 = orchestrator.ask("What is the capital of France?")
    print("Response:", response1)
    
    # Test 2: Repeat the same query (should be a hit)
    print("\n" + "=" * 50)
    print("--- Test 2 ---")
    response2 = orchestrator.ask("What is the capital of France?")
    print("Response:", response2)
    
    # Test 3: Different query
    print("\n" + "=" * 50)
    print("--- Test 3 ---")
    response3 = orchestrator.ask("What's the population of Paris?")
    print("Response:", response3)
    
    # Test 4: Now the population query should be cached
    print("\n" + "=" * 50)
    print("--- Test 4 ---")
    response4 = orchestrator.ask("What's the population of Paris?")
    print("Response:", response4)
    
    # Test 5: Slight variation of first query
    print("\n" + "=" * 50)
    print("--- Test 5 ---")
    response5 = orchestrator.ask("Tell me the capital of France")
    print("Response:", response5)
    
    # Print statistics
    print("\n" + "=" * 50)
    print("System Statistics:")
    stats = orchestrator.get_stats()
    print(f"- Queries processed: {stats['queries_processed']}")
    print(f"- Cache hit ratio: {stats['cache_stats']['hit_ratio']:.2%}")
    print(f"- Avg response time: {stats['telemetry']['avg_response_time']:.2f}ms")

if __name__ == "__main__":
    orchestrator, memory = initialize_system()
    
    # Choose mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--dashboard":
        # Run monitoring dashboard
        run_dashboard(orchestrator)
    else:
        # Run tests
        run_tests(orchestrator, memory)