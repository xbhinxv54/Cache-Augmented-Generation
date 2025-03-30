# main.py
import os
from dotenv import load_dotenv
import warnings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
# from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_groq import ChatGroq # Import Groq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
import time 
from caching.tiered_cache import TieredCacheManager
from processing.response_generator import ResponseGenerator
from orchestration.orchestrator import ImprovedCAGOrchestrator
from monitoring.dashboard import run_dashboard

# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY') # Still needed for embeddings
groq_api_key ='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'

def initialize_system():
    """Initialize the complete system"""
    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        input_key="input"
    )

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

    
    print("[Initializer] Initializing Google Embeddings...")
    embed_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )
    print("[Initializer] Google Embeddings Initialized.")

    # Create vector store
    print("[Initializer] Initializing Chroma Vector Store...")
    vector_store = Chroma(
        embedding_function=embed_model, # Use the chosen embedding function
        persist_directory="./chroma_db"
    )
    print("[Initializer] Chroma Vector Store Initialized.")

    # Initialize LLM (NOW USING GROQ)
    print("[Initializer] Initializing Groq LLM...")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama3-8b-8192" # Or "mixtral-8x7b-32768", etc. - choose a model on Groq
        # temperature=0.7 # Optional
    )
    print(f"[Initializer] Groq LLM Initialized with model: {llm.model_name}")

   
    print("[Initializer] Initializing TieredCacheManager...")
    cache_manager = TieredCacheManager(
        vector_store_instance=vector_store, 
        exact_ttl=3600,
        frequent_ttl=86400,
        l2_similarity_threshold=0.85,
        l2_access_threshold=3
        )
    print("[Initializer] TieredCacheManager Initialized.")

    print("[Initializer] Initializing ResponseGenerator...")
    response_generator = ResponseGenerator(
        llm=llm, 
        base_prompt=base_prompt,
        augmented_prompt=augmented_prompt,
        memory=memory
    )
    print("[Initializer] ResponseGenerator Initialized.")

    print("[Initializer] Initializing ImprovedCAGOrchestrator...")
    orchestrator = ImprovedCAGOrchestrator(
        cache_manager=cache_manager,
        response_generator=response_generator,
        quality_threshold=0.75 
    )
    print("[Initializer] ImprovedCAGOrchestrator Initialized.")

    return orchestrator, memory


def run_tests(orchestrator, memory):
    """Run some test queries"""
    
    memory.clear()

    queries = [
        "What is the capital of France?",
        "What is the capital of France?",
        "What's the population of Paris?",
        "What's the population of Paris?",
        "Tell me the capital of France",
        "How tall is the Eiffel Tower?",
        "How tall is the Eiffel Tower?",
        "summarize the plot of hamlet",
        "Briefly summarize Hamlet's plot",
        "What is the capital of France again?",
    ]

    for i, query in enumerate(queries):
        print("\n" + "=" * 50)
        print(f"--- Test {i+1}: Query: '{query}' ---")
        response = orchestrator.ask(query)
        print(f"Response: {response}")
        # Add a small delay to potentially avoid rate limits and allow time diffs
        time.sleep(0.5) 


    # Print statistics
    
    print("\n" + "=" * 50)
    print("System Statistics:")
    stats = orchestrator.get_stats()
    
    print(f"- Queries processed: {stats['overall_performance']['queries_processed']}")
    print(f"- Actual Response Hit Ratio: {stats['overall_performance']['actual_response_hit_ratio']:.2%}")
    print(f"- Avg response time: {stats['overall_performance']['avg_response_time_ms']:.2f}ms")
    print(f"- Cache Lookup Hit Ratio (TieredCacheManager): {stats['cache_performance']['overall_hit_ratio']:.2%}")
    print(f"- Cache Sizes (L1 / L2 / L3 Approx): {stats['cache_performance']['l1_size']} / {stats['cache_performance']['l2_size']} / {stats['cache_performance']['l3_approx_size']}")
    print(f"- Response Sources Logged: {stats['response_source_distribution']}")
    print(f"- Feedback Summary: {stats['feedback_summary']}")
    print("\n" + "=" * 50)
    # print("Telemetry Details:")
    # import json
    # print(json.dumps(stats['telemetry_details'], indent=2))
    # print("\n" + "=" * 50)
    # print("Final Cache Manager Raw Stats:")
    # print(json.dumps(stats['cache_performance'], indent=2))


if __name__ == "__main__":
    orchestrator, memory = initialize_system()

    # Choose mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--dashboard":
        run_dashboard(orchestrator)
    else:
        run_tests(orchestrator, memory)

    orchestrator.cleanup()