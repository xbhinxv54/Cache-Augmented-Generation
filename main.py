import os
from dotenv import load_dotenv
import warnings
import shutil # For deleting DB directory
import time

# Langchain and Community Imports
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings # USE THIS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings # REMOVE/COMMENT OUT
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

# Import our modules
from caching.tiered_cache import TieredCacheManager
from processing.response_generator import ResponseGenerator
from orchestration.orchestrator import ImprovedCAGOrchestrator
from monitoring.dashboard import run_dashboard # Keep if using dashboard

# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# --- UPDATED FUNCTION ---
def initialize_system():
    """Initialize the complete system with Sentence Transformers and L2 tuning"""
    print("[Initializer] Initializing Memory and Prompts...")
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
    print("[Initializer] Memory and Prompts Initialized.")

    # Initialize embedding model (NOW USING SENTENCE TRANSFORMERS)
    print("[Initializer] Initializing Sentence Transformer Embeddings (all-MiniLM-L6-v2)...")
    # Use HuggingFaceEmbeddings for Sentence Transformers
    model_kwargs = {'device': 'cpu'} # Use 'cuda' if GPU available
    encode_kwargs = {'normalize_embeddings': False} # Keep False for Chroma default L2 distance
    embed_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2", # Popular default choice
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("[Initializer] Sentence Transformer Embeddings Initialized.")

    # Create vector store
    print("[Initializer] Initializing Chroma Vector Store...")
    vector_store = Chroma(
        embedding_function=embed_model, # Use the new embedding function
        persist_directory="./chroma_db"
    )
    print("[Initializer] Chroma Vector Store Initialized.")

    # Initialize LLM (Groq)
    print("[Initializer] Initializing Groq LLM...")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama3-8b-8192"
    )
    print(f"[Initializer] Groq LLM Initialized with model: {llm.model_name}")

    # Initialize TieredCacheManager (Tune L2, Pass Embed Model)
    print("[Initializer] Initializing TieredCacheManager (with L2 tuning and Embed Model)...")
    # Set L2 thresholds here (e.g., 0.8 for cosine similarity, 2 for access)
    l2_sim_threshold = 0.80
    l2_acc_threshold = 2
    cache_manager = TieredCacheManager(
        vector_store_instance=vector_store,
        embedding_model=embed_model, # Pass the embedding model
        exact_ttl=3600,
        frequent_ttl=86400,
        l2_similarity_threshold=l2_sim_threshold,
        l2_access_threshold=l2_acc_threshold
    )
    print(f"[Initializer] TieredCacheManager Initialized (L2 Access: {cache_manager.l2_access_threshold}, L2 Sim: {cache_manager.l2_similarity_threshold})")

    # Initialize ResponseGenerator
    print("[Initializer] Initializing ResponseGenerator...")
    response_generator = ResponseGenerator(
        llm=llm,
        base_prompt=base_prompt,
        augmented_prompt=augmented_prompt,
        memory=memory
    )
    print("[Initializer] ResponseGenerator Initialized.")

    # Initialize Orchestrator (Tune L3, Add L2 threshold, Remove L3 semantic)
    print("[Initializer] Initializing ImprovedCAGOrchestrator (with threshold tuning)...")
    orchestrator = ImprovedCAGOrchestrator(
        cache_manager=cache_manager,
        response_generator=response_generator,
        quality_threshold=0.67, # L3 Vector threshold (adjust as needed)
        l2_quality_threshold=cache_manager.l2_similarity_threshold # Sync L2 threshold
    )
    print("[Initializer] ImprovedCAGOrchestrator Initialized.")

    return orchestrator, memory

# --- UPDATED FUNCTION ---
def run_tests(orchestrator, memory):
    """Run some test queries"""
    # Clear memory before testing
    memory.clear()

    queries = [
        "What is the capital of France?",       # Test 1: Gen -> Cache L1/L3
        "What is the capital of France?",       # Test 2: Expect L1 Hit, Access++ (Count=2) -> Promote L2
        "What's the population of Paris?",      # Test 3: Gen -> Cache L1/L3
        "What's the population of Paris?",      # Test 4: Expect L1 Hit, Access++ (Count=2) -> Promote L2
        "Tell me the capital of France",        # Test 5: Expect L2 Hit (if sim >= thresh) OR L3 Hit
        "How tall is the Eiffel Tower?",        # Test 6: Gen -> Cache L1/L3
        "How tall is the Eiffel Tower?",        # Test 7: Expect L1 Hit, Access++ (Count=2) -> Promote L2
        "summarize the plot of hamlet",         # Test 8: Gen -> Cache L1/L3
        "Briefly summarize Hamlet's plot",      # Test 9: Expect L2 Hit (if sim >= thresh) OR L3 Hit
        "What is the capital of France again?", # Test 10: Expect L2 Hit OR L3 Hit
        "Tell me the capital of France",        # Test 11: Expect L2 Hit (refresh timestamp) OR L3 Hit
    ]

    for i, query in enumerate(queries):
        print("\n" + "=" * 50)
        print(f"--- Test {i+1}: Query: '{query}' ---")
        response = orchestrator.ask(query)
        print(f"Response: {response}")
        time.sleep(0.5) # Keep delay

    # Print statistics
    print("\n" + "=" * 50)
    print("System Statistics:")
    stats = orchestrator.get_stats() # Make sure get_stats is updated for L2/L3 counts if needed
    print(f"- Queries processed: {stats['overall_performance']['queries_processed']}")
    print(f"- Actual Response Hit Ratio: {stats['overall_performance']['actual_response_hit_ratio']:.2%}")
    print(f"- Avg response time: {stats['overall_performance']['avg_response_time_ms']:.2f}ms")
    print(f"- Cache Lookup Hit Ratio (TieredCacheManager): {stats['cache_performance']['overall_hit_ratio']:.2%}")
    print(f"- Cache Sizes (L1 / L2 / L3 Approx): {stats['cache_performance']['l1_size']} / {stats['cache_performance']['l2_size']} / {stats['cache_performance']['l3_approx_size']}")
    print(f"- Response Sources Logged: {stats['response_source_distribution']}") # Verify L2 counts here
    print(f"- Feedback Summary: {stats['feedback_summary']}")

# --- Main execution block ---
if __name__ == "__main__":
    # IMPORTANT: Delete old DB before running!
    db_path = "./chroma_db"
    if os.path.exists(db_path):
        print(f"DELETING existing ChromaDB at {db_path} due to embedding model change.")
        try:
            shutil.rmtree(db_path)
            print("Deletion complete.")
        except OSError as e:
            print(f"Error deleting directory {db_path}: {e}")
            # Decide how to handle error, e.g., exit or warn
            # sys.exit(1) # Optional: exit if deletion fails

    orchestrator, memory = initialize_system()

    # Choose mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--dashboard":
        run_dashboard(orchestrator)
    else:
        run_tests(orchestrator, memory)

    orchestrator.cleanup()