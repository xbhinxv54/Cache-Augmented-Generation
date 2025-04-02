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
        # --- Group 1: France Capital ---
        "What is the capital of France?",               # 1: Miss -> Gen -> L1/L3, A=1
        "What is the capital of France?",               # 2: L1 Hit, A=2 -> Promote L2
        "Tell me the capital of France.",               # 3: Expect L2 Hit
        "France's capital city?",                       # 4: Expect L2 Hit
        "What city serves as the capital of France?",   # 5: Expect L2/L3 Hit

        # --- Group 2: Paris Population ---
        "What's the population of Paris?",              # 6: Miss -> Gen -> L1/L3, A=1
        "What's the population of Paris?",              # 7: L1 Hit, A=2 -> Promote L2
        "Population count for Paris?",                  # 8: Expect L2 Hit
        "How many people live in Paris?",               # 9: Expect L2 Hit
        "Tell me Paris's population number.",           # 10: Expect L2/L3 Hit

        # --- Group 3: Eiffel Tower Height ---
        "How tall is the Eiffel Tower?",                # 11: Miss -> Gen -> L1/L3, A=1
        "How tall is the Eiffel Tower?",                # 12: L1 Hit, A=2 -> Promote L2
        "Height of the Eiffel Tower?",                  # 13: Expect L2 Hit
        "What is the Eiffel Tower's height?",           # 14: Expect L2 Hit
        "Eiffel Tower altitude?",                       # 15: Expect L2/L3 Hit (Slightly different phrasing)

        # --- Group 4: Hamlet Plot ---
        "Summarize the plot of Hamlet.",                # 16: Miss -> Gen -> L1/L3, A=1
        "Summarize the plot of Hamlet.",                # 17: L1 Hit, A=2 -> Promote L2
        "Briefly summarize Hamlet's plot.",             # 18: Expect L2 Hit
        "What happens in the play Hamlet?",             # 19: Expect L2/L3 Hit
        "Give a synopsis of Shakespeare's Hamlet.",     # 20: Expect L2/L3 Hit

        # --- Group 5: Photosynthesis ---
        "Explain the process of photosynthesis.",       # 21: Miss -> Gen -> L1/L3, A=1
        "Explain the process of photosynthesis.",       # 22: L1 Hit, A=2 -> Promote L2
        "What is photosynthesis?",                      # 23: Expect L2 Hit
        "How do plants make food using sunlight?",      # 24: Expect L2/L3 Hit
        "Describe the energy conversion in plants.",    # 25: Expect L2/L3 Hit

        # --- Group 6: Water's Boiling Point ---
        "What is the boiling point of water in Celsius?", # 26: Miss -> Gen -> L1/L3, A=1
        "What is the boiling point of water in Celsius?", # 27: L1 Hit, A=2 -> Promote L2
        "At what Celsius temp does water boil?",        # 28: Expect L2 Hit
        "Water's boiling point (C)?",                   # 29: Expect L2 Hit
        "Boiling temperature for H2O in Celsius?",      # 30: Expect L2/L3 Hit

        # --- Group 7: Author of '1984' ---
        "Who wrote the book '1984'?",                   # 31: Miss -> Gen -> L1/L3, A=1
        "Who wrote the book '1984'?",                   # 32: L1 Hit, A=2 -> Promote L2
        "Author of 1984?",                              # 33: Expect L2 Hit
        "Which author penned Nineteen Eighty-Four?",    # 34: Expect L2/L3 Hit
        "George Orwell wrote which famous dystopian novel?", # 35: Expect L3 Hit (Tests relation)

        # --- Group 8: Solar System Planets ---
        "How many planets are in our solar system?",    # 36: Miss -> Gen -> L1/L3, A=1
        "How many planets are in our solar system?",    # 37: L1 Hit, A=2 -> Promote L2
        "Number of planets orbiting the sun?",          # 38: Expect L2 Hit
        "Count the solar system's planets.",            # 39: Expect L2 Hit
        "Total planets in this solar system?",          # 40: Expect L2/L3 Hit

        # --- Group 9: Artificial Intelligence Definition ---
        "Define Artificial Intelligence.",              # 41: Miss -> Gen -> L1/L3, A=1
        "Define Artificial Intelligence.",              # 42: L1 Hit, A=2 -> Promote L2
        "What is AI?",                                  # 43: Expect L2 Hit
        "Explain the concept of AI.",                   # 44: Expect L2 Hit
        "What does the field of Artificial Intelligence study?", # 45: Expect L2/L3 Hit

        # --- Group 10: Re-asking Earlier Promoted Queries (Testing L1/L2 Interaction) ---
        "What is the capital of France?",               # 46: Expect L1 Hit (Even though promoted, L1 checked first)
        "How tall is the Eiffel Tower?",                # 47: Expect L1 Hit
        "Explain the process of photosynthesis.",       # 48: Expect L1 Hit
        "What is the boiling point of water in Celsius?", # 49: Expect L1 Hit
        "Who wrote the book '1984'?",                   # 50: Expect L1 Hit

        # --- Group 11: More Variations on Promoted Items ---
        "Tell me the capital of France again.",         # 51: Expect L2 Hit (Should hit L2 cache directly)
        "The height of Paris's famous tower?",          # 52: Expect L2/L3 Hit (Variation on Eiffel Tower)
        "Quick summary of Hamlet?",                     # 53: Expect L2 Hit
        "How does AI work (simply)?",                   # 54: Expect L2/L3 Hit (Variation on AI definition)
        "Number of planets we have?",                   # 55: Expect L2 Hit (Variation on solar system)
        "Water boils at what C degree?",                # 56: Expect L2 Hit
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