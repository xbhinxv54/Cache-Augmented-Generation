# Cache-Augmented Generation (CAG) System

This project implements a multi-layered caching system integrated with a Large Language Model (LLM) using LangChain. The primary goal is to optimize response generation by serving requests from caches when possible, reducing latency and API costs, while falling back to the LLM for novel queries.

The system utilizes a tiered caching strategy:
*   **L1 Cache:** In-memory exact match cache for identical queries. Fastest lookup.
*   **L2 Cache:** In-memory semantic cache for *frequently accessed* items. Promotes items based on access count and uses sentence embeddings (via Sentence Transformers) to find matches for paraphrased but common queries.
*   **L3 Cache:** Persistent vector store (ChromaDB) using sentence embeddings for broader semantic search capabilities.

## ✨ Features

*   **Tiered Caching:** Implements L1, L2, and L3 caches with configurable TTLs and thresholds.
*   **Semantic Search:** Uses Sentence Transformers (`all-MiniLM-L6-v2`) for generating embeddings stored in ChromaDB (L3) and for L2 similarity checks.
*   **LLM Integration:** Uses LangChain to interact with an LLM (currently configured for Groq Llama 3). Easily adaptable to other LangChain-supported LLMs.
*   **Context Augmentation:** Passes relevant cached documents (from L2/L3 hits deemed low-quality for direct use) as context to the LLM during generation.
*   **Monitoring Dashboard:** Includes an interactive Streamlit dashboard (`run_dashboard_app.py`) to monitor cache hit rates, response times, cache sizes, and recent queries.
*   **Feedback System:** Basic framework for collecting and potentially acting on user feedback (ratings) for generated/cached responses.
*   **Configurable:** Cache TTLs, similarity thresholds, and access counts can be tuned.

## 🛠️ Technology Stack

*   Python 3.9+
*   LangChain & LangChain Community (`langchain`, `langchain-community`, `langchain-groq`)
*   LLM: Groq API (specifically `llama3-8b-8192` as configured)
*   Embeddings: Sentence Transformers (`sentence-transformers`, using `all-MiniLM-L6-v2`)
*   Vector Store: ChromaDB (`chromadb`)
*   Web Framework (Dashboard): Streamlit (`streamlit`)
*   Supporting Libraries: `python-dotenv`, `numpy`, `scikit-learn` (for cosine similarity), `pandas`, `matplotlib`

## 📂 Project Structure
CAG/
│
├── caching/
│ ├── init.py
│ ├── exact_cache.py # L1: ExactMatchCache
│ ├── vector_cache.py # L3: Vector store wrapper (ChromaDB)
│ ├── tiered_cache.py # Manages L1, L2, L3 interactions
│ └── cache_utils.py # (If you have utils like normalize_key here)
│
├── processing/
│ ├── init.py
│ ├── query_preprocessor.py # Query normalization, stopword removal etc.
│ ├── response_generator.py # Handles LLM chain execution
│ └── similarity.py # Similarity calculation (e.g., Jaccard - now less used)
│
├── monitoring/
│ ├── init.py
│ ├── telemetry.py # Telemetry collection system
│ ├── dashboard.py # Streamlit dashboard UI code
│ 
│
├── orchestration/
│ ├── init.py
│ ├── orchestrator.py # Main ImprovedCAGOrchestrator class
│ 
│
├── main.py # Script for running tests
├── run_dashboard_app.py # Script for running the Streamlit dashboard
├── requirements.txt # Python package dependencies
└── .env # Environment variables (API Keys - DO NOT COMMIT)



## 🚀 Getting Started

### Prerequisites

*   Python 3.9 or higher installed.
*   Conda or `venv` recommended for managing virtual environments.
*   Access to the Groq API (or another LLM API supported by LangChain).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd CAG
    ```
2.  **Create and activate a virtual environment:**
    *   Using Conda:
        ```bash
        conda create -n cag_env python=3.9 -y
        conda activate cag_env
        ```
    *   Using venv:
        ```bash
        python -m venv venv
        source venv/bin/activate # On Linux/macOS
        .\venv\Scripts\activate  # On Windows
        ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(If `requirements.txt` doesn't exist yet, create it after installing manually: `pip freeze > requirements.txt`)*

### Environment Variables

1.  Create a file named `.env` in the project root directory (`CAG/`).
2.  Add your API keys to the `.env` file:
    ```dotenv
    GROQ_API_KEY="gsk_YOUR_GROQ_API_KEY_HERE"
    # Add other keys if you switch LLM/Embedding providers
    ```
    **Important:** Ensure the `.env` file is added to your `.gitignore` file to prevent accidentally committing your API keys.

## ▶️ Usage

### 1. Running Tests

This script runs a predefined set of queries to test the caching layers and generation flow. It prints detailed logs to the console.

```bash
python main.py

⚙️ System Overview & Caching Logic
Query Input: A user query enters the ImprovedCAGOrchestrator.ask method.

Normalization: The query is normalized (lowercase, extra spaces removed).

L1 Check: The ExactMatchCache checks if the exact normalized query exists and hasn't expired. If HIT -> Return response. Access count incremented, potential L2 promotion.

L2 Check: If L1 MISS, the TieredCacheManager checks the frequent_cache.

It embeds the input query.

It calculates cosine similarity between the input embedding and the embeddings of keys (normalized queries) stored in the L2 cache.

If a match with similarity >= l2_similarity_threshold is found -> Check quality -> Return response if high quality. Access count for the matched L2 key is updated.

L3 Check: If L1 & L2 MISS, the TieredCacheManager searches the VectorCache (ChromaDB) using the input query embedding and the _get_dynamic_threshold.

If relevant documents are found above the dynamic threshold: Check quality (_is_high_quality_match). The current L3 check only verifies if the vector score >= quality_threshold. If HIT -> Return response. Access count for the input query is updated.

Generation: If L1, L2, and L3 all miss or return low-quality hits, the ResponseGenerator calls the LLM (Groq).

If the miss followed a low-quality L2/L3 hit, the retrieved relevant_docs can be passed as context to the LLM.

Caching New Response: The newly generated response is typically added to:

L1 Cache (Exact Match)

L3 Cache (Vector Store)

L2 Cache if the access count for the normalized query reaches the l2_access_threshold.

🔧 Configuration & Tuning
Key parameters can be adjusted in the __init__ methods of:

caching/tiered_cache.py -> TieredCacheManager:

exact_ttl: L1 Time-to-live (seconds).

frequent_ttl: L2 Time-to-live (seconds).

l2_similarity_threshold: Minimum cosine similarity for an L2 hit (e.g., 0.80).

l2_access_threshold: Number of times a query must be accessed before being promoted to L2 (e.g., 2).

orchestration/orchestrator.py -> ImprovedCAGOrchestrator:

quality_threshold: Minimum vector score from L3 search required to consider using an L3 hit (e.g., 0.67).

l2_quality_threshold: Minimum cosine similarity score from L2 search required to consider using an L2 hit (should generally match l2_similarity_threshold).

caching/tiered_cache.py -> _get_dynamic_threshold:

base_threshold: The starting point for the L3 vector search relevance score threshold (e.g., 0.55).

