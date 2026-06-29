# config.py
# Configuration file for hallucination mitigation pipeline

# ========================
# Embedding Model Settings
# ========================
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768

# ========================
# Vector DB Paths
# ========================
FAISS_INDEX_PATH = "vectordb/index.faiss"
DOCUMENTS_PATH = "vectordb/documents.json"

# ========================
# Dataset Limits
# ========================
# Can be increased for full-scale experiments
MAX_APPS = 10000
MAX_CODESEARCHNET = 100000
MAX_HUMANEVAL = 164  # All samples
MAX_GITHUB_CODE = 50000

# ========================
# Processing Settings
# ========================
BATCH_SIZE = 64
NORMALIZE_EMBEDDINGS = True
NUM_WORKERS = 4  # For parallel dataset processing

# ========================
# Code Filters
# ========================
MIN_CODE_LENGTH = 50  # Characters
MAX_CODE_LENGTH = 2048  # Characters
LANGUAGE = "python"

# ========================
# Retrieval Settings
# ========================
# Retrieval confidence threshold (average hybrid score)
# Lowered to 0.10 to reduce unnecessary query reformulation
RETRIEVAL_CONFIDENCE_THRESHOLD = 0.10  # Accept lower confidence to avoid reformulation
MAX_REFORMULATION_ATTEMPTS = 3  # Number of query reformulation attempts
DEFAULT_TOP_K = 8

# Hierarchical Retrieval (for hallucination mitigation)
# Stage 1: Hybrid Search retrieves broader set
HYBRID_SEARCH_TOP_K = 100  # Retrieve more candidates initially

# Stage 2: Cross-Encoder Reranking (Hierarchical Retrieval)
ENABLE_CROSS_ENCODER = True  # Set False to disable for ablation studies
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Balanced performance
# Alternatives:
#   - cross-encoder/ms-marco-TinyBERT-L-2-v2 (faster, less accurate)
#   - cross-encoder/ms-marco-MiniLM-L-12-v2 (slower, more accurate)
CROSS_ENCODER_TOP_K = 8  # Final number passed to LLM

# ========================
# HNSW Index Parameters
# ========================
# HNSW is much faster than flat index (6-10x speedup)
# and scales well to millions of vectors
USE_HNSW = True  # Set to False for exact search with flat index

# HNSW Parameters (only used if USE_HNSW=True):
HNSW_M = 32  # Connections per node (16-64, higher = more accurate, more memory)
             # 32 = good balance for most cases
HNSW_EF_CONSTRUCTION = 40  # Build quality (40-200, higher = better index, slower build)
HNSW_EF_SEARCH = 16  # Search quality (16-128, higher = more accurate, slower search)
                     # Can be adjusted at query time for speed/accuracy tradeoff

# ========================
# Pipeline Settings
# ========================
MAX_PIPELINE_ATTEMPTS = 3

# ========================
# Validation Thresholds
# ========================
OUTPUT_SIMILARITY_THRESHOLD = 0.72
EXECUTION_TIMEOUT = 10  # seconds

# ========================
# SAP AI Core / Generative AI Hub Settings
# ========================
# Set these environment variables or create .env file:
# AICORE_AUTH_URL
# AICORE_CLIENT_ID
# AICORE_CLIENT_SECRET
# AICORE_BASE_URL
# AICORE_RESOURCE_GROUP

# LLM Configuration
LLM_MODEL_NAME = "gemini-2.5-flash"  # SAP AI Core model deployment name
LLM_TEMPERATURE = 0.3  # Lower = more deterministic, higher = more creative
LLM_MAX_TOKENS = 512  # Maximum tokens in generated code
NUM_CANDIDATES = 1  # Generate 1 code directly (rely on good retrieval)

# System prompt for code generation
CODE_GENERATION_SYSTEM_PROMPT = """You are an expert Python programmer.
CRITICAL: Your task is to copy the logic from the retrieved examples, NOT to improve or refactor them.

STRICT RULES:
1. Use the EXACT function name from the test cases
2. Copy the EXACT logic and algorithm from the most similar example
3. Keep the EXACT return type (if example returns "None" as string, you must too)
4. DO NOT refactor, optimize, or "improve" the code
5. DO NOT change variable names unless absolutely necessary
6. Only output the Python code without explanations or markdown formatting."""
