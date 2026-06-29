# Hallucination Mitigation RAG Pipeline for Code Generation

A complete Retrieval-Augmented Generation (RAG) system with hierarchical retrieval and multi-layer validation to mitigate hallucinations in LLM-generated code.

---

## 🎯 Research Contributions

### 1. **Hierarchical Retrieval for Hallucination Mitigation**
Three-stage retrieval pipeline for high-quality context:
- **Stage 1**: Hybrid Search (Dense Vector + BM25) → top-100 candidates
- **Stage 2**: Metadata Filtering (Context-Aware Retrieval) → top-50 filtered
- **Stage 3**: Cross-Encoder Reranking → top-8 best examples

### 2. **Context-Aware Retrieval with Metadata Constraints**
Filters examples by language, complexity, code length, and library dependencies to ensure contextually relevant retrieval.

### 3. **Multi-Layer Validation System**
4-layer validation to detect hallucinations:
- **Layer 1**: Syntax checking (AST parsing)
- **Layer 2**: API correctness (hallucination detection)
- **Layer 3**: Type checking (mypy)
- **Layer 4**: Execution testing

### 4. **Targeted Feedback Loop**
Uses validation errors to build targeted queries for improved re-retrieval (max 3 attempts).

---

## 🏗️ Pipeline Architecture

```
User Query
    ↓
┌─────────────────────────────────────────────────────────┐
│ STAGE 1: HIERARCHICAL RETRIEVAL                        │
├─────────────────────────────────────────────────────────┤
│ 1. Hybrid Search (BGE + BM25) → 100 candidates         │
│ 2. Calculate Retrieval Confidence                      │
│    - If confidence < 0.63 → Query Reformulation        │
│    - If confidence ≥ 0.63 → Continue                   │
│ 3. Metadata Filtering → 50 candidates                  │
│ 4. Cross-Encoder Reranking → 8 best                    │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ STAGE 2: CODE GENERATION                               │
├─────────────────────────────────────────────────────────┤
│ 5. Build Prompt with top-8 examples                    │
│ 6. LLM (GPT-4o) generates code (n=1)                   │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ STAGE 3: VALIDATION & FEEDBACK                         │
├─────────────────────────────────────────────────────────┤
│ 7. 4-Layer Validation                                  │
│ 8. If failed → Feedback Loop → Retry (max 3 attempts)  │
└─────────────────────────────────────────────────────────┘
    ↓
Final Code + Metrics
```

---

## 📊 Evaluation Metrics

### Primary Research Metrics:
1. **pass@k**: Functional correctness (pass@1, pass@2, pass@3)
2. **Hallucination Rate**: % of generations with hallucinated APIs
3. **Validation Breakdown**: Per-layer pass rates

### Ablation Study Matrix:
| Configuration | Hybrid | Metadata | Cross-Encoder |
|--------------|--------|----------|---------------|
| Baseline | ✓ | ✗ | ✗ |
| +Metadata | ✓ | ✓ | ✗ |
| +Cross-Encoder | ✓ | ✗ | ✓ |
| **Full (Ours)** | **✓** | **✓** | **✓** |

---

## 🗂️ Project Structure

```
hallucination_mitigation/
├── config.py                      # Configuration & hyperparameters
├── requirements.txt               # Dependencies
│
├── vectordb/                      # Vector database & retrieval
│   ├── build_index.py            # Build FAISS index
│   └── retriever.py              # Hierarchical retrieval pipeline
│
├── retrieval/                     # Retrieval components
│   ├── metadata_filter.py        # Context-aware filtering
│   ├── cross_encoder_reranker.py # Cross-encoder reranking
│   └── query_reformulator.py    # LLM-based query reformulation
│
├── generation/                    # Code generation
│   ├── pipeline.py               # Main RAG pipeline
│   ├── llm_generator.py          # GPT-4o via SAP AI Core
│   └── prompt_builder.py         # Prompt construction
│
├── validation/                    # Multi-layer validation
│   ├── validator.py              # 4-layer validation orchestrator
│   ├── layer1_syntax.py          # Syntax checking
│   ├── layer2_api.py             # API correctness
│   ├── layer3_types.py           # Type checking
│   └── layer4_execution.py       # Execution testing
│
├── feedback/                      # Feedback loop
│   └── feedback_enrichment.py    # Error analysis & targeted queries
│
└── evaluation/                    # Metrics tracking
    └── metrics_tracker.py        # pass@k, hallucination rate, etc.
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.8+
- SAP AI Core access (for GPT-4o)
- Virtual environment

### 2. Install Dependencies
```bash
cd hallucination_mitigation
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Environment
Create `.env` file:
```bash
AICORE_SERVICE_KEY=<your-service-key-json>
AICORE_RESOURCE_GROUP=default
```

### 4. Build Vector Database
```bash
# (If not already built)
python vectordb/build_index.py
```

---

## 🎮 Usage

### Run Pipeline
```bash
python generation/pipeline.py
```

### Test Individual Components

**Test Metadata Filter:**
```bash
python retrieval/metadata_filter.py
```

**Test Cross-Encoder:**
```bash
python retrieval/cross_encoder_reranker.py
```

**Test Validation:**
```bash
python validation/validator.py
```

---

## ⚙️ Configuration

Key settings in `config.py`:

### Retrieval Settings
```python
RETRIEVAL_CONFIDENCE_THRESHOLD = 0.63  # Hybrid score threshold
HYBRID_SEARCH_TOP_K = 100              # Initial retrieval
METADATA_FILTER_TOP_K = 50             # After filtering
CROSS_ENCODER_TOP_K = 8                # Final examples to LLM
```

### Metadata Filtering
```python
ENABLE_METADATA_FILTERING = True       # Toggle for ablation
METADATA_MIN_CODE_LENGTH = 50
METADATA_MAX_CODE_LENGTH = 1000
```

### Cross-Encoder Reranking
```python
ENABLE_CROSS_ENCODER = True            # Toggle for ablation
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

### Generation Settings
```python
LLM_MODEL_NAME = "gpt-4o"             # SAP AI Core model
NUM_CANDIDATES = 1                     # Generate 1 code directly
MAX_PIPELINE_ATTEMPTS = 3              # Feedback loop attempts
```

---

## 📈 Key Features

### 1. Hierarchical Retrieval
- **Fast bi-encoder** (BGE) for initial retrieval
- **BM25 keyword matching** for exact matches
- **Cross-encoder** for accurate reranking
- **Metadata filtering** for context relevance

### 2. Confidence-Based Query Reformulation
- Calculates **retrieval confidence** from hybrid scores
- Threshold: 0.63 (empirically determined)
- Automatic query reformulation on low confidence
- Max 3 reformulation attempts

### 3. Hallucination Detection
- **Layer 2 validation** detects non-existent APIs
- Tracks hallucinated API names
- Measures hallucination rate and correction rate

### 4. Targeted Feedback Loop
- Analyzes validation errors
- Builds targeted queries for re-retrieval
- Priority: hallucination > execution > types > syntax

### 5. Comprehensive Metrics
- **pass@k**: Success within k attempts
- **Hallucination rate**: % with detected hallucinations
- **Correction rate**: % hallucinations fixed by feedback
- **Validation breakdown**: Per-layer success rates

---

## 🧪 Ablation Studies

Test each component independently by toggling config flags:

```python
# Test without metadata filtering
ENABLE_METADATA_FILTERING = False

# Test without cross-encoder
ENABLE_CROSS_ENCODER = False

# Test with different confidence thresholds
RETRIEVAL_CONFIDENCE_THRESHOLD = 0.60  # or 0.65, 0.70
```

Expected results:
- Each component improves retrieval quality
- Better retrieval → Lower hallucination rate
- pass@k increases with hierarchical retrieval

---

## 🔬 Research Results

### Metrics Output Example:
```python
{
    # Primary Metrics
    "pass@1": False,
    "pass@2": True,
    "pass@3": True,
    "hallucination_rate": 1.0,
    "validation_breakdown": {
        "layer1_syntax": True,
        "layer2_api": True,
        "layer3_types": True,
        "layer4_execution": True
    },
    
    # Secondary
    "hallucination_detected": True,
    "hallucination_corrected": True,  # Feedback worked!
    "total_attempts": 2,
    "total_time_seconds": 8.5
}
```

---

## 📝 Key Terminology

### Retrieval Confidence
- **Definition**: Average hybrid similarity score
- **Formula**: `mean(0.7 × dense_vector + 0.3 × BM25)`
- **Range**: 0-1 (higher = better)
- **Threshold**: 0.63

### Dense Vector (Bi-Encoder)
- **Model**: BAAI/bge-base-en-v1.5
- **Output**: 768-dim embeddings
- **Speed**: Fast (pre-computed)

### Cross-Encoder
- **Model**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **Purpose**: Accurate reranking
- **Speed**: Slower (query-time computation)

---

