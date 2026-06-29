# vectordb/build_index.py
"""
Build FAISS vector index from preprocessed documents.
Generates embeddings using BAAI/bge-base-en-v1.5 and stores in FAISS index.
"""

import json
import os
import sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    FAISS_INDEX_PATH,
    DOCUMENTS_PATH,
    BATCH_SIZE,
    NORMALIZE_EMBEDDINGS,
    USE_HNSW,
    HNSW_M,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH
)


def load_embedding_model():
    """
    Load sentence transformer embedding model.

    Returns:
        SentenceTransformer model
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"✓ Model loaded (dimension: {EMBEDDING_DIM})")
    return model


def generate_embeddings(documents: list, model: SentenceTransformer) -> np.ndarray:
    """
    Generate embeddings for all document queries.

    Args:
        documents: List of document dictionaries
        model: Embedding model

    Returns:
        NumPy array of embeddings (N x D)
    """
    print(f"\nGenerating embeddings for {len(documents)} documents...")

    # Extract query field from each document
    queries = [doc["query"] for doc in documents]

    # Generate embeddings in batches with progress bar
    embeddings = model.encode(
        queries,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        convert_to_numpy=True
    )

    # Ensure float32 type for FAISS
    embeddings = np.array(embeddings).astype('float32')

    print(f"✓ Generated embeddings: shape {embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings: np.ndarray, use_hnsw: bool = True) -> faiss.Index:
    """
    Build FAISS index from embeddings using HNSW for fast approximate search.

    Args:
        embeddings: NumPy array of embeddings (N x D)
        use_hnsw: Use HNSW index (default: True). If False, uses flat index.

    Returns:
        FAISS index
    """
    print(f"\nBuilding FAISS index...")

    dimension = embeddings.shape[1]
    num_vectors = embeddings.shape[0]

    # Normalize embeddings for cosine similarity (if not already normalized)
    if not NORMALIZE_EMBEDDINGS:
        faiss.normalize_L2(embeddings)
        print("✓ Embeddings normalized")

    if use_hnsw:
        # HNSW Index - Fast approximate search, scalable to millions
        # Parameters from config.py or defaults

        M = HNSW_M  # Connections per node
        print(f"Creating HNSW index with M={M}...")

        index = faiss.IndexHNSWFlat(dimension, M)

        # Set build parameters
        index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION

        # Add vectors to index
        print(f"Adding {num_vectors} vectors to HNSW index...")
        index.add(embeddings)

        # Set default search parameter (can be adjusted at query time)
        index.hnsw.efSearch = HNSW_EF_SEARCH

        print(f"✓ HNSW index built:")
        print(f"    - Dimension: {dimension}")
        print(f"    - Total vectors: {num_vectors}")
        print(f"    - Index type: IndexHNSWFlat")
        print(f"    - M (connections): {M}")
        print(f"    - efConstruction: {index.hnsw.efConstruction}")
        print(f"    - efSearch: {index.hnsw.efSearch}")
        print(f"    - Expected speedup: 6-10x faster than flat index")
        print(f"    - Expected accuracy: ~97-99%")

    else:
        # Flat Index - Exact search, slower for large datasets
        print("Creating flat index (exact search)...")
        index = faiss.IndexFlatIP(dimension)

        # Add vectors to index
        index.add(embeddings)

        print(f"✓ Flat index built:")
        print(f"    - Dimension: {dimension}")
        print(f"    - Total vectors: {num_vectors}")
        print(f"    - Index type: IndexFlatIP (Inner Product)")
        print(f"    - Accuracy: 100% exact")

    return index


def save_index_and_documents(index: faiss.Index, documents: list):
    """
    Save FAISS index and document store to disk.

    Args:
        index: FAISS index
        documents: List of document dictionaries
    """
    print(f"\nSaving index and documents...")

    # Create vectordb directory if it doesn't exist
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"✓ FAISS index saved to: {FAISS_INDEX_PATH}")

    # Save documents as JSON
    with open(DOCUMENTS_PATH, 'w') as f:
        json.dump(documents, f, indent=2)
    print(f"✓ Documents saved to: {DOCUMENTS_PATH}")

    # Print file sizes
    index_size_mb = os.path.getsize(FAISS_INDEX_PATH) / (1024 * 1024)
    docs_size_mb = os.path.getsize(DOCUMENTS_PATH) / (1024 * 1024)
    print(f"\nFile sizes:")
    print(f"  - Index: {index_size_mb:.2f} MB")
    print(f"  - Documents: {docs_size_mb:.2f} MB")


def test_index(index: faiss.Index, documents: list, model: SentenceTransformer):
    """
    Test the built index with a sample query and measure search speed.

    Args:
        index: FAISS index
        documents: List of documents
        model: Embedding model
    """
    print("\n" + "=" * 60)
    print("TESTING INDEX")
    print("=" * 60)

    test_query = "sort a list of dictionaries by key"
    print(f"Test query: '{test_query}'")

    # Generate query embedding
    query_embedding = model.encode(
        [test_query],
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        convert_to_numpy=True
    ).astype('float32')

    # Measure search time
    import time
    start_time = time.time()

    # Search index
    top_k = 5
    scores, indices = index.search(query_embedding, top_k)

    search_time = (time.time() - start_time) * 1000  # Convert to milliseconds

    print(f"\n⚡ Search time: {search_time:.2f}ms")
    print(f"\nTop {top_k} results:")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        doc = documents[idx]
        print(f"\n{rank}. Score: {score:.4f}")
        print(f"   Source: {doc['source']}")
        print(f"   Query: {doc['query'][:80]}...")
        print(f"   Code preview: {doc['code'][:100]}...")


def build_index(documents_path: str = None, use_hnsw: bool = True):
    """
    Main function to build vector index from documents.

    Args:
        documents_path: Path to preprocessed documents JSON file
        use_hnsw: Use HNSW index for fast approximate search (default: True)
    """
    print("=" * 60)
    print("BUILDING VECTOR INDEX")
    print("=" * 60)

    # Load documents
    if documents_path is None:
        documents_path = DOCUMENTS_PATH  # Use path from config (vectordb/documents.json)

    if not os.path.exists(documents_path):
        print(f"✗ Error: Documents file not found at {documents_path}")
        print("Please run dataset_loader.py first to generate documents.")
        return

    print(f"Loading documents from: {documents_path}")
    with open(documents_path, 'r') as f:
        documents = json.load(f)
    print(f"✓ Loaded {len(documents)} documents")

    # Load embedding model
    model = load_embedding_model()

    # Generate embeddings
    embeddings = generate_embeddings(documents, model)

    # Build FAISS index
    index = build_faiss_index(embeddings, use_hnsw=use_hnsw)

    # Save to disk
    save_index_and_documents(index, documents)

    # Test the index
    test_index(index, documents, model)

    print("\n" + "=" * 60)
    print("INDEX BUILDING COMPLETE!")
    print("=" * 60)
    print(f"You can now use the retriever with:")
    print(f"  - Index: {FAISS_INDEX_PATH}")
    print(f"  - Documents: {DOCUMENTS_PATH}")
    if use_hnsw:
        print(f"\n💡 HNSW Benefits:")
        print(f"  - 6-10x faster search than flat index")
        print(f"  - Scales to millions of vectors")
        print(f"  - ~97-99% accuracy (excellent for RAG)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build FAISS vector index from documents")
    parser.add_argument(
        "--documents",
        type=str,
        default=DOCUMENTS_PATH,  # Use path from config
        help="Path to preprocessed documents JSON file"
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Use flat index instead of HNSW (exact search, slower)"
    )

    args = parser.parse_args()
    build_index(args.documents, use_hnsw=not args.flat)
