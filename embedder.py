import os
import re
import math
import json
from collections import Counter
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

CHUNKS_DIR = os.path.join(os.path.dirname(__file__), "data", "chunks")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Stopwords for sparse tokenization ───────────────────────
# Must match the stopword set in backend/app/tools/qdrant_search.py
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "not", "no", "nor", "so", "if", "as",
})


def load_model():
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    print("✅ Model loaded\n")
    return model


def generate_sparse_vector(text: str) -> tuple[list[int], list[float]]:
    """
    Generate a BM25-style sparse vector from text.

    Uses hash-based token IDs for deterministic mapping without needing
    a pre-built vocabulary. Tokens are lowercased and filtered for stopwords.

    This function MUST produce identical token IDs as the one in
    backend/app/tools/qdrant_search.py — they share the same logic.

    Returns:
        Tuple of (indices, values) for Qdrant SparseVector.
        indices: list of token hash IDs (positive integers)
        values: list of TF-IDF-style weights
    """
    # Tokenize: lowercase, split on non-alphanumeric, filter stopwords + short tokens
    tokens = re.findall(r'[a-z0-9]+(?:\([^)]*\))?', text.lower())
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]

    if not tokens:
        return [], []

    # Count term frequencies
    tf = Counter(tokens)
    total_tokens = len(tokens)

    indices = []
    values = []

    for token, count in tf.items():
        # Hash token to a positive integer index (deterministic)
        token_id = abs(hash(token)) % (2**31 - 1)

        # TF weight: log(1 + count/total) — normalized term frequency
        weight = math.log(1 + count / total_tokens)

        indices.append(token_id)
        values.append(round(weight, 6))

    return indices, values


def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> list[dict]:
    texts = [chunk["text"] for chunk in chunks]

    # batch embed all chunks at once (faster than one by one)
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    for i, chunk in enumerate(chunks):
        chunk["vector"] = vectors[i].tolist()  # numpy → python list for JSON

        # Generate sparse vector for hybrid search
        sparse_indices, sparse_values = generate_sparse_vector(chunk["text"])
        chunk["sparse_indices"] = sparse_indices
        chunk["sparse_values"] = sparse_values

    return chunks


def embed_all(model: SentenceTransformer = None) -> dict[str, list[dict]]:
    """
    Returns dict of regulation_name → embedded chunks.
    Called by ingestor.py directly so we don't write vectors to disk.
    Each chunk now includes both dense vector and sparse vector data.
    """
    if model is None:
        model = load_model()

    chunk_files = [
        f for f in os.listdir(CHUNKS_DIR)
        if f.endswith("_chunks.json")
    ]

    if not chunk_files:
        print(f"❌ No chunk files found in {CHUNKS_DIR}")
        print("Run chunker.py first.")
        return {}

    all_embedded = {}

    for filename in chunk_files:
        regulation_name = filename.replace("_chunks.json", "")
        filepath = os.path.join(CHUNKS_DIR, filename)

        print(f"\n📐 Embedding: {regulation_name}")

        with open(filepath, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        embedded = embed_chunks(chunks, model)
        all_embedded[regulation_name] = embedded

        print(f"✅ {regulation_name} → {len(embedded)} chunks embedded "
              f"(dense + sparse)")

    return all_embedded


if __name__ == "__main__":
    # when run standalone, just embed and print summary
    results = embed_all()
    print(f"\nTotal regulations embedded: {len(results)}")
    for name, chunks in results.items():
        print(f"  {name}: {len(chunks)} chunks, "
              f"vector dim: {len(chunks[0]['vector'])}, "
              f"sparse tokens: {len(chunks[0]['sparse_indices'])}")