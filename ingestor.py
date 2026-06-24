import os
import uuid
import time
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    SparseVectorParams,
    SparseIndexParams,
    SparseVector,
)
from dotenv import load_dotenv
from downloader import download_all
from extractor import extract_all
from chunker import chunk_all
from embedder import embed_all, load_model

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "regulations")
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 output dimension


def wait_for_qdrant(retries: int = 10, delay: int = 3):
    url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/healthz"
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                print(f"✅ Qdrant is ready (attempt {attempt})")
                return
        except Exception:
            pass
        print(f"⏳ Waiting for Qdrant... attempt {attempt}/{retries}")
        time.sleep(delay)
    raise RuntimeError("❌ Qdrant did not become ready in time")


def get_qdrant_client() -> QdrantClient:
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print(f"✅ Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    return client


def create_collection(client: QdrantClient):
    """
    Create the regulations collection with named vectors config for hybrid search.

    Vectors:
      - "dense": 384-dim cosine similarity (SentenceTransformer all-MiniLM-L6-v2)
      - "sparse": BM25-style keyword matching (hash-based token IDs)

    If the collection exists with OLD (unnamed) vector config, it is deleted
    and recreated with the new named vector schema. This is necessary because
    Qdrant does not support migrating from unnamed to named vectors in place.
    """
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        # Check if the collection uses the new named vector config
        collection_info = client.get_collection(COLLECTION_NAME)
        vectors_config = collection_info.config.params.vectors

        # If vectors_config is a VectorParams (unnamed), migration needed
        if isinstance(vectors_config, VectorParams):
            print(f"⚠️  Collection '{COLLECTION_NAME}' has old (unnamed) vector config")
            print(f"   Deleting and recreating with named vectors for hybrid search...")
            client.delete_collection(COLLECTION_NAME)
        elif isinstance(vectors_config, dict) and "dense" in vectors_config:
            print(f"⏭  Collection '{COLLECTION_NAME}' already has named vector config, skipping creation")
            return
        else:
            # Unknown config format — recreate to be safe
            print(f"⚠️  Collection '{COLLECTION_NAME}' has unexpected config, recreating...")
            client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        }
    )
    print(f"✅ Collection '{COLLECTION_NAME}' created (dense + sparse vectors)")


def store_chunks(client: QdrantClient, embedded_chunks: dict[str, list[dict]]):
    """
    Store embedded chunks in Qdrant with both dense and sparse vectors.

    Each point has:
      - Named vector "dense": 384-dim embedding from SentenceTransformer
      - Named sparse vector "sparse": BM25-style token weights
      - Payload: chunk_id, regulation_name, source_file, page_number, text, token_count
    """
    total_stored = 0

    for regulation_name, chunks in embedded_chunks.items():
        print(f"\n📦 Storing: {regulation_name} ({len(chunks)} chunks)")

        points = []
        for chunk in chunks:
            # Build sparse vector if available
            sparse_vectors = {}
            if chunk.get("sparse_indices"):
                sparse_vectors["sparse"] = SparseVector(
                    indices=chunk["sparse_indices"],
                    values=chunk["sparse_values"]
                )

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": chunk["vector"],
                },
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "regulation_name": chunk["regulation_name"],
                    "source_file": chunk["source_file"],
                    "page_number": chunk["page_number"],
                    "text": chunk["text"],
                    "token_count": chunk["token_count"],
                    "source_type": "regulation",
                }
            )
            points.append(point)

        # upload in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i: i + batch_size]
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch
            )

        total_stored += len(chunks)
        print(f"✅ {regulation_name} → stored {len(chunks)} chunks (dense + sparse)")

    return total_stored


def run():
    print("=" * 50)
    print("  AI Regulatory Compliance — Ingestion Pipeline")
    print("  (Hybrid Search: Dense + Sparse Vectors)")
    print("=" * 50)

    # Wait for Qdrant to be ready before doing anything
    print("\n[0/5] Waiting for Qdrant...")
    wait_for_qdrant()

    # Step 1 — Download (skips if no URLs configured)
    print("\n[1/5] Downloading PDFs...")
    download_all()

    # Step 2 — Extract text from PDFs
    print("\n[2/5] Extracting text from PDFs...")
    extract_all()

    # Step 3 — Chunk text
    print("\n[3/5] Chunking extracted text...")
    chunk_all()

    # Step 4 — Embed chunks (dense + sparse)
    print("\n[4/5] Embedding chunks (dense + sparse vectors)...")
    model = load_model()
    embedded_chunks = embed_all(model)

    if not embedded_chunks:
        print("❌ No chunks to store. Exiting.")
        return

    # Step 5 — Store in Qdrant (with named vectors)
    print("\n[5/5] Storing in Qdrant (hybrid: dense + sparse)...")
    client = get_qdrant_client()
    create_collection(client)
    total = store_chunks(client, embedded_chunks)

    print("\n" + "=" * 50)
    print(f"✅ Ingestion complete — {total} chunks stored in Qdrant")
    print(f"   Collection: '{COLLECTION_NAME}' (hybrid search enabled)")
    print(f"   Dense vectors: {VECTOR_SIZE}-dim cosine similarity")
    print(f"   Sparse vectors: BM25-style keyword matching")
    print(f"   Dashboard:  http://localhost:6333/dashboard")
    print("=" * 50)


if __name__ == "__main__":
    run()