import os
import uuid
import time
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
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
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        print(f"⏭  Collection '{COLLECTION_NAME}' already exists, skipping creation")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )
    print(f"✅ Collection '{COLLECTION_NAME}' created")


def store_chunks(client: QdrantClient, embedded_chunks: dict[str, list[dict]]):
    total_stored = 0

    for regulation_name, chunks in embedded_chunks.items():
        print(f"\n📦 Storing: {regulation_name} ({len(chunks)} chunks)")

        points = []
        for chunk in chunks:
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=chunk["vector"],
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "regulation_name": chunk["regulation_name"],
                    "source_file": chunk["source_file"],
                    "page_number": chunk["page_number"],
                    "text": chunk["text"],
                    "token_count": chunk["token_count"]
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
        print(f"✅ {regulation_name} → stored {len(chunks)} chunks")

    return total_stored


def run():
    print("=" * 50)
    print("  AI Regulatory Compliance — Ingestion Pipeline")
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

    # Step 4 — Embed chunks
    print("\n[4/5] Embedding chunks...")
    model = load_model()
    embedded_chunks = embed_all(model)

    if not embedded_chunks:
        print("❌ No chunks to store. Exiting.")
        return

    # Step 5 — Store in Qdrant
    print("\n[5/5] Storing in Qdrant...")
    client = get_qdrant_client()
    create_collection(client)
    total = store_chunks(client, embedded_chunks)

    print("\n" + "=" * 50)
    print(f"✅ Ingestion complete — {total} chunks stored in Qdrant")
    print(f"   Collection: '{COLLECTION_NAME}'")
    print(f"   Dashboard:  http://localhost:6333/dashboard")
    print("=" * 50)


if __name__ == "__main__":
    run()