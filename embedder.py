import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

CHUNKS_DIR = os.path.join(os.path.dirname(__file__), "data", "chunks")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def load_model():
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    print("✅ Model loaded\n")
    return model


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

    return chunks


def embed_all(model: SentenceTransformer = None) -> dict[str, list[dict]]:
    """
    Returns dict of regulation_name → embedded chunks.
    Called by ingestor.py directly so we don't write vectors to disk.
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

        print(f"✅ {regulation_name} → {len(embedded)} chunks embedded")

    return all_embedded


if __name__ == "__main__":
    # when run standalone, just embed and print summary
    results = embed_all()
    print(f"\nTotal regulations embedded: {len(results)}")
    for name, chunks in results.items():
        print(f"  {name}: {len(chunks)} chunks, "
              f"vector dim: {len(chunks[0]['vector'])}")