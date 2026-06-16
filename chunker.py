import os
import json
import re
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

EXTRACTED_DIR = os.path.join(os.path.dirname(__file__), "data", "extracted")
CHUNKS_DIR = os.path.join(os.path.dirname(__file__), "data", "chunks")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))


def parse_page_number(text_before_chunk: str) -> int:
    """
    Try to find the most recent [PAGE X] marker
    before this chunk to track source page.
    """
    matches = re.findall(r"\[PAGE (\d+)\]", text_before_chunk)
    if matches:
        return int(matches[-1])
    return 1


def chunk_file(txt_path: str, regulation_name: str) -> list[dict]:
    with open(txt_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    raw_chunks = splitter.split_text(full_text)
    chunks = []

    for i, chunk_text in enumerate(raw_chunks):
        # find position of this chunk in the full text to detect page number
        chunk_start = full_text.find(chunk_text[:50])
        text_before = full_text[:chunk_start] if chunk_start > 0 else ""
        page_number = parse_page_number(text_before)

        # clean up [PAGE X] markers that ended up inside chunks
        clean_chunk = re.sub(r"\[PAGE \d+\]", "", chunk_text).strip()

        if not clean_chunk:
            continue

        chunks.append({
            "chunk_id": f"{regulation_name}_chunk_{i:04d}",
            "regulation_name": regulation_name,
            "source_file": f"{regulation_name}.pdf",
            "page_number": page_number,
            "text": clean_chunk,
            "token_count": len(clean_chunk.split())
        })

    return chunks


def chunk_all():
    os.makedirs(CHUNKS_DIR, exist_ok=True)

    txt_files = [f for f in os.listdir(EXTRACTED_DIR) if f.endswith(".txt")]

    if not txt_files:
        print(f"❌ No .txt files found in {EXTRACTED_DIR}")
        print("Run extractor.py first.")
        return

    print(f"Found {len(txt_files)} file(s) to chunk\n")

    for filename in tqdm(txt_files, desc="Chunking"):
        regulation_name = filename.replace(".txt", "")
        txt_path = os.path.join(EXTRACTED_DIR, filename)
        output_path = os.path.join(CHUNKS_DIR, f"{regulation_name}_chunks.json")

        if os.path.exists(output_path):
            print(f"⏭  Already chunked, skipping: {filename}")
            continue

        try:
            chunks = chunk_file(txt_path, regulation_name)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)

            print(f"✅ {filename} → {len(chunks)} chunks")

        except Exception as e:
            print(f"❌ Failed to chunk {filename}: {e}")


if __name__ == "__main__":
    chunk_all()