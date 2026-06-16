import os
import re
import fitz  # PyMuPDF
from tqdm import tqdm

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
EXTRACTED_DIR = os.path.join(os.path.dirname(__file__), "data", "extracted")


def clean_text(text: str) -> str:
    """
    Remove common PDF noise:
    - excessive whitespace
    - page number patterns like "Page 1 of 20"
    - non-ASCII garbage characters
    - multiple blank lines
    """
    # remove "Page X of Y" patterns
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)

    # remove standalone page numbers
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # remove non-ASCII characters
    text = text.encode("ascii", errors="ignore").decode("ascii")

    # collapse multiple spaces into one
    text = re.sub(r"[ \t]+", " ", text)

    # collapse more than 2 newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_pdf(pdf_path: str, output_path: str, regulation_name: str):
    """
    Extract text from a PDF file page by page.
    Saves output as a .txt file with page markers.
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    extracted_pages = []

    for page_num in range(total_pages):
        page = doc[page_num]
        raw_text = page.get_text("text")
        cleaned = clean_text(raw_text)

        if cleaned:  # skip empty pages
            extracted_pages.append(
                f"[PAGE {page_num + 1}]\n{cleaned}"
            )

    doc.close()

    full_text = "\n\n".join(extracted_pages)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    return total_pages, len(extracted_pages)


def extract_all():
    os.makedirs(EXTRACTED_DIR, exist_ok=True)

    # find all PDFs in raw/
    pdf_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"❌ No PDFs found in {RAW_DIR}")
        print("Place your PDFs there and run again.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to extract\n")

    for filename in tqdm(pdf_files, desc="Extracting PDFs"):
        regulation_name = filename.replace(".pdf", "")
        pdf_path = os.path.join(RAW_DIR, filename)
        output_path = os.path.join(EXTRACTED_DIR, f"{regulation_name}.txt")

        if os.path.exists(output_path):
            print(f"⏭  Already extracted, skipping: {filename}")
            continue

        try:
            total_pages, extracted_pages = extract_pdf(
                pdf_path, output_path, regulation_name
            )
            print(
                f"✅ {filename} → "
                f"{extracted_pages}/{total_pages} pages extracted"
            )
        except Exception as e:
            print(f"❌ Failed to extract {filename}: {e}")


if __name__ == "__main__":
    extract_all()