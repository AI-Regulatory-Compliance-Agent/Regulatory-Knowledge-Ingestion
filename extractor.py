import os
import re
import io
import fitz                  # PyMuPDF  — native PDF text extraction + page rendering
import pytesseract           # OCR engine wrapper (calls Tesseract under the hood)
from PIL import Image        # Pillow  — converts raw pixmap bytes into an image object
from tqdm import tqdm        # progress bar for the terminal

# langdetect is optional — install with: pip install langdetect
# If missing, language filtering is skipped entirely (all pages are kept)
try:
    from langdetect import detect
except ImportError:
    detect = None

# ── Directory paths ────────────────────────────────────────────────────────────
# os.path.dirname(__file__)  →  folder that contains THIS script
# os.path.join(...)          →  builds OS-safe paths (handles / vs \ automatically)
RAW_DIR       = os.path.join(os.path.dirname(__file__), "data", "raw")
EXTRACTED_DIR = os.path.join(os.path.dirname(__file__), "data", "extracted")

# ── Constants ─────────────────────────────────────────────────────────────────
# If a page yields fewer than this many characters via native extraction,
# we treat it as a scanned page and fall back to OCR.
# 50 is a safe floor — real text pages almost always exceed it.
MIN_NATIVE_TEXT_LEN = 50

# Tesseract DPI for page rasterisation.
# 300 DPI is the recommended minimum for accurate OCR results.
OCR_DPI = 300


# ══════════════════════════════════════════════════════════════════════════════
# HELPER 1 — clean_text
# Purpose : strip PDF noise from any text string (native or OCR output)
# Input   : raw text string
# Output  : cleaned text string
# ══════════════════════════════════════════════════════════════════════════════
def clean_text(text: str) -> str:
    # Remove "Page 1 of 20" / "page 3 of 7" style markers left by some PDFs
    # re.IGNORECASE  →  matches "Page", "PAGE", "page", etc.
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)

    # Remove lines that contain ONLY a number (standalone page numbers)
    # re.MULTILINE  →  ^ and $ match start/end of each line, not the whole string
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # Drop every character outside the printable ASCII range (0x20–0x7E)
    # encode("ascii", errors="ignore")  →  silently discards non-ASCII bytes
    # .decode("ascii")                  →  turns the bytes back into a string
    text = text.encode("ascii", errors="ignore").decode("ascii")

    # Collapse runs of spaces or tabs into a single space
    # [ \t]+  →  one or more spaces OR tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3+ consecutive newlines down to exactly 2
    # (keeps paragraph breaks but removes large blank gaps)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace from the whole block
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# HELPER 2 — is_english
# Purpose : detect whether a block of text is written in English
#           used to skip non-English pages (e.g. Hindi annexures in Indian laws)
# Input   : any text string
# Output  : True  → text is English (or langdetect unavailable)
#           False → text is a different language; page should be skipped
# Note    : Set SKIP_NON_ENGLISH = False in extract_pdf() if your documents
#           intentionally contain bilingual content you want to keep
# ══════════════════════════════════════════════════════════════════════════════
def is_english(text: str) -> bool:
    if detect is None:
        # langdetect not installed — skip filtering, keep all pages
        return True
    try:
        return detect(text) == "en"
    except Exception:
        # langdetect raises if text is too short or ambiguous → keep the page
        return True


# ══════════════════════════════════════════════════════════════════════════════
# HELPER 3 — is_scanned_page
# Purpose : decide whether a page needs OCR or not
# Input   : a fitz.Page object
# Output  : True  → page is scanned / no usable text layer; OCR needed
#           False → page has enough native text; OCR not needed
# ══════════════════════════════════════════════════════════════════════════════
def is_scanned_page(page: fitz.Page) -> bool:
    # page.get_text("text")  →  extracts the embedded text layer (instant, no OCR)
    # .strip()               →  removes surrounding whitespace before measuring
    native_text = page.get_text("text").strip()

    # If the native text is shorter than our threshold, the page is likely
    # an image scan with no meaningful text layer → needs OCR
    return len(native_text) < MIN_NATIVE_TEXT_LEN


# ══════════════════════════════════════════════════════════════════════════════
# HELPER 4 — ocr_page
# Purpose : rasterise a PDF page to a PNG image, then run Tesseract OCR on it
# Input   : a fitz.Page object
# Output  : OCR'd text string (quality depends on scan resolution)
# ══════════════════════════════════════════════════════════════════════════════
def ocr_page(page: fitz.Page) -> str:
    # Step 1 — Render the page to a pixel map at OCR_DPI (default 300 DPI)
    # Higher DPI = better OCR accuracy but more RAM and slower processing
    pix = page.get_pixmap(dpi=OCR_DPI)

    # Step 2 — Convert the raw pixmap bytes to a PNG byte string
    # tobytes("png")  →  encodes the pixmap as a lossless PNG in memory
    img_bytes = pix.tobytes("png")

    # Step 3 — Wrap the raw bytes in a BytesIO buffer so PIL can read it
    # PIL.Image.open() expects a file-like object, not a plain bytes value
    image = Image.open(io.BytesIO(img_bytes))

    # Step 4 — Run Tesseract OCR on the PIL image
    # lang="eng"    →  use the English language model
    # --psm 6       →  Page Segmentation Mode 6: assume a single uniform block
    #                  of text. Works well for full-page regulatory documents.
    #                  Other useful modes: --psm 3 (auto), --psm 4 (single column)
    ocr_text = pytesseract.image_to_string(image, lang="eng", config="--psm 6")

    return ocr_text


# ══════════════════════════════════════════════════════════════════════════════
# HELPER 5 — extract_page_text
# Purpose : single entry-point for getting text from ONE page
#           tries native extraction first; falls back to OCR automatically
# Input   : a fitz.Page object
# Output  : (text, used_ocr) tuple
#           text     — best available text for this page
#           used_ocr — True if OCR was used (lets the caller track stats)
# ══════════════════════════════════════════════════════════════════════════════
def extract_page_text(page: fitz.Page) -> tuple[str, bool]:
    if is_scanned_page(page):
        # Page has no usable text layer → OCR path (slower)
        return ocr_page(page), True
    else:
        # Page has native text → fast path, no OCR needed
        return page.get_text("text"), False


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION 1 — extract_pdf
# Purpose : process a single PDF file end-to-end
#           iterates every page, chooses native vs OCR per page,
#           optionally skips non-English pages, cleans the text,
#           and writes everything to a .txt file
# Inputs  : pdf_path         — absolute path to the source .pdf
#           output_path      — absolute path for the output .txt
#           regulation_name  — human-readable label used in log output
#           skip_non_english — if True, drops pages detected as non-English
#                              set to False for bilingual documents
# Returns : (total_pages, extracted_pages, ocr_pages) tuple of ints
# ══════════════════════════════════════════════════════════════════════════════
def extract_pdf(
    pdf_path: str,
    output_path: str,
    regulation_name: str,
    skip_non_english: bool = True,
) -> tuple[int, int, int]:
    # fitz.open()  →  loads the PDF into memory; returns a Document object
    doc = fitz.open(pdf_path)

    # len(doc)  →  total number of pages in the document
    total_pages = len(doc)

    extracted_pages = []   # will hold formatted strings, one per non-empty page
    ocr_page_count  = 0    # how many pages required OCR (reported in summary)
    skipped_lang    = 0    # how many pages were dropped due to language filtering

    for page_num in range(total_pages):
        # doc[page_num]  →  retrieves a fitz.Page object (0-indexed)
        page = doc[page_num]

        # Get the best available text for this page; also know whether OCR was used
        raw_text, used_ocr = extract_page_text(page)

        # Remove noise, collapse whitespace, etc.
        cleaned = clean_text(raw_text)

        # Skip pages that are truly empty after cleaning
        if not cleaned:
            continue

        # Optionally skip non-English pages (e.g. Hindi sections in Indian laws)
        # Warning: disable this if your docs are intentionally multilingual
        if skip_non_english and not is_english(cleaned):
            skipped_lang += 1
            continue

        # Page passed all filters — add it to the output
        extracted_pages.append(
            f"[PAGE {page_num + 1}]\n{cleaned}"
            # page_num is 0-indexed → +1 makes the marker human-readable
        )
        if used_ocr:
            ocr_page_count += 1    # track OCR usage for the summary log

    doc.close()                    # release file handle and memory

    # Join all page blocks with a blank line separator for readability
    full_text = "\n\n".join(extracted_pages)

    # Write the combined text to disk in UTF-8
    # "w" mode creates the file if it doesn't exist; overwrites if it does
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    # Return stats for the caller to display
    return total_pages, len(extracted_pages), ocr_page_count, skipped_lang


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION 2 — extract_all
# Purpose : orchestrate extraction across all PDFs in the raw/ directory
#           handles skipping already-processed files, error catching,
#           and progress display
# ══════════════════════════════════════════════════════════════════════════════
def extract_all():
    # Create the output directory if it doesn't already exist
    # exist_ok=True  →  no error if the folder is already there
    os.makedirs(EXTRACTED_DIR, exist_ok=True)

    # List every file in raw/ whose name ends with ".pdf" (case-sensitive)
    pdf_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"No PDFs found in {RAW_DIR}")
        print("Place your PDFs there and run again.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to extract\n")

    # tqdm wraps the list and prints a live progress bar in the terminal
    for filename in tqdm(pdf_files, desc="Extracting PDFs"):
        # Strip the ".pdf" extension to get a clean document name
        regulation_name = filename.replace(".pdf", "")

        pdf_path    = os.path.join(RAW_DIR,       filename)
        output_path = os.path.join(EXTRACTED_DIR, f"{regulation_name}.txt")

        # Skip files that were already extracted in a previous run
        if os.path.exists(output_path):
            print(f"Already extracted, skipping: {filename}")
            continue

        try:
            total, extracted, ocr_pages, skipped_lang = extract_pdf(
                pdf_path, output_path, regulation_name
            )
            print(
                f"{filename} → "
                f"{extracted}/{total} pages extracted "
                f"({ocr_pages} via OCR, {skipped_lang} non-English skipped)"
                # e.g. "report.pdf → 18/20 pages extracted (6 via OCR, 1 non-English skipped)"
            )
        except Exception as e:
            # Catch-all: if any PDF fails (corrupt, password-protected, etc.)
            # we log the error and keep going rather than crashing the whole run
            print(f"Failed to extract {filename}: {e}")


# ── Entry point ────────────────────────────────────────────────────────────────
# This block only runs when the script is executed directly (python extract_pdf.py)
# It does NOT run when the file is imported as a module by another script
if __name__ == "__main__":
    extract_all()