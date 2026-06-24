# 📄 ComplianceAI - Ingestion Pipeline

> One-shot document processing pipeline that converts government regulation PDFs into searchable vector embeddings stored in Qdrant. Handles downloading, text extraction (with OCR fallback), chunking, embedding, and vector storage.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC382D?logo=data:image/svg+xml;base64,&logoColor=white)](https://qdrant.tech)
[![License: GPL v2](https://img.shields.io/badge/License-GPL_v2-blue.svg)](../backend/LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Pipeline Flow](#pipeline-flow)
- [Pipeline Components](#pipeline-components)
- [Data Flow](#data-flow)
- [Output Format](#output-format)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Local Development](#local-development)
- [Tech Stack](#tech-stack)

---

## Overview

The ingestion pipeline is a **one-shot process** that runs once to populate the Qdrant vector database with regulation document embeddings. Once the vectors are stored, the pipeline does not need to run again unless new regulations are added.

The pipeline serves as the **data foundation** for the entire ComplianceAI system — the backend's RAG (Retrieval-Augmented Generation) agents query Qdrant to ground their analysis in actual regulation text.

---

## Pipeline Flow

```
📄 Government Regulation PDFs
        │
        ▼
┌──────────────────┐
│  1. downloader   │  Downloads PDFs from configured URLs
│     downloader.py│  → data/raw/*.pdf
└────────┬─────────┘
         ▼
┌──────────────────┐
│  2. extractor    │  Extracts text from each PDF page
│     extractor.py │  PyMuPDF + Tesseract OCR fallback
│                  │  → data/extracted/{name}.json
└────────┬─────────┘
         ▼
┌──────────────────┐
│  3. chunker      │  Splits text into 500-token chunks
│     chunker.py   │  with 50-token overlap
│                  │  → data/chunks/{name}_chunks.json
└────────┬─────────┘
         ▼
┌──────────────────┐
│  4. embedder     │  Generates 384-dim vectors using
│     embedder.py  │  all-MiniLM-L6-v2 (CPU-only)
└────────┬─────────┘
         ▼
┌──────────────────┐
│  5. ingestor     │  Uploads vectors + metadata to Qdrant
│     ingestor.py  │  in batches of 100
└────────┬─────────┘
         ▼
   🗄️ Qdrant Collection: "regulations"
```

---

## Pipeline Components

### 1. Downloader (`downloader.py`)

Downloads regulation PDFs from configured URLs to the `data/raw/` directory.

- Skips files that already exist locally
- Handles HTTP errors gracefully
- Outputs: `data/raw/*.pdf`

### 2. Extractor (`extractor.py`)

Extracts text from PDF files page-by-page using a dual-engine approach:

- **Primary**: PyMuPDF (`fitz`) for digital PDFs with embedded text
- **Fallback**: Tesseract OCR for scanned documents or pages with no extractable text
- Language detection with `langdetect` to filter non-English content
- Outputs: `data/extracted/{regulation_name}.json`

```json
{
  "regulation_name": "GDPR",
  "source_file": "gdpr.pdf",
  "pages": [
    {"page_number": 1, "text": "...extracted text..."},
    {"page_number": 2, "text": "..."}
  ]
}
```

### 3. Chunker (`chunker.py`)

Splits extracted text into overlapping chunks optimized for embedding and retrieval:

- **Chunk size**: 500 tokens
- **Overlap**: 50 tokens (maintains context continuity)
- Uses `langchain-text-splitters` for intelligent chunking
- Outputs: `data/chunks/{regulation_name}_chunks.json`

```json
[
  {
    "chunk_id": "GDPR_chunk_001",
    "regulation_name": "GDPR",
    "source_file": "gdpr.pdf",
    "page_number": 1,
    "text": "...chunk text...",
    "token_count": 487
  }
]
```

### 4. Embedder (`embedder.py`)

Generates dense vector representations of each text chunk:

- **Model**: `all-MiniLM-L6-v2` (Sentence Transformers)
- **Vector dimensions**: 384
- **Runtime**: CPU-only (no GPU required)
- Pre-downloads the model at Docker build time for offline operation

### 5. Ingestor (`ingestor.py`)

The orchestrator that runs the full pipeline and stores results in Qdrant:

- Waits for Qdrant to be ready (health check with retries)
- Creates the `regulations` collection if it doesn't exist
- Uploads vectors in batches of 100 for efficiency
- Uses cosine similarity distance metric

---

## Data Flow

```
data/
├── raw/                    # Source PDFs (input)
│   ├── gdpr.pdf
│   ├── ccpa.pdf
│   └── hipaa.pdf
│
├── extracted/              # Extracted text (intermediate)
│   ├── gdpr.json
│   ├── ccpa.json
│   └── hipaa.json
│
└── chunks/                 # Chunked text (intermediate)
    ├── gdpr_chunks.json
    ├── ccpa_chunks.json
    └── hipaa_chunks.json

         ↓ (embedded + stored)

Qdrant Collection: "regulations"
```

---

## Output Format

Each vector point stored in Qdrant has:

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Unique point identifier |
| `vector` | `float[384]` | Dense embedding from all-MiniLM-L6-v2 |
| `payload.chunk_id` | string | Chunk identifier (e.g., `GDPR_chunk_001`) |
| `payload.regulation_name` | string | Name of the regulation |
| `payload.source_file` | string | Original PDF filename |
| `payload.page_number` | int | Source page number |
| `payload.text` | string | Raw text content |
| `payload.token_count` | int | Number of tokens in the chunk |

---

## Project Structure

```
ingestion/
├── Dockerfile              # Python 3.11-slim + Tesseract OCR + CPU PyTorch
├── requirements.txt        # Pinned dependencies
│
├── ingestor.py             # Main orchestrator (entry point)
├── downloader.py           # Step 1: PDF download
├── extractor.py            # Step 2: Text extraction (PyMuPDF + OCR)
├── chunker.py              # Step 3: Text chunking
└── embedder.py             # Step 4: Vector embedding
```

---

## Configuration

Environment variables (loaded from `Infrastructure/.env`):

| Variable | Description | Default |
|----------|-------------|---------|
| `QDRANT_HOST` | Qdrant server hostname | `localhost` |
| `QDRANT_PORT` | Qdrant server port | `6333` |
| `QDRANT_COLLECTION` | Collection name for regulation vectors | `regulations` |
| `EMBEDDING_MODEL` | Sentence transformer model name | `all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Tokens per chunk | `500` |
| `CHUNK_OVERLAP` | Token overlap between chunks | `50` |

---

## Local Development

### With Docker (Recommended)

```bash
cd Infrastructure
docker compose up --build ingestion
```

The ingestion container runs once and exits when complete. Check logs:

```bash
docker compose logs ingestion
```

### Standalone

```bash
cd ingestion

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install CPU-only PyTorch first
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
pip install -r requirements.txt

# Ensure Tesseract OCR is installed on your system
# Windows: https://github.com/UB-Mannheim/tesseract/wiki

# Ensure Qdrant is running
# docker run -p 6333:6333 qdrant/qdrant:latest

# Place PDFs in data/raw/
# Run the pipeline
python ingestor.py
```

### Re-running Ingestion

To re-ingest with new documents:

```bash
# Option 1: Run the ingestion container again
docker compose run --rm ingestion

# Option 2: Delete the Qdrant collection first for a clean re-ingest
# (Qdrant Dashboard → Collections → Delete → Re-run ingestion)
```

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| **Runtime** | Python 3.11 |
| **PDF Extraction** | PyMuPDF (fitz) |
| **OCR** | Tesseract via pytesseract |
| **Text Splitting** | langchain-text-splitters |
| **Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`) |
| **Vector Storage** | Qdrant Client |
| **Image Processing** | Pillow |
| **Language Detection** | langdetect |
| **Progress Bars** | tqdm |
