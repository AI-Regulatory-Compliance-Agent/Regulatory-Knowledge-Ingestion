FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# ── Install PyTorch CPU-only FIRST ───────────────────────────
# Prevents sentence-transformers from pulling 2.5GB CUDA version.
# Ingestion only needs CPU for embedding — no GPU required.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# ── Install remaining dependencies ───────────────────────────
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

CMD ["python", "ingestor.py"]