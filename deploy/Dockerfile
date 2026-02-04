FROM python:3.11-slim

WORKDIR /app

# Predictable runtime, fewer edge-cases with logs/buffers and wheels
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=300

# System deps:
# - curl/ca-certificates: диагностика и TLS
# - libxml2/libxslt1.1: runtime для lxml
# - build-essential + libxml2-dev + libxslt1-dev: сборка wheels, если нет готовых
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    build-essential \
    libxml2 \
    libxslt1.1 \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps (отдельным слоем для кеша)
COPY src/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
 && pip install --retries=10 -r /app/requirements.txt

# App code
COPY src/ /app/

# data output (snippets, meta, faiss index) goes to mounted volume
ENV OUT_DIR=/data/knowledge_base

# daily schedule (seconds)
ENV INTERVAL_SEC=86400

# crawling params
ENV BASE_URL=http://sniprf.ru/snip
ENV LIMIT_PAGES=60
ENV SLEEP_SEC=0.4

# chunking params
ENV CHUNK_CHARS=1600
ENV CHUNK_OVERLAP=250

# embeddings params
ENV EMB_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
ENV EMB_BATCH_SIZE=32

# Optional: where FAISS index lives (единый путь для parser и bot)
ENV FAISS_DIR=/data/knowledge_base/faiss

# One mount root; OUT_DIR/FAISS_DIR are внутри него
VOLUME ["/data/knowledge_base"]

CMD ["python", "run_daily.py"]
