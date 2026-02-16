FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# deps: lxml + faiss + диагностика
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2 \
    libxslt1.1 \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY src/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --default-timeout=300 --retries=10 -r /app/requirements.txt

COPY src/ /app/

ENV OUT_DIR=/data/knowledge_base \
    INTERVAL_SEC=86400 \
    BASE_URL=http://sniprf.ru/snip \
    LIMIT_PAGES=60 \
    SLEEP_SEC=0.4 \
    CHUNK_CHARS=1600 \
    CHUNK_OVERLAP=250 \
    EMB_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
    EMB_BATCH_SIZE=32 \
    HF_HOME=/data/.hf \
    TRANSFORMERS_CACHE=/data/.hf

VOLUME ["/data"]

CMD ["python", "run_daily.py"]
