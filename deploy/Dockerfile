FROM python:3.11-slim

WORKDIR /app

# deps for lxml / bs4 parsers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2 \
    libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

# python deps
COPY src/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# app code
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

VOLUME ["/data"]

CMD ["python", "run_daily.py"]
