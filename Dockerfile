# CPU image for the FastAPI service.
# Weights stay out of the image: first start downloads
# newnus/justid-rechtsgebieden and newnus/justid-bijzondere-kenmerken,
# plus the embedding and reranker models when the RAG index is mounted.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-serve.txt

COPY app.py predict.py config.py ./
COPY rag/ ./rag/

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
