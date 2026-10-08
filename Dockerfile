# Maintenance RAG assistant - Streamlit UI by default; see docker-compose.yml for the API too.
#
#   docker build -t rag-maintenance-qa .
#   docker run -p 8501:8501 --env-file .env rag-maintenance-qa
#
# The embedding and reranker models are baked into the image (PRELOAD_MODELS=true),
# so the container starts without a ~1 GB download. Build with
# --build-arg PRELOAD_MODELS=false for a smaller image that downloads them on first use.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app

# CPU-only PyTorch first: otherwise sentence-transformers pulls the multi-GB CUDA build
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

ARG PRELOAD_MODELS=true
ARG EMBED_MODEL=intfloat/multilingual-e5-small
ARG RERANK_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
ENV EMBED_MODEL=${EMBED_MODEL} RERANK_MODEL=${RERANK_MODEL}
RUN if [ "$PRELOAD_MODELS" = "true" ]; then \
      python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('${EMBED_MODEL}'); CrossEncoder('${RERANK_MODEL}')"; \
    fi

COPY . .

RUN useradd --create-home appuser && mkdir -p /app/data /app/knowledge/uploads /models \
    && chown -R appuser /app /models
USER appuser

EXPOSE 8501 8000
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
