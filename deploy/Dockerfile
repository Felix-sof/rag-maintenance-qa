# The cloud image: the FastAPI service only, on PostgreSQL + pgvector.
#
#   docker build -f deploy/Dockerfile -t rag-api .
#   docker build -f deploy/Dockerfile --secret id=hf_token,env=HF_TOKEN -t rag-api .   # authenticated model download
#
# Three stages. "deps" installs the Python packages into a virtualenv; "models"
# downloads the embedding model and the reranker; the final stage copies only
# the virtualenv, the models and the code into a fresh slim image - no pip
# cache, no Streamlit, no ChromaDB (requirements-api.txt).
#
# The models are baked in (~1 GB) rather than downloaded at startup: a Cloud
# Run instance that scales from zero would otherwise fetch them from Hugging
# Face on every cold start - slower, and subject to its rate limits. At run
# time HF_HUB_OFFLINE=1 forbids any download, so a start never depends on it.

FROM python:3.11-slim AS deps
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /venv
ENV PATH=/venv/bin:$PATH
# CPU-only PyTorch first: otherwise sentence-transformers pulls the multi-GB CUDA build
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

FROM deps AS models
ARG EMBED_MODEL=intfloat/multilingual-e5-small
ARG RERANK_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
ENV HF_HOME=/models
# An optional Hugging Face token (a build secret, never stored in a layer) lifts
# the anonymous rate limit that can fail a build with HTTP 429.
RUN --mount=type=secret,id=hf_token \
    if [ -s /run/secrets/hf_token ]; then export HF_TOKEN="$(cat /run/secrets/hf_token)"; fi; \
    python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('${EMBED_MODEL}'); CrossEncoder('${RERANK_MODEL}')"

FROM python:3.11-slim
ARG EMBED_MODEL=intfloat/multilingual-e5-small
ARG RERANK_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
ENV PATH=/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    HF_HUB_OFFLINE=1 \
    EMBED_MODEL=${EMBED_MODEL} \
    RERANK_MODEL=${RERANK_MODEL} \
    VECTOR_BACKEND=pgvector \
    READ_ONLY=1 \
    WARM_UP=1
COPY --from=deps /venv /venv
COPY --from=models /models /models
WORKDIR /app
COPY *.py ./
COPY db/ db/
COPY analytics/ analytics/
COPY knowledge/ knowledge/
COPY ai4i/ ai4i/
RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8080
# Cloud Run sets PORT. --proxy-headers: the client address comes from the
# platform's X-Forwarded-For, for the per-client rate limit.
CMD ["sh", "-c", "exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers --forwarded-allow-ips='*'"]
