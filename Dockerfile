# syntax=docker/dockerfile:1
#
# Multi-stage build. The model weights (~200 MB for the embedder + reranker)
# are baked into the image at build time rather than downloaded on first
# request, so a cold container serves its first query immediately instead of
# stalling for a Hugging Face download.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*


# --- dependencies ----------------------------------------------------------
FROM base AS deps

COPY pyproject.toml README.md ./
RUN mkdir -p src api \
    && touch src/__init__.py api/__init__.py \
    && pip install --upgrade pip \
    # CPU-only torch keeps the image ~2 GB smaller than the CUDA default.
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -e .


# --- model cache -----------------------------------------------------------
FROM deps AS models

ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('${EMBEDDING_MODEL}'); \
CrossEncoder('${RERANKER_MODEL}')"


# --- runtime ---------------------------------------------------------------
FROM models AS runtime

COPY src ./src
COPY api ./api
COPY frontend ./frontend
COPY .streamlit ./.streamlit

RUN mkdir -p data/raw data/processed data/indexes data/metadata data/evaluation

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app /opt/hf
USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
