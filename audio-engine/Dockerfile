# ---- Stage 1: builder ----
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Stage 2: runtime ----
FROM python:3.11-slim
# libsndfile1: lecture des .wav par soundfile/librosa
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home appuser
# numba (utilisé par librosa) a besoin d'un dossier de cache accessible en écriture
ENV NUMBA_CACHE_DIR=/tmp/numba_cache PYTHONUNBUFFERED=1
COPY --from=builder /install /usr/local
WORKDIR /srv
COPY app ./app
COPY models ./models
USER appuser
EXPOSE 8001
HEALTHCHECK --interval=10s --timeout=3s --start-period=40s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]