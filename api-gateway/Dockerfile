# ---- Stage 1: builder (installe les dépendances) ----
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Stage 2: runtime (image finale, légère) ----
FROM python:3.11-slim
RUN useradd --create-home appuser
COPY --from=builder /install /usr/local
WORKDIR /srv
COPY main.py schemas.py ./
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]