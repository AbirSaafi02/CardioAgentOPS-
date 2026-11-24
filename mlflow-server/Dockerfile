FROM python:3.11-slim
RUN pip install --no-cache-dir mlflow==2.17.2 SQLAlchemy==2.0.36

# Dossier de données (monté en volume par docker-compose)
RUN mkdir -p /mlflow

EXPOSE 5000

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", \
     "--serve-artifacts"]
