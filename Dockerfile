FROM python:3.11-slim

# long timeout and retries: the downloads are large and a slow connection must not fail the build
ENV PYTHONUNBUFFERED=1 PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=10 HF_HOME=/models/hf
WORKDIR /app

# Set to true to find DOCX page numbers inside the container (adds about 500 MB)
ARG INSTALL_LIBREOFFICE=false
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && if [ "$INSTALL_LIBREOFFICE" = "true" ]; then apt-get install -y --no-install-recommends libreoffice-writer; fi \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch keeps the image much smaller than the default CUDA build.
# It is its own layer, before requirements.txt, so changing requirements.txt never downloads it again.
# The cache mounts keep downloaded packages between builds (BuildKit, the default in Docker Desktop).
RUN --mount=type=cache,target=/root/.cache/pip pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

COPY app ./app
COPY data ./data

EXPOSE 8000
# The first start downloads the embedding model, so allow a long start period
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
  CMD curl -fs http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
