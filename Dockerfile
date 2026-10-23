# RAGLab service — standalone container image.
# Build from the REPO ROOT (so docs/ is in the context):
#   docker build -t raglab-service .
# Run (keys via env; the image never bakes them):
#   docker run --rm -p 8000:8000 --env-file raglab/.env raglab-service
# Interactive API docs: http://localhost:8000/docs
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app/raglab

# Install first (cached layer), then copy the code.
COPY raglab/requirements.txt raglab/requirements-service.txt ./
RUN pip install -r requirements-service.txt

# The whole lab folder (service.py, profiles.py, the core modules) + the
# evaluation corpus as the default RAGLAB_DATA_DIRS target.
COPY raglab/ /app/raglab/
COPY docs/ /app/docs/

ENV RAGLAB_DATA_DIRS=/app/docs

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
