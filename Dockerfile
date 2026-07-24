FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LIGWEB_HOST=0.0.0.0 \
    LIGWEB_PORT=8000 \
    LIGWEB_TRAIN_DATA_DIR=/data/train \
    LIGWEB_CORRECTION_DATA_DIR=/data/correction \
    LIGWEB_FEEDBACK_DIR=/data/correction/.ligedit \
    LIGWEB_EXPORT_DIR=/data/correction/exports \
    LIGWEB_BASE_MODEL_PATH=/data/correction/.ligedit/main_model/current.onnx \
    LIGWEB_BASE_MODEL_METADATA_PATH=/data/correction/.ligedit/main_model/current.json \
    LIGWEB_AUTO_CORRECTION_TRAINING=1 \
    LIGWEB_AUTO_IC_SYNC=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)" || exit 1

CMD ["python", "-m", "uvicorn", "ligweb.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
