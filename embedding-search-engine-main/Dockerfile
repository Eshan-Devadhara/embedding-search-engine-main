# ── Dockerfile ────────────────────────────────────────────────────────────────
# Lightweight production image for the Bedrock Embedding Search Engine.
#
# Authentication is provided at runtime by the ECS task IAM role via boto3's
# credential provider chain.  No AWS credentials are baked into this image.
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

# Non-root user for least-privilege execution
RUN addgroup --system app && adduser --system --ingroup app app

WORKDIR /app

# Install dependencies first (layer-caching friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app.py search.py bedrock.py subfeatures_dataset.csv ./

# Give the app user ownership
RUN chown -R app:app /app

USER app

EXPOSE 5000

# Gunicorn is intentionally NOT used here to keep the image minimal for the
# sandbox lab.  For production you would add gunicorn or uvicorn.
CMD ["python", "app.py"]
