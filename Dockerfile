FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/

# Train on synthetic data at build time so the image is runnable out of the
# box. Swap this for a COPY of a real trained artifact in production.
RUN python scripts/train.py --synthetic

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
