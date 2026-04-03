FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/

RUN mkdir -p data logs

# Run as a non-root user for improved container security.
RUN adduser --system --no-create-home --group appuser \
    && chown -R appuser:appuser /app
USER appuser

CMD ["python", "src/main.py"]