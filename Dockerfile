# Torch-free RAG API image (deps slimmed in chore/slim-deps → small, fast cold starts).
FROM python:3.10-slim

WORKDIR /app

# System certs for outbound HTTPS (OpenAI / Cohere / Pinecone / Supabase).
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render/Cloud Run inject $PORT.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
