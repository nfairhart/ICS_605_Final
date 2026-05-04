FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for layer caching
COPY requirements.txt requirements-webapp.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-webapp.txt

# Copy application source
COPY app/ ./app/
COPY matching.py ./

# Bundle the pre-built ChromaDB vector store (read-only at runtime)
COPY chroma_db/ ./chroma_db/

# Streamlit server config for Cloud Run
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8080

CMD ["streamlit", "run", "app/app.py", "--server.port=8080", "--server.address=0.0.0.0"]
