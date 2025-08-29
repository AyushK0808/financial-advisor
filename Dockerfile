FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc g++ curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --no-deps -r requirements.txt --no-build-isolation --no-compile --upgrade --disable-pip-version-check

# Download NLTK data
RUN python -c "import nltk; nltk.download('vader_lexicon')"

# Copy application
COPY . .

# Create directories
RUN mkdir -p models/financial_model data/user_data vector_store/chromadb

EXPOSE 8501

CMD ["streamlit", "run", "src/streamlit_app.py", "--server.address", "0.0.0.0"]