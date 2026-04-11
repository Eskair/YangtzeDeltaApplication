FROM python:3.11-slim

# System dependencies: tesseract (OCR), poppler (pdf2image), and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    poppler-utils \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create data directories so the app can write files
RUN mkdir -p src/data/proposals \
    src/data/prepared \
    src/data/extracted \
    src/data/refined_answers \
    src/data/expert_reports \
    src/data/reports \
    src/data/config/question_sets

# HF Spaces requires port 7860
EXPOSE 7860

CMD ["python", "-m", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "7860"]
