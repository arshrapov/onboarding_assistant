# ============================================================
# Multi-stage Dockerfile for Onboarding Assistant
# ============================================================
# Stage 1: Builder - Install dependencies
# Stage 2: Runtime - Lean production image
# ============================================================

# ============================================================
# Stage 1: Builder
# ============================================================
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data to avoid runtime permission issues
RUN python -c "import nltk; nltk.download('punkt', download_dir='/opt/venv/nltk_data'); nltk.download('punkt_tab', download_dir='/opt/venv/nltk_data'); nltk.download('stopwords', download_dir='/opt/venv/nltk_data')"

# ============================================================
# Stage 2: Runtime
# ============================================================
FROM python:3.11-slim

# Set metadata
LABEL maintainer="Onboarding Assistant"
LABEL description="AI-powered repository onboarding with RAG"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    NLTK_DATA=/opt/venv/nltk_data \
    # Default directories (can be overridden)
    DATA_DIR=/app/data \
    REPOS_DIR=/app/data/repos \
    INDEXES_DIR=/app/data/indexes \
    CACHE_DIR=/app/data/cache

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY app/ /app/app/

# Create data directories with proper permissions
RUN mkdir -p /app/data/repos /app/data/indexes /app/data/cache /app/data/jobs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port (default 7860, can be overridden)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7860/api/health || exit 1

# Default command: run FastAPI + Gradio app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
