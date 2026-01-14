FROM python:3.11-slim

# Set environment variables to optimize Python execution in Docker
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_TECH_HOME=/app

# Set the working directory
WORKDIR /app

# Install system dependencies required for building Python packages and curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only the files needed for installation first to leverage caching
COPY pyproject.toml README.md ./

# Copy the source code
COPY src/ ./src/

# Install the package and its dependencies
RUN pip install --upgrade pip && \
    pip install .

# Create directories for data and artifacts
RUN mkdir -p data artifacts

# Create a non-root user for security
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Expose the port the application runs on
EXPOSE 8000

# Define the command to run the application
CMD ["model-tech", "serve", "--host", "0.0.0.0", "--port", "8000"]

