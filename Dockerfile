# Multi-stage Dockerfile for Housing Price Prediction API
# Stage 1: Model Training
FROM python:3.9-slim as training-stage

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all necessary files for training
COPY application/ ./application/


# Create models directory and run training with validation
RUN mkdir -p models && \
    python application/train.py && \
    ls -la models/ && \
    test -f models/housing_model.keras && \
    test -f models/scaler.pkl && \
    test -f models/metadata.json && \
    echo "✅ All model artifacts created successfully"

# Stage 2: API Runtime  
FROM python:3.9-slim as runtime-stage

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy trained models from training stage with verification
COPY --from=training-stage /app/models ./models
RUN ls -la models/ && \
    test -f models/housing_model.keras && \
    test -f models/scaler.pkl && \
    test -f models/metadata.json && \
    echo "✅ Model artifacts successfully copied"

# Copy API application
COPY application/api.py .

# Set environment variables
ENV PORT=8080
ENV MODEL_PATH=models/housing_model.keras
ENV SCALER_PATH=models/scaler.pkl
ENV METADATA_PATH=models/metadata.json

# Expose the port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the API
CMD ["python", "api.py"]