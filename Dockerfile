# Use the kundajelab/chrombpnet Docker image as base.
FROM kundajelab/chrombpnet:latest

# Install system dependencies for venv (if not already included in the base image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /scratch/venv

RUN pip install fsspec
RUN pip install gcsfs
# Add GCP dependencies.
RUN pip install --upgrade protobuf==3.20.0

COPY . /scratch/variant-scorer
RUN pip install -e ./variant-scorer

# Make the script executable.
# RUN chmod +x /scratch/variant-scorer/src/variant_scoring.per_chrom.py

