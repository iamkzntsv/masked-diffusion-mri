FROM continuumio/miniconda3

WORKDIR /app

# Install system-level dependencies
USER root
RUN apt-get update && \
    apt-get install -y gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Create Conda environment
COPY environment.yml .
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "masked-diffusion-mri", "/bin/bash", "-c"]

COPY . .
