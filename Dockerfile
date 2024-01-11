FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies including MPI, OpenGL libraries for OpenCV, and GLib for libgthread
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    libopenmpi-dev \
    openmpi-bin \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project into the container
COPY . .
