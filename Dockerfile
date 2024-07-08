# Use a Python base image
FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy the Medusa repository
COPY . .

RUN mkdir /work
RUN cd /work && git clone https://github.com/lm-sys/FastChat.git
RUN cd /work/FastChat && pip3 install -e ".[model_worker,webui]"

# Install Medusa in editable mode
RUN pip install -e '.[train]'
RUN pip install deepspeed fschat accelerate sentencepiece numpy==1.26.4 matplotlib pygraphviz tensorboardX
#RUN fschat@git+https://github.com/lm-sys/FastChat.git

# Set the environment variables
ENV PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/app/cache

# Expose the port (if needed)
EXPOSE 2222 6006 18000-18100
