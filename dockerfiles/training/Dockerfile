# syntax=docker/dockerfile:1.7

###########################
# 1️⃣ Base image with CUDA runtime + PyTorch
###########################
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Optional: if CPU-only build needed, switch to:
# FROM pytorch/pytorch:2.4.0-cpu

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV MAMBA_NO_BANNER=1
WORKDIR /app

###########################
# 2️⃣ System dependencies
###########################
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates build-essential vim \
    && rm -rf /var/lib/apt/lists/*

###########################
# 3️⃣ Copy project files
###########################
COPY . /app

###########################
# 4️⃣ Install conda-free dependencies
###########################
# Everything needed only for training; no SAM, OpenSlide, etc.
RUN pip install --no-cache-dir \
    lightning>=2.2 \
    albumentations \
    opencv-python-headless \
    scikit-learn \
    pandas \
    numpy \
    matplotlib \
    toml \
    tensorboard \
    tqdm
RUN pip install --no-cache-dir \
    torchmetrics \
    colorama \
    scikit-image
RUN pip install --no-cache-dir \
    pynrrd

###########################
# 5️⃣ Set default runtime options
###########################
# Use mixed precision for faster matmul ops
ENV TORCH_CUDNN_V8_API_DISABLED=1
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV PYTHONPATH=/app

###########################
# 6️⃣ Entrypoint
###########################
# Example: run training with your config
# docker run --gpus all -v /path/to/data:/data omgnet-train \
#     python Training/Classifier.py Configs/Classifier_config.ini
CMD [ "/bin/bash" ]