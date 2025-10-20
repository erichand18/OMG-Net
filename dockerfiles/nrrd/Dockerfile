# syntax=docker/dockerfile:1.7

#########################
# 1. Base image with PyTorch + CUDA
#########################
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

#########################
# 2. System dependencies
#########################
RUN apt-get update && apt-get install -y \
    git wget curl build-essential \
    libssl-dev libffi-dev python3-dev \
    openslide-tools libopenslide0 libopenslide-dev \
    vim \
    && rm -rf /var/lib/apt/lists/*

#########################
# 3. Python dependencies
#########################
WORKDIR /app

RUN pip install --no-cache-dir -U pip wheel setuptools && \
    pip install --no-cache-dir \
        pandas==1.3.5 \
        numpy==1.21.6 \
        opencv-python-headless==4.5.5.64 \
        matplotlib==3.5.3 \
        tqdm==4.66.1 \
        torch==2.2.2 \
        torchvision==0.17.2 \
        pynrrd \
        openslide-python==1.1.2 \
        git+https://github.com/facebookresearch/segment-anything.git \
        scikit-learn==1.1.3 \
        fastai==1.0.61 \
        hydra-core==1.3.2 \
        omegaconf==2.2.3

#########################
# 4. Copy your code
#########################
COPY . /app

#########################
# 5. Environment variables
#########################
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/tmp/matplotlib

#########################
# 6. Default command
#########################
CMD ["/bin/bash"]
