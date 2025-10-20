# syntax=docker/dockerfile:1.7

###############################
# 1️⃣ Base image with CUDA runtime
###############################
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/conda/bin:$PATH \
    MAMBA_NO_BANNER=1 \
    PIP_NO_CACHE_DIR=1

###############################
# 2️⃣ System dependencies
###############################
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git build-essential \
    ninja-build pkg-config curl ca-certificates \
    python3 iputils-ping && \
    rm -rf /var/lib/apt/lists/*

###############################
# 3️⃣ Install Miniforge (Mamba-based)
###############################
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh && \
    conda clean -afy

###############################
# 4️⃣ Configure conda/mamba
###############################
RUN conda config --set always_yes true && \
    conda config --set channel_priority strict && \
    conda config --remove-key channels || true && \
    conda config --add channels conda-forge && \
    conda config --add channels defaults

###############################
# 5️⃣ Copy environment files separately for caching
###############################
WORKDIR /app
COPY environment.yaml /app/environment.yaml

###############################
# 6️⃣ Create base conda environment (conda-only deps)
###############################
# Split environment: keep heavy pip installs out of conda solver step
RUN --mount=type=cache,target=/opt/conda/pkgs \
    mamba env create -f environment.yaml --name mitotic --quiet && \
    conda clean -afy

###############################
# 7️⃣ Install pip packages separately (for caching + speed)
###############################
# Extract pip requirements into a temporary file
RUN mamba run -n mitotic python -m pip install --upgrade pip && \
    mamba run -n mitotic pip install \
        torch==2.4.0+cu121 torchvision==0.19.0+cu121 \
        --extra-index-url https://download.pytorch.org/whl/cu121 && \
    mamba run -n mitotic pip install \
        monai pynrrd scikit-learn scikit-image opencv-python pillow \
        git+https://github.com/facebookresearch/segment-anything.git \
        git+https://github.com/ChaoningZhang/MobileSAM.git \
        triton cupy-cuda12x cucim timm && \
    conda clean -afy

###############################
# 8️⃣ Copy rest of app
###############################
COPY . /app

###############################
# 9️⃣ Default command
###############################
SHELL ["/bin/bash", "-c"]
CMD ["bash"]