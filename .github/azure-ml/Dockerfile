# L3 VFA/PMA Pipeline - Reproducible Docker Environment
# Based on NVIDIA CUDA for GPU support

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

LABEL maintainer="L3 VFA/PMA Team"
LABEL description="Azure ML L3 VFA/PMA Processing Environment"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install medical imaging libraries
RUN pip install \
    monai==1.2.0 \
    SimpleITK==2.3.0 \
    pydicom==2.4.0 \
    scikit-image==0.21.0

# Install Azure ML SDK
RUN pip install \
    azureml-sdk==1.56.0 \
    azureml-core==1.56.0

# Install scientific computing
RUN pip install \
    numpy==1.23.5 \
    scipy==1.10.1 \
    pandas==2.0.3 \
    matplotlib==3.7.2

# Install computer vision
RUN pip install opencv-python-headless==4.8.0.76

# Install utilities
RUN pip install \
    tqdm==4.66.1 \
    Pillow==10.0.0 \
    pyyaml==6.0.1

# Create workspace directory
WORKDIR /workspace

# Copy project files (will be mounted at runtime)
COPY . /workspace/

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV OMP_NUM_THREADS=4

# Expose ports (if needed for Jupyter or TensorBoard)
EXPOSE 8888 6006

# Default command
CMD ["/bin/bash"]
