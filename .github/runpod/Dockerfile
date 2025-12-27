# RunPod GPU Container for L3 VFA/PMA Teacher Generation
# CUDA-enabled PyTorch with TotalSegmentator

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir \
    totalsegmentator>=2.0.0 \
    nibabel>=5.0.0 \
    SimpleITK>=2.2.0 \
    pydicom>=2.3.0 \
    opencv-python-headless>=4.7.0 \
    scikit-image>=0.19.0 \
    scipy>=1.9.0 \
    numpy>=1.21.0 \
    tqdm>=4.65.0 \
    matplotlib>=3.7.0 \
    monai>=1.3.0

# Set working directory
WORKDIR /workspace

# Copy project files
COPY .github/azure-ml/step1_real_teachers_all.py /workspace/
COPY .github/runpod/run_teacher_generation.sh /workspace/
RUN chmod +x /workspace/run_teacher_generation.sh

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV TOTALSEG_SCRATCH=/tmp
ENV SCRATCH=/tmp
ENV OMP_NUM_THREADS=8

# RunPod expects data at /runpod-volume
# Input: /runpod-volume/amos22/
# Output: /runpod-volume/teacher_labels/

# Default command
CMD ["/workspace/run_teacher_generation.sh"]
