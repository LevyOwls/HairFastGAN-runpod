# =========================================
# Dockerfile para HairFastGAN en RunPod
# =========================================

# 1) Imagen devel para incluir nvcc
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 2) No prompts en apt
ENV DEBIAN_FRONTEND=noninteractive

# 3) Enable repos y apuntar a mirror US
RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common \
      apt-transport-https \
      ca-certificates \
    && add-apt-repository main \
    && add-apt-repository universe \
    && add-apt-repository restricted \
    && add-apt-repository multiverse \
    && sed -i 's|https://archive.ubuntu.com/ubuntu|https://us.archive.ubuntu.com/ubuntu|g; s|http://archive.ubuntu.com/ubuntu|http://us.archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        git \
        git-lfs \
        build-essential \
        cmake \
        pkg-config \
        libx11-dev \
        libatlas-base-dev \
        libgtk-3-dev \
        libboost-python-dev \
        ninja-build \
        wget \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# 4) workspace
WORKDIR /workspace

# 5) Clonar código y pesos de HuggingFace
RUN git clone https://github.com/AIRI-Institute/HairFastGAN.git && \
    cd HairFastGAN && git lfs pull && \
    cd .. && \
    git clone https://huggingface.co/AIRI-Institute/HairFastGAN && \
    cd HairFastGAN && git lfs pull && \
    cd .. && \
    mv HairFastGAN/pretrained_models pretrained_models && \
    mv HairFastGAN/input        input && \
    rm -rf HairFastGAN

# 6) Pre-descarga de un checkpoint de ejemplo (resnet18)
RUN python3 - << 'EOF'
import torch
torch.hub.load_state_dict_from_url(
    'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    model_dir='/root/.cache/torch/hub/checkpoints',
    progress=True
)
EOF

# 7) Instalar Python deps (+cu117)
RUN pip3 install --no-cache-dir \
      torch==1.13.1+cu117 \
      torchvision==0.14.1+cu117 \
      -f https://download.pytorch.org/whl/torch_stable.html \
    && pip3 install --no-cache-dir -r pretrained_models/requirements.txt \
    && pip3 install --no-cache-dir runpod

# 8) Paths de CUDA y PYTHONPATH
ENV PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    PYTHONPATH=/workspace/pretrained_models:${PYTHONPATH}

# 9) Directorio final
WORKDIR /workspace/pretrained_models

# 10) Copiar scripts
COPY rp_handler.py    .
COPY run_hairfast.py .

# 11) default runpod
# Para RunPod serverless:
CMD ["python3", "rp_handler.py"]
