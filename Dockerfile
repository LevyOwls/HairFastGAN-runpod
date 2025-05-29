# ======= Dockerfile para HairFastGAN en RunPod (Ubuntu 22.04 + CUDA 11.8) =======
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Evitar interacciones en apt
ENV DEBIAN_FRONTEND=noninteractive

# (Opcional) Cambiar mirror si da 403
# RUN sed -i 's|http://archive.ubuntu.com/ubuntu|http://us.archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list

# 1) Dependencias base (sin git)
RUN apt-get update \
    && apt-get install -y --no-install-recommends --fix-missing \
       python3.10 \
       python3-pip \
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
       libgl1-mesa-glx \
       libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2) Instalar git solo si no existe
RUN if ! command -v git >/dev/null 2>&1; then \
      apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*; \
    fi \
    && git --version

# 3) Crear workspace
WORKDIR /workspace

# 4) Clonar repo y LFS
RUN git clone https://github.com/AIRI-Institute/HairFastGAN.git \
    && cd HairFastGAN \
    && git lfs pull

# 5) Modelos preentrenados de HF
RUN cd HairFastGAN \
    && git clone https://huggingface.co/AIRI-Institute/HairFastGAN pretrained_hf \
    && cd pretrained_hf \
    && git lfs pull \
    && cd .. \
    && mv pretrained_hf/pretrained_models pretrained_models \
    && mv pretrained_hf/input input \
    && rm -rf pretrained_hf

# 6) Dependencias Python
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir \
         torch==1.13.1+cu116 torchvision==0.14.1+cu116 \
         --extra-index-url https://download.pytorch.org/whl/cu116 \
    && python3 -m pip install --no-cache-dir -r HairFastGAN/requirements.txt \
    && python3 -m pip install --no-cache-dir runpod

# 7) PYTHONPATH
ENV PYTHONPATH=/workspace/HairFastGAN

# 8) Directorio final
WORKDIR /workspace/HairFastGAN

# 9) Copiar handler
COPY rp_handler.py /workspace/HairFastGAN/

# 10) CMD por defecto
CMD ["python3", "rp_handler.py"]
