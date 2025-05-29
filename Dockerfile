FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 0) Cambiar mirror a Chile y desbloquear universe
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|http://cl.archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list \
 && sed -i 's|^# deb .* universe|deb http://cl.archive.ubuntu.com/ubuntu universe|g' /etc/apt/sources.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends software-properties-common \
 && add-apt-repository universe

# 1) Instalar dependencias del sistema (incluye git-lfs, build-essential, atlas, boost…)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
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

# 2) Instalar git si por algún motivo no está (y chequear versión)
RUN if ! command -v git >/dev/null; then \
      apt-get update \
   && apt-get install -y --no-install-recommends git \
   && rm -rf /var/lib/apt/lists/*; \
    fi \
 && git --version

WORKDIR /workspace

# 3) Clonar repo + LFS
RUN git clone https://github.com/AIRI-Institute/HairFastGAN.git \
 && cd HairFastGAN \
 && git lfs pull

# 4) Modelos preentrenados de HF
RUN cd HairFastGAN \
 && git clone https://huggingface.co/AIRI-Institute/HairFastGAN pretrained_hf \
 && cd pretrained_hf \
 && git lfs pull \
 && cd .. \
 && mv pretrained_hf/pretrained_models pretrained_models \
 && mv pretrained_hf/input input \
 && rm -rf pretrained_hf

# 5) Dependencias Python
RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir \
      torch==1.13.1+cu116 torchvision==0.14.1+cu116 \
      --extra-index-url https://download.pytorch.org/whl/cu116 \
 && python3 -m pip install --no-cache-dir -r HairFastGAN/requirements.txt \
 && python3 -m pip install --no-cache-dir runpod

ENV PYTHONPATH=/workspace/HairFastGAN
WORKDIR /workspace/HairFastGAN
COPY rp_handler.py /workspace/HairFastGAN/

CMD ["python3", "rp_handler.py"]
