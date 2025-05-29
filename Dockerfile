# =========================================
# Dockerfile para HairFastGAN en RunPod
# =========================================

# 1) Usamos la imagen devel para tener nvcc disponible
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 2) Evitar prompts durante la instalación
ENV DEBIAN_FRONTEND=noninteractive

# 3) Habilitar universe y instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common \
    && add-apt-repository universe \
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

# 4) Crear y posicionar el workspace
WORKDIR /workspace

# 5) Clonar repositorio y bajar modelos con Git LFS
RUN git clone https://github.com/AIRI-Institute/HairFastGAN.git \
    && cd HairFastGAN \
    && git lfs pull

# 6) Descargar los pesos preentrenados desde HuggingFace
RUN cd HairFastGAN \
    && git clone https://huggingface.co/AIRI-Institute/HairFastGAN \
    && cd HairFastGAN \
    && git lfs pull \
    && cd .. \
    && mv HairFastGAN/pretrained_models pretrained_models \
    && mv HairFastGAN/input input \
    && rm -rf HairFastGAN

# 7) Instalar dependencias de Python
#    - Torch y torchvision empatan con CUDA 11.8 (+cu118)
#    - El resto desde requirements.txt
RUN pip3 install --no-cache-dir \
      torch==1.13.1+cu118 \
      torchvision==0.14.1+cu118 \
      -f https://download.pytorch.org/whl/torch_stable.html \
    && pip3 install --no-cache-dir -r HairFastGAN/requirements.txt \
    && pip3 install --no-cache-dir runpod

# 8) Asegurar accesibilidad a nvcc y librerías CUDA
ENV PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# 9) Añadir HairFastGAN al PYTHONPATH
ENV PYTHONPATH=/workspace/HairFastGAN:$PYTHONPATH

# 10) Directorio final de trabajo
WORKDIR /workspace/HairFastGAN

# 11) Copiar tu handler de RunPod
COPY rp_handler.py /workspace/HairFastGAN/

# 12) Comando por defecto al iniciar el contenedor
CMD ["python3", "rp_handler.py"]
