# =========================================
# Dockerfile para HairFastGAN en RunPod
# =========================================

# 1) Imagen devel para incluir nvcc
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 2) Evitar prompts durante apt
ENV DEBIAN_FRONTEND=noninteractive

# 3) Habilitar repos y cambiar mirror, luego instalar deps del sistema
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

# 4) Directorio de trabajo inicial
WORKDIR /workspace

# 5) Clonar HairFastGAN y descargar con Git LFS
RUN git clone https://github.com/AIRI-Institute/HairFastGAN.git \
    && cd HairFastGAN \
    && git lfs pull

# 6) Bajar pesos preentrenados de HuggingFace y reordenar carpetas
RUN cd HairFastGAN \
    && git clone https://huggingface.co/AIRI-Institute/HairFastGAN \
    && cd HairFastGAN && git lfs pull \
    && cd .. \
    && mv HairFastGAN/pretrained_models pretrained_models \
    && mv HairFastGAN/input        input \
    && rm -rf HairFastGAN

# 7) Instalar dependencias Python con +cu117
RUN pip3 install --no-cache-dir \
      torch==1.13.1+cu117 \
      torchvision==0.14.1+cu117 \
      -f https://download.pytorch.org/whl/torch_stable.html \
    && pip3 install --no-cache-dir -r HairFastGAN/requirements.txt \
    && pip3 install --no-cache-dir runpod

# 8) Asegurar nvcc y librerías CUDA en PATH
ENV PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    PYTHONPATH=/workspace/HairFastGAN:${PYTHONPATH}

# 9) Directorio final de trabajo
WORKDIR /workspace/HairFastG


# 10) Copiar tu handler al contenedor
COPY rp_handler.py /workspace/HairFastGAN/

# 11) Arrancar el servidor de RunPod
CMD ["python3", "rp_handler.py"]
