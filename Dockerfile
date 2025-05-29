FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Evitar interacciones durante la instalación
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3.10 \
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
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /workspace

# Clonar el repositorio
RUN git clone https://github.com/AIRI-Institute/HairFastGAN.git && \
    cd HairFastGAN && \
    git lfs pull

# Descargar modelos preentrenados
RUN cd HairFastGAN && \
    git clone https://huggingface.co/AIRI-Institute/HairFastGAN && \
    cd HairFastGAN && \
    git lfs pull && \
    cd .. && \
    mv HairFastGAN/pretrained_models pretrained_models && \
    mv HairFastGAN/input input && \
    rm -rf HairFastGAN

# Instalar dependencias de Python
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    torch==1.13.1+cu116 \
    torchvision==0.14.1+cu116 \
    -f https://download.pytorch.org/whl/torch_stable.html \
    && pip3 install --no-cache-dir -r HairFastGAN/requirements.txt \
    && pip3 install --no-cache-dir runpod

# Configurar variables de entorno
ENV PYTHONPATH=/workspace/HairFastGAN:$PYTHONPATH

# Directorio de trabajo final
WORKDIR /workspace/HairFastGAN

# Copiar el handler
COPY rp_handler.py /workspace/HairFastGAN/

# Comando por defecto para RunPod
CMD ["python3", "rp_handler.py"] 
