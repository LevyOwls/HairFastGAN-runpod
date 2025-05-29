# Usa la versión "devel" para tener nvcc incluido
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Evitar interacciones durante la instalación
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
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

# Crear directorio de trabajo
WORKDIR /workspace

# Clonar el repositorio
RUN git clone https://github.com/AIRI-Institute/HairFastGAN.git && \
    cd HairFastGAN && git lfs pull

# Descargar modelos preentrenados
RUN cd HairFastGAN && \
    git clone https://huggingface.co/AIRI-Institute/HairFastGAN && \
    cd HairFastGAN && git lfs pull && \
    cd .. && \
    mv HairFastGAN/pretrained_models pretrained_models && \
    mv HairFastGAN/input input && \
    rm -rf HairFastGAN

# Instalar dependencias de Python
# Alineamos la versión de Torch a CUDA 11.8
RUN pip3 install --no-cache-dir \
    torch==1.13.1+cu118 \
    torchvision==0.14.1+cu118 \
    -f https://download.pytorch.org/whl/torch_stable.html && \
    pip3 install --no-cache-dir -r HairFastGAN/requirements.txt && \
    pip3 install --no-cache-dir runpod

# Añadir el PATH a CUDA
ENV PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Configurar PYTHONPATH
ENV PYTHONPATH=/workspace/HairFastGAN:$PYTHONPATH

# Directorio de trabajo final
WORKDIR /workspace/HairFastGAN

# Copiar el handler
COPY rp_handler.py /workspace/HairFastGAN/

# Comando por defecto para RunPod
CMD ["python3", "rp_handler.py"]
