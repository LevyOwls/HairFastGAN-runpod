FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git git-lfs build-essential cmake pkg-config \
    libx11-dev libatlas-base-dev libgtk-3-dev libboost-python-dev \
    ninja-build wget unzip libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install && \
    git clone https://github.com/AIRI-Institute/HairFastGAN.git

RUN cd HairFastGAN && \
    git clone https://huggingface.co/AIRI-Institute/HairFastGAN HairFastGAN_models && \
    cd HairFastGAN_models && git lfs pull && \
    cd .. && \
    mv HairFastGAN_models/pretrained_models pretrained_models && \
    mv HairFastGAN_models/input input && \
    rm -rf HairFastGAN_models

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch==1.13.1 torchvision==0.14.1 && \
    pip3 install --no-cache-dir -r HairFastGAN/requirements.txt && \
    pip3 install --no-cache-dir runpod

ENV PYTHONPATH=/workspace/HairFastGAN:$PYTHONPATH

WORKDIR /workspace/HairFastGAN

COPY rp_handler.py /workspace/HairFastGAN/

CMD ["python3", "rp_handler.py"]
