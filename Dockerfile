FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies
RUN rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-dev python3.10-venv \
    ffmpeg git wget libgl1 libglib2.0-0 libsndfile1 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Wan2.2
RUN git clone https://github.com/Wan-Video/Wan2.2.git /opt/Wan2.2 \
    && cd /opt/Wan2.2 \
    && pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
       --index-url https://download.pytorch.org/whl/cu124 \
    && sed -i '/flash_attn/d' requirements.txt \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements_s2v.txt \
    && pip install --no-cache-dir flash_attn --no-build-isolation

# Model download (~50GB)
RUN pip install --no-cache-dir "huggingface_hub[cli]" \
    && huggingface-cli download Wan-AI/Wan2.2-S2V-14B \
       --local-dir /opt/models/Wan2.2-S2V-14B

# App dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY app/main.py .
COPY app/resources/ /app/resources/
COPY docker-entrypoint.sh .
RUN chmod +x /app/docker-entrypoint.sh

ENV WAN_PATH=/opt/Wan2.2
ENV CKPT_DIR=/opt/models/Wan2.2-S2V-14B

EXPOSE 11113

ENTRYPOINT ["sh", "/app/docker-entrypoint.sh"]
