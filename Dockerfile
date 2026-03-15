FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-dev \
    ffmpeg git wget libgl1 libglib2.0-0 libsndfile1 g++ \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN git clone https://github.com/Wan-Video/Wan2.2.git /opt/Wan2.2 \
    && cd /opt/Wan2.2 \
    && pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
       --index-url https://download.pytorch.org/whl/cu124 \
    && sed -i '/flash_attn/d' requirements.txt \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements_s2v.txt \
    && pip install --no-cache-dir flash_attn --no-build-isolation

RUN pip install --no-cache-dir "huggingface_hub[cli]" \
    && huggingface-cli download Wan-AI/Wan2.2-S2V-14B \
       --local-dir /opt/models/Wan2.2-S2V-14B \
    && huggingface-cli download kresnik/wav2vec2-large-xlsr-korean \
       --local-dir /opt/models/wav2vec2-large-xlsr-korean

# 오디오 인코더를 한국어 Wav2Vec2로 교체
RUN sed -i "s|wav2vec2-large-xlsr-53-english|/opt/models/wav2vec2-large-xlsr-korean|g" \
    /opt/Wan2.2/wan/configs/wan_s2v_14B.py

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY app/main.py .
COPY app/resources/ /app/resources/
COPY docker-entrypoint.sh .
RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 11113

ENTRYPOINT ["sh", "/app/docker-entrypoint.sh"]
