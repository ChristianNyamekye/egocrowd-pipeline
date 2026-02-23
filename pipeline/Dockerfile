FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    ffmpeg curl unzip git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY egodex_to_lerobot.py .
COPY run.sh .
RUN chmod +x run.sh

CMD ["bash", "run.sh"]
