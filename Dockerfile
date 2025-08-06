FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    unzip \
    coreutils \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir \
    git+https://github.com/openai/whisper.git \
    vosk \
    gradio \
    numpy==1.26.4 \
    librosa \
    chatterbox-tts \
    peft

# Install PyTorch with CUDA support and Triton
RUN pip install --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    triton>=2.0.0

# Set working directory
WORKDIR /app

# Copy application code and scripts
COPY app.py /app/app.py
COPY download_models.sh /app/download_models.sh
COPY stop_inactive_pod.py /app/stop_inactive_pod.py
RUN chmod +x /app/download_models.sh

# Expose Gradio port
EXPOSE 7860

# Run model download, start Gradio with logging, and run inactivity monitor
CMD ["/bin/bash", "-c", "./download_models.sh && nohup python stop_inactive_pod.py & python app.py > app.log 2>&1"]