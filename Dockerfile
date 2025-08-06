FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    unzip \
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
RUN chmod +x /app/download_models.sh

# Expose Gradio port
EXPOSE 7860

# Run the model download script and start Gradio
CMD ["/bin/bash", "-c", "./download_models.sh && python app.py"]