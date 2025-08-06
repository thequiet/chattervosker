#!/bin/bash
# Download VOSK model with improved error handling and resume capability

# Configuration
MODEL_NAME="vosk-model-en-us-0.22"
MODEL_DIR="/app/models"
MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"
ZIP_FILE="vosk-model.zip"
MODEL_URL="https://alphacephei.com/vosk/models/${MODEL_NAME}.zip"
MIN_FILE_SIZE=1000000000  # 1GB minimum
MAX_RETRIES=5
RETRY_DELAY=10

echo "Starting VOSK model download process..."
echo "Model: ${MODEL_NAME}"
echo "Target directory: ${MODEL_PATH}"

if [ ! -d "${MODEL_PATH}" ]; then
    echo "Downloading VOSK model..."
    
    # Create models directory
    mkdir -p "${MODEL_DIR}"
    cd "${MODEL_DIR}"
    
    # Download with multiple fallback attempts
    DOWNLOAD_SUCCESS=false
    for attempt in $(seq 1 ${MAX_RETRIES}); do
        echo "Download attempt ${attempt}/${MAX_RETRIES}..."
        
        # Use curl with resume capability and better settings
        curl -L -C - --max-time 3600 --retry 3 --retry-delay 5 \
             --connect-timeout 30 --speed-time 60 --speed-limit 50000 \
             -o "${ZIP_FILE}" \
             "${MODEL_URL}"
        
        # Check if download was successful
        if [ $? -eq 0 ] && [ -f "${ZIP_FILE}" ]; then
            # Verify file size
            FILE_SIZE=$(stat -c%s "${ZIP_FILE}" 2>/dev/null || stat -f%z "${ZIP_FILE}" 2>/dev/null)
            if [ "$FILE_SIZE" -ge "${MIN_FILE_SIZE}" ]; then
                echo "Download successful! File size: ${FILE_SIZE} bytes"
                DOWNLOAD_SUCCESS=true
                break
            else
                echo "Downloaded file too small (${FILE_SIZE} bytes), retrying..."
                rm -f "${ZIP_FILE}"
            fi
        else
            echo "Download failed, retrying in ${RETRY_DELAY} seconds..."
            rm -f "${ZIP_FILE}"
            sleep ${RETRY_DELAY}
        fi
    done
    
    # Check if download ultimately failed
    if [ "$DOWNLOAD_SUCCESS" = false ]; then
        echo "Error: Failed to download VOSK model after ${MAX_RETRIES} attempts"
        rm -f "${ZIP_FILE}"
        exit 1
    fi
    
    # Verify zip integrity
    echo "Verifying download integrity..."
    unzip -t vosk-model.zip > /dev/null
    if [ $? -ne 0 ]; then
        echo "Error: Downloaded file is not a valid zip archive"
        rm -f vosk-model.zip
        exit 1
    fi
    
    echo "Extracting VOSK model..."
    unzip vosk-model.zip -d /app/models
    if [ $? -ne 0 ]; then
        echo "Error: Failed to extract VOSK model"
        rm -f vosk-model.zip
        exit 1
    fi
    
    rm -f vosk-model.zip
    echo "VOSK model downloaded and extracted successfully"
else
    echo "VOSK model already exists at /app/models/vosk-model-en-us-0.22"
fi