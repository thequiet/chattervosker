import gradio as gr
import whisper
from vosk import Model, KaldiRecognizer
import json
import wave
import numpy as np
import os
import torch
import torchaudio
import logging
import traceback
import sys
import subprocess
import tempfile
import signal
from datetime import datetime
from chatterbox.tts import ChatterboxTTS
import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError, ClientError
from pod_shutdown import perform_pod_shutdown

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set up specific loggers for different components
gradio_logger = logging.getLogger('gradio')
gradio_logger.setLevel(logging.DEBUG)

# Custom exception handler
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

logger.info("="*50)
logger.info("Starting ChatterVosker Application")
logger.info(f"Python version: {sys.version}")
logger.info(f"Start time: {datetime.now()}")
logger.info("="*50)

# Load models
logger.info("Initializing models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Optimize Triton for your GPU
if torch.cuda.is_available():
    logger.info("Setting up CUDA optimizations...")
    os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
    os.environ["TRITON_PRINT_AUTOTUNING"] = "0"
    logger.info(f"GPU Memory before loading: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB total")

# Check Triton availability
try:
    import triton
    logger.info(f"✓ Triton is available: {triton.__version__}")
    triton_available = True
except ImportError as e:
    logger.warning(f"✗ Triton is not installed: {e}")
    triton_available = False

logger.info(f"✓ PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"✓ CUDA device: {torch.cuda.get_device_name()}")
    logger.info(f"✓ CUDA version: {torch.version.cuda}")

# Load Whisper model with error handling
try:
    logger.info("Loading Whisper model...")
    whisper_model = whisper.load_model("turbo", device=device)
    logger.info("✓ Whisper model loaded successfully")
except Exception as e:
    logger.error(f"✗ Failed to load Whisper model: {e}")
    logger.error(traceback.format_exc())
    raise

# Load VOSK model with error handling
vosk_model = None
vosk_model_path = "/app/models/vosk-model-en-us-0.22"
logger.info(f"Checking VOSK model at: {vosk_model_path}")

try:
    if os.path.exists(vosk_model_path) and os.path.exists(f"{vosk_model_path}/am/final.mdl"):
        logger.info("Loading VOSK model...")
        vosk_model = Model(vosk_model_path)
        logger.info("✓ VOSK model loaded successfully")
    else:
        logger.warning(f"VOSK model not found at {vosk_model_path}. Attempting to download...")
        result = subprocess.run(["/app/download_models.sh"], capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(f"{vosk_model_path}/am/final.mdl"):
            logger.info("VOSK model downloaded successfully. Loading...")
            vosk_model = Model(vosk_model_path)
            logger.info("✓ VOSK model loaded successfully")
        else:
            logger.error(f"Failed to download VOSK model: {result.stderr}")
            logger.warning("VOSK transcription will be disabled.")
except Exception as e:
    logger.error(f"✗ Failed to load VOSK model: {e}")
    logger.error(traceback.format_exc())
    logger.warning("VOSK transcription will be disabled.")

# Load Chatterbox model with error handling
try:
    logger.info("Loading Chatterbox TTS model...")
    chatterbox_model = ChatterboxTTS.from_pretrained(device=device)
    logger.info("✓ Chatterbox model loaded successfully")
except Exception as e:
    logger.error(f"✗ Failed to load Chatterbox model: {e}")
    logger.error(traceback.format_exc())
    raise

if torch.cuda.is_available():
    logger.info(f"GPU Memory after loading models: {torch.cuda.memory_allocated() / 1024**3:.1f}GB allocated")

logger.info("All available models loaded successfully!")

# S3 configuration and helpers
S3_BUCKET = os.environ.get("S3_BUCKET") or os.environ.get("AWS_S3_BUCKET")
S3_PREFIX = os.environ.get("S3_PREFIX") or os.environ.get("S3_DIR") or "uploads"
S3_REGION = os.environ.get("S3_REGION") or os.environ.get("AWS_DEFAULT_REGION")
AWS_ACCESS_KEY_ID = os.environ.get("S3_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("S3_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")

_s3_client = None

def get_s3_client():
    global _s3_client
    if _s3_client is not None:
        return _s3_client
    try:
        session_kwargs = {}
        client_kwargs = {}
        if S3_REGION:
            session_kwargs["region_name"] = S3_REGION
        if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
            client_kwargs.update({
                "aws_access_key_id": AWS_ACCESS_KEY_ID,
                "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
            })
        session = boto3.session.Session(**session_kwargs) if session_kwargs else boto3
        _s3_client = session.client("s3", **client_kwargs)
        logger.info("✓ Initialized S3 client")
        return _s3_client
    except Exception as e:
        logger.error(f"Failed to initialize S3 client: {e}")
        return None

def s3_upload_file(local_path, key_prefix=S3_PREFIX):
    if not S3_BUCKET:
        logger.info("S3_BUCKET not set; skipping S3 upload")
        return None
    client = get_s3_client()
    if client is None:
        return None
    try:
        basename = os.path.basename(local_path)
        date_prefix = datetime.utcnow().strftime("%Y/%m/%d")
        key_parts = [p for p in [key_prefix, date_prefix, basename] if p]
        s3_key = "/".join([part.strip("/") for part in key_parts])
        logger.info(f"Uploading {local_path} to s3://{S3_BUCKET}/{s3_key}")
        client.upload_file(local_path, S3_BUCKET, s3_key)
        url = f"s3://{S3_BUCKET}/{s3_key}"
        # Try to construct HTTPS URL if region and public access allow
        https_url = None
        try:
            region = S3_REGION or client.meta.region_name
            if region:
                https_url = f"https://{S3_BUCKET}.s3.{region}.amazonaws.com/{s3_key}"
        except Exception:
            pass
        return {"bucket": S3_BUCKET, "key": s3_key, "s3_uri": url, "https_url": https_url}
    except (BotoCoreError, NoCredentialsError, ClientError) as e:
        logger.error(f"S3 upload failed: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected S3 upload error: {e}")
        return {"error": str(e)}

def transcribe_whisper(audio_file):
    logger.info(f"Whisper transcription started for file: {audio_file}")
    start_time = datetime.now()
    
    try:
        # Log file info
        if audio_file and os.path.exists(audio_file):
            file_size = os.path.getsize(audio_file) / (1024 * 1024)  # MB
            logger.info(f"Audio file size: {file_size:.2f}MB")
        else:
            logger.error(f"Audio file not found: {audio_file}")
            return {"error": "Audio file not found"}
        
        logger.info("Starting Whisper transcription...")
        result = whisper_model.transcribe(audio_file, word_timestamps=False, beam_size=1)
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"✓ Whisper transcription completed in {duration:.2f}s")
        logger.info(f"Transcribed text length: {len(result.get('text', ''))}")
        
        return result
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        error_msg = f"Whisper transcription error after {duration:.2f}s: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg}

def transcribe_whisper_filepath(file_path):
    """Whisper transcription function that accepts server-side file paths"""
    logger.info(f"Whisper filepath transcription started for: {file_path}")
    
    # Check if file exists on server
    if not os.path.exists(file_path):
        error_msg = f"File not found on server: {file_path}"
        logger.error(error_msg)
        return {"error": error_msg}
    
    # Use the existing transcribe_whisper function
    return transcribe_whisper(file_path)

def transcribe_vosk(audio_file, sample_rate=24000):
    logger.info(f"VOSK transcription started for file: {audio_file} with sample rate: {sample_rate}")
    start_time = datetime.now()
    
    if vosk_model is None:
        logger.error("VOSK model not available. Transcription disabled.")
        return {"error": "VOSK model not available"}
    
    try:
        # Log file info
        if audio_file and os.path.exists(audio_file):
            file_size = os.path.getsize(audio_file) / (1024 * 1024)  # MB
            logger.info(f"Audio file size: {file_size:.2f}MB")
        else:
            logger.error(f"Audio file not found: {audio_file}")
            return {"error": "Audio file not found"}
        
        # Initialize the recognizer with the model
        logger.info(f"Initializing VOSK recognizer with sample rate: {sample_rate}...")
        recognizer = KaldiRecognizer(vosk_model, sample_rate)
        recognizer.SetWords(True)
        
        # Open the audio file
        logger.info("Processing audio chunks...")
        chunks_processed = 0
        with open(audio_file, "rb") as audio:
            while True:
                # Read a chunk of the audio file
                data = audio.read(4000)
                if len(data) == 0:
                    break
                # Recognize the speech in the chunk
                recognizer.AcceptWaveform(data)
                chunks_processed += 1

        logger.info(f"Processed {chunks_processed} audio chunks")
        result = recognizer.FinalResult()
        result_dict = json.loads(result)
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"✓ VOSK transcription completed in {duration:.2f}s")
        logger.info(f"Transcribed text: {result_dict.get('text', 'No text')}")
        
        return result_dict
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        error_msg = f"VOSK transcription error after {duration:.2f}s: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg}

def transcribe_vosk_filepath(file_path, sample_rate=24000):
    """VOSK transcription function that accepts server-side file paths"""
    logger.info(f"VOSK filepath transcription started for: {file_path}")
    
    # Check if file exists on server
    if not os.path.exists(file_path):
        error_msg = f"File not found on server: {file_path}"
        logger.error(error_msg)
        return {"error": error_msg}
    
    # Use the existing transcribe_vosk function
    return transcribe_vosk(file_path, sample_rate)

def chatterbox_clone(text, audio_prompt=None, exaggeration=0.5, cfg_weight=0.5, temperature=1.0, random_seed=None, output_filename=None):
    logger.info(f"Chatterbox TTS started for text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
    start_time = datetime.now()
    
    try:
        # Log parameters
        logger.info(f"Parameters - exaggeration: {exaggeration}, cfg_weight: {cfg_weight}, temperature: {temperature}, seed: {random_seed}, custom_filename: {output_filename}")
        
        # Create output directory if it doesn't exist
        output_dir = "/app/audio/output"
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory ensured: {output_dir}")
        
        # Determine output filename based on priority: custom filename > audio prompt filename > timestamp
        if output_filename:
            # User provided custom filename
            name_without_ext = os.path.splitext(output_filename)[0]
            final_output_filename = f"{name_without_ext}.wav"
            logger.info(f"Using custom filename: {final_output_filename}")
        elif audio_prompt and os.path.exists(audio_prompt):
            prompt_size = os.path.getsize(audio_prompt) / (1024 * 1024)  # MB
            logger.info(f"Using audio prompt: {audio_prompt} ({prompt_size:.2f}MB)")
            
            # Extract filename from audio prompt path and use it for output
            input_filename = os.path.basename(audio_prompt)
            # Keep the same name but ensure .wav extension
            name_without_ext = os.path.splitext(input_filename)[0]
            final_output_filename = f"{name_without_ext}.wav"
            logger.info(f"Using audio prompt filename: {final_output_filename}")
        else:
            # If no custom filename or audio prompt, create a generic filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_output_filename = f"chatterbox_output_{timestamp}.wav"
            logger.info("No custom filename or audio prompt provided, using timestamp-based filename")
        
        output_path = os.path.join(output_dir, final_output_filename)
        logger.info(f"Output will be saved as: {output_path}")
        
        # Prepare generation parameters
        generation_params = {
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature
        }
        if random_seed is not None:
            torch.manual_seed(random_seed)  # Set seed for reproducibility if provided
            logger.info(f"Set random seed: {random_seed}")

        logger.info("Generating audio...")
        if audio_prompt and os.path.exists(audio_prompt):
            wav = chatterbox_model.generate(text, audio_prompt_path=audio_prompt, **generation_params)
        else:
            wav = chatterbox_model.generate(text, **generation_params)
        
        # Try direct encoding first (faster), fallback to FFmpeg if needed
        logger.info("Converting audio to standard format (mono, 24kHz, 16-bit)...")
        
        try:
            # Method 1: Direct PyTorch conversion (faster)
            logger.info("Attempting direct PyTorch conversion...")
            
            # Convert to mono if stereo
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)
                logger.info("Converted stereo to mono")
            
            # Resample to 24kHz if needed
            original_sr = chatterbox_model.sr
            target_sr = 24000
            if original_sr != target_sr:
                logger.info(f"Resampling from {original_sr}Hz to {target_sr}Hz")
                resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
                wav = resampler(wav)
            
            # Save directly with target format
            logger.info(f"Saving audio directly to: {output_path}")
            torchaudio.save(output_path, wav, target_sr, bits_per_sample=16)
            
            # Verify the file was created successfully
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:  # At least 1KB
                logger.info("✓ Direct PyTorch conversion successful")
                conversion_method = "PyTorch (direct)"
            else:
                raise RuntimeError("Direct conversion produced invalid file")
                
        except Exception as e:
            logger.warning(f"Direct conversion failed: {e}, falling back to FFmpeg...")
            conversion_method = "FFmpeg (fallback)"
            
            # Method 2: FFmpeg fallback (more reliable)
            logger.info("Saving to temporary file for FFmpeg conversion...")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                # Save with original sample rate first
                torchaudio.save(temp_path, wav, chatterbox_model.sr)
            
            logger.info(f"Converting with FFmpeg...")
            
            # Use FFmpeg to convert to desired format
            ffmpeg_cmd = [
                "ffmpeg", "-y",  # -y to overwrite output file
                "-i", temp_path,  # input file
                "-ac", "1",       # mono (1 channel)
                "-ar", "24000",   # 24kHz sample rate
                "-sample_fmt", "s16",  # 16-bit signed integer
                "-f", "wav",      # WAV format
                output_path       # output file
            ]
            
            try:
                logger.info(f"Running FFmpeg command: {' '.join(ffmpeg_cmd)}")
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
                logger.info("✓ FFmpeg conversion completed successfully")
                
                # Clean up temporary file
                os.unlink(temp_path)
                logger.info("Temporary file cleaned up")
                
            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg conversion failed: {e}")
                logger.error(f"FFmpeg stderr: {e.stderr}")
                
                # Last resort: move temp file to final location
                logger.info("Using original audio format (no conversion)")
                os.rename(temp_path, output_path)
                conversion_method = "Original (no conversion)"
                
            except FileNotFoundError:
                logger.error("FFmpeg not found! Using original audio format")
                os.rename(temp_path, output_path)
                conversion_method = "Original (FFmpeg not found)"
        
        logger.info(f"Audio saved as: {output_path} (method: {conversion_method})")
        
        # Log output info
        if os.path.exists(output_path):
            output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"✓ Chatterbox TTS completed in {duration:.2f}s")
            logger.info(f"Generated audio file: {output_path} ({output_size:.2f}MB)")
            
            # Verify audio format with FFprobe if available (restored)
            try:
                ffprobe_cmd = [
                    "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", output_path
                ]
                probe_result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True)
                probe_data = json.loads(probe_result.stdout)
                if "streams" in probe_data and len(probe_data["streams"]) > 0:
                    audio_stream = probe_data["streams"][0]
                    logger.info(
                        f"Audio format verification - Channels: {audio_stream.get('channels', 'unknown')}, "
                        f"Sample Rate: {audio_stream.get('sample_rate', 'unknown')}, "
                        f"Bit Depth: {audio_stream.get('bits_per_sample', 'unknown')}"
                    )
            except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
                logger.info("Could not verify audio format with ffprobe")
            
            # Upload to S3 if configured
            s3_info = s3_upload_file(output_path)
            if s3_info:
                logger.info(f"S3 upload info: {s3_info}")
            # Return detailed result for API
            result = {
                "audio_file": output_path,
                "output_path": output_path,
                "filename": final_output_filename,
                "file_size_mb": round(output_size, 2),
                "generation_time_seconds": round(duration, 2),
                "audio_format": "mono WAV, 24kHz, 16-bit",
                "conversion_method": conversion_method,
                "parameters": {
                    "exaggeration": exaggeration,
                    "cfg_weight": cfg_weight,
                    "temperature": temperature,
                    "random_seed": random_seed,
                    "custom_filename": output_filename
                },
                "text_length": len(text),
                "used_audio_prompt": bool(audio_prompt and os.path.exists(audio_prompt)),
                "s3": s3_info,
            }
            logger.info(f"API result: {result}")
            return result
        else:
            error_msg = f"Generated file not found at {output_path}"
            logger.error(error_msg)
            return {"error": error_msg, "output_path": output_path}
        
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        error_msg = f"Chatterbox cloning error after {duration:.2f}s: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg}

def chatterbox_clone_gradio(text, audio_prompt=None, exaggeration=0.5, cfg_weight=0.5, temperature=1.0, random_seed=None, output_filename=None):
    """Wrapper function for Gradio interface that returns just the audio file path"""
    result = chatterbox_clone(text, audio_prompt, exaggeration, cfg_weight, temperature, random_seed, output_filename)
    
    if isinstance(result, dict):
        if "error" in result:
            return None, result  # Return None for audio, error dict for JSON
        else:
            return result["audio_file"], result  # Return audio file path and full result
    else:
        return result, {"error": "Unexpected result format"}  # Fallback

# Gradio Interface with custom endpoint names and new parameters
logger.info("Setting up Gradio interfaces...")

try:
    logger.info("Creating Whisper interface...")
    whisper_iface = gr.Interface(
        fn=transcribe_whisper,
        inputs=gr.Audio(type="filepath", label="Upload audio for Whisper transcription"),
        outputs=gr.JSON(label="Whisper Result"),
        title="OpenAI Whisper Turbo Transcription",
        api_name="whisper"
    )
    logger.info("✓ Whisper interface created")

    logger.info("Creating Whisper filepath interface...")
    whisper_filepath_iface = gr.Interface(
        fn=transcribe_whisper_filepath,
        inputs=gr.Textbox(label="Server File Path", placeholder="/app/audio/output/filename.wav", info="Path to audio file on server"),
        outputs=gr.JSON(label="Whisper Result"),
        title="OpenAI Whisper Turbo (Server Files)",
        api_name="whisper_filepath"
    )
    logger.info("✓ Whisper filepath interface created")

    logger.info("Creating VOSK interface...")
    vosk_iface = gr.Interface(
        fn=transcribe_vosk,
        inputs=[
            gr.Audio(type="filepath", label="Upload audio for VOSK transcription"),
            gr.Number(label="Sample Rate", value=24000, precision=0)
        ],
        outputs=gr.JSON(label="VOSK Result"),
        title="VOSK Transcription",
        api_name="vosk"
    )
    logger.info("✓ VOSK interface created")

    logger.info("Creating VOSK filepath interface...")
    vosk_filepath_iface = gr.Interface(
        fn=transcribe_vosk_filepath,
        inputs=[
            gr.Textbox(label="Server File Path", placeholder="/app/audio/output/filename.wav", info="Path to audio file on server"),
            gr.Number(label="Sample Rate", value=24000, precision=0)
        ],
        outputs=gr.JSON(label="VOSK Result"),
        title="VOSK Transcription (Server Files)",
        api_name="vosk_filepath"
    )
    logger.info("✓ VOSK filepath interface created")

    logger.info("Creating Chatterbox interface...")
    chatterbox_iface = gr.Interface(
        fn=chatterbox_clone_gradio,
        inputs=[
            gr.Textbox(label="Text to clone"),
            gr.Audio(type="filepath", label="Reference audio (optional for voice cloning)"),
            gr.Slider(minimum=0, maximum=1, value=0.5, label="Exaggeration (emotion intensity)"),
            gr.Slider(minimum=0, maximum=1, value=0.5, label="CFG Weight (pacing control)"),
            gr.Slider(minimum=0.1, maximum=2.0, value=1.0, label="Temperature"),
            gr.Number(label="Random Seed", value=None, precision=0),
            gr.Textbox(label="Output Filename (optional)", placeholder="e.g., my_voice_clone", info="Will be saved as .wav file. If empty, uses audio prompt filename or timestamp.")
        ],
        outputs=[
            gr.Audio(type="filepath", label="Generated Audio"),
            gr.JSON(label="Generation Details")
        ],
        title="Resemble.AI Chatterbox Voice Cloning",
        api_name="chatterbox"
    )
    logger.info("✓ Chatterbox interface created")

    logger.info("Creating tabbed interface...")
    app = gr.TabbedInterface([whisper_iface, whisper_filepath_iface, vosk_iface, vosk_filepath_iface, chatterbox_iface], 
                           ["Whisper", "Whisper Files", "VOSK", "VOSK Files", "Chatterbox"])
    logger.info("✓ All Gradio interfaces created successfully")

except Exception as e:
    logger.error(f"✗ Failed to create Gradio interfaces: {e}")
    logger.error(traceback.format_exc())
    raise

if __name__ == "__main__":
    try:
        logger.info("="*50)
        logger.info("Starting Gradio application...")
        logger.info(f"Server configuration: 0.0.0.0:7860")
        logger.info("="*50)
        
        # Add signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            perform_pod_shutdown(shutdown_reason=f"signal:{signum}", logger=logger)
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Launch the app with detailed logging
        logger.info("Launching Gradio application...")
        app.launch(
            server_name="0.0.0.0", 
            server_port=7860,
            show_error=True,
            quiet=False
        )
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.critical(f"Critical error during application startup: {e}")
        logger.critical(traceback.format_exc())
        
        # Try to clean up resources
        try:
            perform_pod_shutdown(shutdown_reason="startup_exception", logger=logger)
        except:
            pass
            
        sys.exit(1)
    finally:
        # Upload logs to S3 on shutdown
        try:
            perform_pod_shutdown(shutdown_reason="finally", logger=logger)
        except Exception as e:
            logger.error(f"Failed during shutdown: {e}")
        logger.info("Application terminating...")
        logger.info(f"End time: {datetime.now()}")
        logger.info("="*50)