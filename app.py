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
from datetime import datetime
from chatterbox.tts import ChatterboxTTS

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
logger.info("Starting ChatteVosker Application")
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
vosk_model_path = "/app/models/vosk-model-en-us-0.22"
logger.info(f"Checking VOSK model at: {vosk_model_path}")

try:
    if os.path.exists(vosk_model_path):
        logger.info("Loading VOSK model...")
        vosk_model = Model(vosk_model_path)
        logger.info("✓ VOSK model loaded successfully")
    else:
        error_msg = f"VOSK model not found at {vosk_model_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
except Exception as e:
    logger.error(f"✗ Failed to load VOSK model: {e}")
    logger.error(traceback.format_exc())
    raise

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

logger.info("All models loaded successfully!")

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

def transcribe_vosk(audio_file, sample_rate=16000):
    logger.info(f"VOSK transcription started for file: {audio_file} with sample rate: {sample_rate}")
    start_time = datetime.now()
    
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

def chatterbox_clone(text, audio_prompt=None, exaggeration=0.5, cfg_weight=0.5, temperature=1.0, random_seed=None):
    logger.info(f"Chatterbox TTS started for text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
    start_time = datetime.now()
    
    try:
        # Log parameters
        logger.info(f"Parameters - exaggeration: {exaggeration}, cfg_weight: {cfg_weight}, temperature: {temperature}, seed: {random_seed}")
        
        if audio_prompt:
            if os.path.exists(audio_prompt):
                prompt_size = os.path.getsize(audio_prompt) / (1024 * 1024)  # MB
                logger.info(f"Using audio prompt: {audio_prompt} ({prompt_size:.2f}MB)")
            else:
                logger.warning(f"Audio prompt file not found: {audio_prompt}")
        
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
        
        output_path = "output_audio.wav"
        logger.info(f"Saving audio to: {output_path}")
        torchaudio.save(output_path, wav, chatterbox_model.sr)
        
        # Log output info
        if os.path.exists(output_path):
            output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"✓ Chatterbox TTS completed in {duration:.2f}s")
            logger.info(f"Generated audio file: {output_size:.2f}MB")
        
        return output_path
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        error_msg = f"Chatterbox cloning error after {duration:.2f}s: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return error_msg

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

    logger.info("Creating VOSK interface...")
    vosk_iface = gr.Interface(
        fn=transcribe_vosk,
        inputs=[
            gr.Audio(type="filepath", label="Upload audio for VOSK transcription"),
            gr.Number(label="Sample Rate", value=16000, precision=0)
        ],
        outputs=gr.JSON(label="VOSK Result"),
        title="VOSK Transcription",
        api_name="vosk"
    )
    logger.info("✓ VOSK interface created")

    logger.info("Creating Chatterbox interface...")
    chatterbox_iface = gr.Interface(
        fn=chatterbox_clone,
        inputs=[
            gr.Textbox(label="Text to clone"),
            gr.Audio(type="filepath", label="Reference audio (optional for voice cloning)"),
            gr.Slider(minimum=0, maximum=1, value=0.5, label="Exaggeration (emotion intensity)"),
            gr.Slider(minimum=0, maximum=1, value=0.5, label="CFG Weight (pacing control)"),
            gr.Slider(minimum=0.1, maximum=2.0, value=1.0, label="Temperature"),
            gr.Number(label="Random Seed", value=None, precision=0)
        ],
        outputs=gr.Audio(type="filepath", label="Generated Audio"),
        title="Resemble.AI Chatterbox Voice Cloning",
        api_name="chatterbox"
    )
    logger.info("✓ Chatterbox interface created")

    logger.info("Creating tabbed interface...")
    app = gr.TabbedInterface([whisper_iface, vosk_iface, chatterbox_iface], ["Whisper", "VOSK", "Chatterbox"])
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
        import signal
        
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            if torch.cuda.is_available():
                logger.info("Clearing CUDA cache...")
                torch.cuda.empty_cache()
            logger.info("Application shutdown complete")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Launch the app with detailed logging
        logger.info("Launching Gradio application...")
        app.launch(
            server_name="0.0.0.0", 
            server_port=7860,
            show_error=True,  # Show detailed error messages
            quiet=False       # Enable verbose logging
        )
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.critical(f"Critical error during application startup: {e}")
        logger.critical(traceback.format_exc())
        
        # Try to clean up resources
        try:
            if torch.cuda.is_available():
                logger.info("Attempting to clear CUDA cache...")
                torch.cuda.empty_cache()
        except:
            pass
            
        sys.exit(1)
    finally:
        logger.info("Application terminating...")
        logger.info(f"End time: {datetime.now()}")
        logger.info("="*50)