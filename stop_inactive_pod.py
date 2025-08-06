import os
import time
import re
import subprocess
import logging
from datetime import datetime

# Configuration
INITIAL_DELAY = 720  # 12 minutes to allow boot
CHECK_INTERVAL = 60  # Check every 60 seconds
INACTIVITY_THRESHOLD = 900  # 15 minutes until pod stops
LOG_FILE = "/app/app.log"
MONITOR_LOG_FILE = "/app/inactivity_monitor.log"

# Set up logging for the monitor
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(MONITOR_LOG_FILE),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)
ACTIVITY_PATTERNS = [
    r"Whisper transcription started",
    r"VOSK transcription started", 
    r"Chatterbox TTS started",
    r"POST /whisper",
    r"POST /whisper_filepath",
    r"POST /vosk",
    r"POST /vosk_filepath", 
    r"POST /chatterbox",
    r"POST /health",
    r"Processing TTS request",
    r"Completed TTS request",
    r"API result:",  # This will catch your actual log output
    r"generation_time_seconds",  # Another pattern from your log
    r"audio_file.*\.wav",  # Pattern for audio file generation
    r"TTS.*started",  # Generic TTS activity
    r"TTS.*completed",  # Generic TTS completion
    r"Processing.*request",  # Generic request processing
    r"gradio.*request"  # Gradio request patterns
]

def has_recent_activity():
    try:
        # Check if log file exists
        if not os.path.exists(LOG_FILE):
            logger.info(f"Log file {LOG_FILE} not found - assuming active during startup")
            return True  # Assume active if no log file yet (e.g., during boot)

        # Check file modification time
        last_modified = os.path.getmtime(LOG_FILE)
        current_time = time.time()
        time_since_modified = current_time - last_modified
        
        logger.debug(f"Log file last modified {time_since_modified:.1f} seconds ago")
        
        if time_since_modified < INACTIVITY_THRESHOLD:
            logger.debug(f"Log file modified recently (within {INACTIVITY_THRESHOLD}s threshold)")
            return True  # Log file was modified recently

        # Check recent log lines for activity patterns
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()[-100:]  # Check last 100 lines for efficiency
            recent_lines = []
            
            # Look for lines with timestamps within the threshold
            for line in lines:
                # Extract timestamp from log line (assuming format: YYYY-MM-DD HH:MM:SS,mmm)
                if len(line) > 23:  # Minimum length for timestamp
                    timestamp_str = line[:23]
                    try:
                        # Parse the timestamp - handle milliseconds by splitting on comma
                        if ',' in timestamp_str:
                            datetime_part = timestamp_str.split(',')[0]
                            log_time = time.strptime(datetime_part, "%Y-%m-%d %H:%M:%S")
                        else:
                            log_time = time.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                        
                        log_timestamp = time.mktime(log_time)
                        
                        # Check if this log entry is within our threshold
                        if current_time - log_timestamp < INACTIVITY_THRESHOLD:
                            recent_lines.append(line)
                            # Also check for activity patterns in recent lines
                            for pattern in ACTIVITY_PATTERNS:
                                if re.search(pattern, line, re.IGNORECASE):
                                    logger.info(f"Found recent activity pattern: {pattern}")
                                    logger.debug(f"In log line: {line.strip()}")
                                    return True
                    except ValueError as e:
                        # Skip lines that don't have proper timestamp format
                        logger.debug(f"Could not parse timestamp from: {timestamp_str} - {e}")
                        continue
            
            if recent_lines:
                logger.info(f"Found {len(recent_lines)} recent log entries, but no activity patterns matched")
                # Show the most recent few lines for debugging
                for line in recent_lines[-3:]:
                    logger.debug(f"Recent log: {line.strip()}")
            else:
                logger.info("No recent log entries found within threshold")
                
        return False
    except Exception as e:
        logger.error(f"Error checking log activity: {e}")
        return True  # Assume active if error to avoid premature shutdown

def stop_pod():
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if pod_id:
        logger.warning(f"Stopping pod {pod_id} due to inactivity")
        subprocess.run(["runpodctl", "stop", "pod", pod_id])
        logger.info(f"Pod {pod_id} stopped due to inactivity.")
    else:
        logger.error("Error: RUNPOD_POD_ID not found.")

def main():
    logger.info("="*50)
    logger.info("Starting Inactivity Monitor")
    logger.info(f"Initial delay: {INITIAL_DELAY} seconds")
    logger.info(f"Check interval: {CHECK_INTERVAL} seconds") 
    logger.info(f"Inactivity threshold: {INACTIVITY_THRESHOLD} seconds")
    logger.info(f"Monitoring log file: {LOG_FILE}")
    logger.info(f"Monitor log file: {MONITOR_LOG_FILE}")
    logger.info("="*50)
    
    logger.info(f"Waiting for {INITIAL_DELAY} seconds to allow application boot...")
    time.sleep(INITIAL_DELAY)
    logger.info("Starting log-based inactivity monitoring.")

    inactive_time = 0
    while True:
        if has_recent_activity():
            if inactive_time > 0:  # Only log if we were previously inactive
                logger.info(f"Pod became active again after {inactive_time} seconds of inactivity")
            inactive_time = 0  # Reset if activity detected
            logger.debug("Pod active (recent log activity detected)")
        else:
            inactive_time += CHECK_INTERVAL
            logger.info(f"Inactive for {inactive_time} seconds (no recent log activity)")
            if inactive_time >= INACTIVITY_THRESHOLD:
                logger.critical("Inactivity threshold reached. Stopping pod.")
                stop_pod()
                break
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()