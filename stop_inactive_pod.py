import os
import time
import re
import subprocess

# Configuration
INITIAL_DELAY = 720  # 12 minutes to allow boot
CHECK_INTERVAL = 60  # Check every 60 seconds
INACTIVITY_THRESHOLD = 900  # 15 minutes
LOG_FILE = "/app/app.log"
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
    r"Processing TTS request",  # From your example logging
    r"Completed TTS request"
]

def has_recent_activity():
    try:
        # Check if log file exists
        if not os.path.exists(LOG_FILE):
            print(f"Log file {LOG_FILE} not found.")
            return True  # Assume active if no log file yet (e.g., during boot)

        # Check file modification time
        last_modified = os.path.getmtime(LOG_FILE)
        current_time = time.time()
        if current_time - last_modified < INACTIVITY_THRESHOLD:
            return True  # Log file was modified recently

        # Check recent log lines for activity patterns
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()[-100:]  # Check last 100 lines for efficiency
            for line in lines:
                for pattern in ACTIVITY_PATTERNS:
                    if re.search(pattern, line, re.IGNORECASE):
                        return True  # Found recent activity
        return False
    except Exception as e:
        print(f"Error checking log activity: {e}")
        return True  # Assume active if error to avoid premature shutdown

def stop_pod():
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if pod_id:
        subprocess.run(["runpodctl", "stop", "pod", pod_id])
        print(f"Pod {pod_id} stopped due to inactivity.")
    else:
        print("Error: RUNPOD_POD_ID not found.")

def main():
    print(f"Waiting for {INITIAL_DELAY} seconds to allow application boot...")
    time.sleep(INITIAL_DELAY)
    print("Starting log-based inactivity monitoring.")

    inactive_time = 0
    while True:
        if has_recent_activity():
            inactive_time = 0  # Reset if activity detected
            print("Pod active (recent log activity detected)")
        else:
            inactive_time += CHECK_INTERVAL
            print(f"Inactive for {inactive_time} seconds (no recent log activity)")
            if inactive_time >= INACTIVITY_THRESHOLD:
                print("Inactivity threshold reached. Stopping pod.")
                stop_pod()
                break
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()