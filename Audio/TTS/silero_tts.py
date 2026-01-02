import torch
import subprocess
import soundfile as sf
from pathlib import Path
import numpy as np
import os, sys

# Add project to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from AI_Model.log import log

BASE_PATH = Path(__file__).parent / "models"
MODEL_PATH = BASE_PATH / "v3_en.pt"

# Create temp output folder
TEMP_DIR = Path(__file__).parent / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# Output file path
OUTPUT_FILE = TEMP_DIR / "silero_output.wav"

language = 'en'
model = torch.package.PackageImporter(str(MODEL_PATH)).load_pickle("tts_models", "model")

if torch.cuda.is_available():
    log("CUDA is available. Using GPU for inference.", "TTS_SILERO")
    model.to("cuda")
else:
    log("CUDA not available. Using CPU for inference.", "TTS_SILERO")
    model.to("cpu")

# Optional: Get available voices
SPEAKERS = model.speakers

# Choose a speaker voice
VOICE = SPEAKERS[10]  # change later after testing

def speak(text: str):
    if not text.strip():
        return

    # Generate audio (list of floats)
    audio = model.apply_tts(text=text, speaker=VOICE, sample_rate=48000)

    # Convert to numpy (float32)
    audio_np = np.array(audio).astype(np.float32)

    # Save using soundfile instead of torchaudio
    sf.write(OUTPUT_FILE, audio_np, 48000)

    # Play with MPV (best compatibility)
    subprocess.run(['mpv', '--really-quiet', str(OUTPUT_FILE)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(str(OUTPUT_FILE))

def list_voices():
    """Print all available speakers."""
    print("\nAvailable Speakers:")
    for idx, voice in enumerate(SPEAKERS):
        print(f"{idx}: {voice}")

if __name__ == "__main__":
    test_text = input("Enter text to speak: ")
    speak(test_text)