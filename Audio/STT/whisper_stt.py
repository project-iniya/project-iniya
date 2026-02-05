import sounddevice as sd
import numpy as np
import queue
import threading
import time
import torch
import os, sys
from pathlib import Path

BASE_PATH = Path(__file__).parent.parent.parent

CUDA_ROOT = r".\cuda\v12.9"
CUDA_BIN  = os.path.join(BASE_PATH ,CUDA_ROOT, "bin")

os.environ["CUDA_PATH"] = CUDA_ROOT
os.environ["PATH"] = CUDA_BIN + ";" + os.environ.get("PATH", "")

from faster_whisper import WhisperModel
from AI_Model.log import log

SAMPLE_RATE = 16000
CHUNK_SECONDS = 1          # lower = less latency
SILENCE_THRESHOLD = 0.01   # RMS silence threshold
SILENCE_TIMEOUT = 5      # seconds of silence to auto-stop

BLACKLIST_KEYWORDS = [
    "mapper",
    "primary sound",
    "stereo mix",
    "output",
    "speaker"
]

audio_q = queue.Queue()

def callback(indata, frames, time_info, status):
    audio_q.put(indata.copy())


def get_device_by_index(index: int):
    try:
        dev = sd.query_devices(index)
        if dev["max_input_channels"] <= 0:
            raise ValueError("Device is not an input device")

        return {
            "index": index,
            "name": dev["name"],
            "channels": dev["max_input_channels"],
            "samplerate": dev["default_samplerate"],
            "hostapi": sd.query_hostapis(dev["hostapi"])["name"]
        }
    except Exception as e:
        return None

class WhisperSTT:
    def __init__(self):
        self.model = None
        self.running = False
        self.device_index = None  # None = default
        self.transcript = ""
        self.thread = None
        self.last_text = ""
        self.last_voice_time = time.time()

        self._load_model()

    def list_usable_input_devices(self):
        usable = []

        for idx, dev in enumerate(sd.query_devices()):
            name = dev["name"].lower()

            # must be input
            if dev["max_input_channels"] <= 0:
                continue

            # filter obvious junk
            if any(bad in name for bad in BLACKLIST_KEYWORDS):
                continue

            # must support Whisper settings
            try:
                sd.check_input_settings(
                    device=idx,
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32"
                )
            except Exception:
                continue

            usable.append({
                "index": idx,
                "name": dev["name"],
                "channels": dev["max_input_channels"],
                "samplerate": dev["default_samplerate"]
            })

        return usable

    # ---------------- MODEL ----------------
    def _load_model(self):
        if torch.cuda.is_available():
            log("Whisper → GPU", "STT_WHISPER")
            self.model = WhisperModel(
                "tiny",
                device="cuda",
                compute_type="float16"
            )
        else:
            log("Whisper → CPU (int8)", "STT_WHISPER")
            self.model = WhisperModel(
                "tiny",
                device="cpu",
                compute_type="int8"
            )

    # ---------------- PUSH TO TALK ----------------
    def press(self):
        """Call when button is pressed"""
        if self.running:
            return

        self.running = True
        self.transcript = ""
        self.last_text = ""
        self.last_voice_time = time.time()

        self.thread = threading.Thread(
            target=self._listen_loop,
            daemon=True
        )
        self.thread.start()

    def release(self):
        """Call when button is released"""
        self.running = False

    def get_text(self):
        return self.transcript.strip()

    # ---------------- INPUT DEVICES ----------------


    def set_input_device(self, index: int):
      self.device_index = index
      dev = get_device_by_index(index)
      if dev:
          log(f"Input device set to: {dev['name']}", "STT_WHISPER")
          self.SAMPLE_RATE = int(dev["samplerate"])
      else:
          log(f"Invalid input device index: {index}", "STT_WHISPER")
      
    

    # ---------------- INTERNAL LOOP ----------------
    def _listen_loop(self):
        buffer = np.empty((0, 1), dtype=np.float32)
        
        sr = self.SAMPLE_RATE if hasattr(self, 'SAMPLE_RATE') else SAMPLE_RATE
        with sd.InputStream(
            samplerate=sr,
            device=self.device_index,
            channels=1,
            dtype="float32",
            callback=callback
        ):
            while self.running:
                try:
                    data = audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                buffer = np.concatenate((buffer, data))

                # ---- SILENCE DETECTION ----
                rms = np.sqrt(np.mean(np.square(data)))
                if rms > SILENCE_THRESHOLD:
                    self.last_voice_time = time.time()

                # Auto-stop if silence too long
                if time.time() - self.last_voice_time > SILENCE_TIMEOUT:
                    log("Silence detected → auto stop", "STT_WHISPER")
                    self.running = False
                    break

                if len(buffer) >= SAMPLE_RATE * CHUNK_SECONDS:
                    chunk = buffer[:SAMPLE_RATE * CHUNK_SECONDS]
                    buffer = buffer[SAMPLE_RATE * CHUNK_SECONDS:]

                    segments, _ = self.model.transcribe(
                        chunk.flatten(),
                        language="en"
                    )

                    for seg in segments:
                        self._append_dedup(seg.text)

    # ---------------- DEDUPLICATION ----------------
    def _append_dedup(self, text):
        text = text.strip()
        if not text:
            return

        # Prevent Whisper repeats
        if text == self.last_text:
            return

        self.last_text = text
        self.transcript += text + " "


if __name__ == "__main__":
    stt = WhisperSTT()
    print("Available Input Devices:")
    for dev in stt.list_usable_input_devices():
        print(f"{dev['index']}: {dev['name']} ({dev['channels']} channels)")

    device_index = int(input("Select input device index: "))
    stt.set_input_device(device_index)

    input("Press Enter to start recording...")

    stt.press()
    print("Recording... Press Enter to stop.")
    input()
    stt.release()
    print("Transcription Result:")
    print(stt.get_text())

    # Wait for thread to finish
    if stt.thread:
        stt.thread.join