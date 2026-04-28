from pathlib import Path
import subprocess
import shutil
import sys, os
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

def exists(cmd):
    return shutil.which(cmd) is not None

def run(cmd):
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)

# ---------------- OLLAMA (UNCHANGED) ---------------- #

def download_ollama_models():
    if not exists("ollama"):
        print("❌ ollama not found. Skipping Ollama model downloads.")
        sys.exit(1)
        return

    models = [
        "qwen2.5:7b-instruct-q4_0",
        "qwen2.5:0.5b",
        "qwen3-vl:235b-cloud",
    ]

    for model in models:
        print(f"⬇ Downloading Ollama model: {model}...")
        run(["ollama", "pull", model])

    print("✅ Ollama models ready")

# ---------------- Transfromers/Detoxify -------------- #
def download_transformer_models():
    import transformers, detoxify

    print("⬇ Downloading transformer Models")

    transformer_models=["facebook/roberta-hate-speech-dynabench-r4-target"]    
    for tm in transformer_models:
        transformers.AutoTokenizer.from_pretrained(tm)
        transformers.AutoModelForSequenceClassification.from_pretrained(tm)

    print("⬇ Downloading Detoxify Models")
    detoxify.Detoxify("original")
    
    print("✅ Transformer and Detoxify Model Downloaded")

# ---------------- ENTRYPOINT ---------------- #

if __name__ == "__main__":
    print("=== Setting up Project Iniya assets ===")
    print("Note: This may take a good bit of time depending on your internet connection.")
    download_ollama_models()
    download_transformer_models()
