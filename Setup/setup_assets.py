from huggingface_hub import snapshot_download
from pathlib import Path
import subprocess
import shutil
import sys, os

REPO_ID = "night-games-20/project-Iniya-Assets" 
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

def exists(cmd):
    return shutil.which(cmd) is not None

def run(cmd):
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)


def download_assets():
    print("⬇ Downloading assets from Hugging Face...")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=PROJECT_ROOT,
        local_dir_use_symlinks=False,
        allow_patterns=[
            "cuda/**",
            "Audio/**",
            "point_e_model_cache/**",
            "Visualizers/**",
        ],
    )

    print("✅ Assets ready")

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

if __name__ == "__main__":
    print("=== Setting up Project Iniya assets ===")
    print("Note: This may take a good bit of time depending on your internet connection.")
    download_ollama_models()
    download_assets()
