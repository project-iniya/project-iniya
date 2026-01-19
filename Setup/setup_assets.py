from huggingface_hub import HfApi, hf_hub_download
from pathlib import Path
import subprocess
import shutil
import sys, os
import time

REPO_ID = "night-games-20/project-Iniya-Assets"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

def exists(cmd):
    return shutil.which(cmd) is not None

def run(cmd):
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)


ALLOW_PREFIXES = (
    "cuda/",
    "Audio/",
    "point_e_model_cache/",
    "Visualizer/",
    "poppler/",
)

MAX_RETRIES = 5

def should_download(path: str) -> bool:
    return path.startswith(ALLOW_PREFIXES)

def download_assets():
    print("⬇ Downloading assets from Hugging Face...")

    api = HfApi()
    files = api.list_repo_files(
        repo_id=REPO_ID,
        repo_type="dataset",
    )

    target_files = [f for f in files if should_download(f)]

    print(f"📦 {len(target_files)} files queued for download")

    for idx, file in enumerate(target_files, 1):
        print(f"[{idx}/{len(target_files)}] {file}")

        local_path = Path(PROJECT_ROOT) / file
        local_path.parent.mkdir(parents=True, exist_ok=True)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=file,
                    local_dir=PROJECT_ROOT,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                break
            except Exception as e:
                print(f"⚠ Retry {attempt}/{MAX_RETRIES} for {file}")
                print(f"  ↳ {e}")
                time.sleep(3)
        else:
            raise RuntimeError(f"❌ Failed to download {file}")

    print("✅ Assets ready")

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
    download_assets()
    download_transformer_models()
