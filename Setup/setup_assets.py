from huggingface_hub import snapshot_download
from pathlib import Path

REPO_ID = "night-games-20/project-Iniya-Assets" 
PROJECT_ROOT = Path(__file__).parent.resolve()

def download_assets():
    print("⬇ Downloading assets from Hugging Face...")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        local_dir=PROJECT_ROOT,
        local_dir_use_symlinks=False,
        allow_patterns=[
            "cuda/**",
            "Audio/**",
            "point_e_models/**",
        ],
    )

    print("✅ Assets ready")

if __name__ == "__main__":
    download_assets()
