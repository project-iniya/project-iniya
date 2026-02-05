import subprocess
import sys
from pathlib import Path
import uuid

VENV_DIR = Path(".venv")
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"


def ensure_windows():
    if not sys.platform.startswith("win"):
        print("❌ Project Iniya is only supported on Windows.")
        sys.exit(1)


def ensure_venv():
    """
    Phase 1:
    - Create venv if missing
    - Re-run this script using venv python
    """
    if sys.prefix.endswith(str(VENV_DIR)):
        # Already running inside venv
        return

    if not VENV_DIR.exists():
        print("📦 Creating virtual environment...")
        subprocess.run(
            [sys.executable, "-m", "venv", str(VENV_DIR)],
            check=True
        )
        print("✅ Virtual environment created")

    print("🔁 Restarting setup inside virtual environment...")
    subprocess.run(
        [str(VENV_PYTHON), *sys.argv],
        check=True
    )
    sys.exit(0)


def main_setup():
    """
    Phase 2:
    Runs ONLY inside venv
    """
    print("=== Project Iniya setup (venv) ===")
    print(f"🐍 Using Python: {sys.executable}")

    # Upgrade pip tooling
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        check=True
    )
    print("✅ pip upgraded")

    # Install dependencies

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements/requirements.txt"],
        check=True
    )
    print("✅ Python dependencies installed")

    try:
        import torch  # type: ignore
    except ImportError:
        # Force reinstall torch
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements/torch_cu130.txt"],
            check=True
        )

    if not torch.cuda.is_available():
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"],
            check=True
        )
        print("🧹 Removed any existing torch installations")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements/torch_cu130.txt"],
            check=True
        )        

    print("✅ Python dependencies installed")

    # Windows-specific setup
    subprocess.run(
        [sys.executable, "Setup/setup_windows.py"],
        check=True
    )
    print("✅ Windows setup complete")
    
    # Run asset setup
    subprocess.run(
        [sys.executable, "Setup/setup_assets.py"],
        check=True
    )
    print("✅ Assets downloaded")

    #Creating First Chat
    import AI_Model.config as aiconf
    if aiconf.CURRENT_CHAT_ID: 
        pass
    else:
        aiconf.CURRENT_CHAT_ID = str(uuid.uuid4())
  

if __name__ == "__main__":
    ensure_windows()
    ensure_venv()
    main_setup()
    print("🎉 Setup finished successfully!")
