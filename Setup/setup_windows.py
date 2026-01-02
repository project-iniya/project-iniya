import shutil
import subprocess
import sys

def exists(cmd):
    return shutil.which(cmd) is not None

def run(cmd):
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)

def ensure_winget():
    if not exists("winget"):
        print("❌ winget not found.")
        print("Please install App Installer from Microsoft Store.")
        sys.exit(1)

def install_ffmpeg():
    if exists("ffmpeg"):
        print("✅ ffmpeg already installed")
    else:
        print("⬇ Installing ffmpeg...")
        ensure_winget()
        run(["winget", "install", "-e", "--id", "Gyan.FFmpeg"])

def install_mpv():
    if exists("mpv"):
        print("✅ mpv already installed")
    else:
        print("⬇ Installing mpv...")
        ensure_winget()
        run(["winget", "install", "-e", "--id", "mpv.net"])

def has_nvidia_gpu():
    try:
        subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        return True
    except Exception:
        return False


def main():
    print("=== Project Iniya setup (Windows) ===")
    install_ffmpeg()
    install_mpv()
    
    if has_nvidia_gpu():
        print("✅ NVIDIA GPU detected")
    else:
        print("No NVIDIA GPU detected. CUDA features will be disabled.")
        print("This project is Highly Dependent on NVIDIA CUDA for performance.")
        print("If you have an NVIDIA GPU, ensure the drivers are installed correctly.")

    print("✅ Setup complete")

if __name__ == "__main__":
    main()
