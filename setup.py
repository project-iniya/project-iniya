import subprocess, sys

def main():
    print("=== Project Iniya setup ===")

    if sys.platform.startswith("win"):
        pass  # Windows-specific setup
    else:
        print("Project Iniya Is only Supported on Windows.")
        sys.exit(1)

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True
        )
        print("✅ Python dependencies installed")
    except subprocess.CalledProcessError:
        print("❌ Failed to install Python dependencies")
        sys.exit(1)

    try:
        subprocess.run(
            [sys.executable, "Setup/setup_assets.py"],
            check=True
        )
        print("✅ Assets downloaded")
    except subprocess.CalledProcessError:
        print("❌ Failed to download assets")
        sys.exit(1)
    
    try:
        subprocess.run(
            [sys.executable, "Setup/setup_windows.py"],
            check=True
        )
        print("✅ Windows setup complete")
    except subprocess.CalledProcessError:
        print("❌ Failed Windows setup")
        sys.exit(1)    

if __name__ == "__main__":
    main()
    print("🎉 Setup finished successfully!")     