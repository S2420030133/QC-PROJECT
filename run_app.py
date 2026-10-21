import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    print("=========================================================")
    print("Launching Quantum Machine Learning Web App...")
    print("=========================================================")
    root = Path(__file__).resolve().parent
    local_python = root / ".venv" / "Scripts" / "python.exe"
    python = str(local_python) if local_python.exists() else sys.executable
    cmd = [python, "-m", "streamlit", "run", str(root / "app.py")]
    try:
        subprocess.run(cmd, cwd=root, check=True)
    except KeyboardInterrupt:
        print("\nWeb app stopped.")
