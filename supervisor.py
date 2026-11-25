
import subprocess
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / "venv" / "Scripts" / "python.exe"

FLAG = ROOT / ".roboware_shutdown"
URL = "http://127.0.0.1:8000/game.html"

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)


def wait_for_server(timeout=25):
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(URL, timeout=1):
                return True
        except Exception:
            time.sleep(0.5)

    return False


def stop_process(process):
    if process.poll() is not None:
        return

    process.terminate()

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main():
    if not PYTHON.is_file():
        return

    FLAG.unlink(missing_ok=True)
    processes = []

    try:
        processes.append(subprocess.Popen(
            [str(PYTHON), str(ROOT / "app.py")],
            cwd=ROOT,
            creationflags=CREATE_NO_WINDOW
        ))

        processes.append(subprocess.Popen(
            [str(PYTHON), str(ROOT / "hand_tracking.py")],
            cwd=ROOT,
            creationflags=CREATE_NO_WINDOW
        ))

        processes.append(subprocess.Popen(
            [
                str(PYTHON), "-m", "http.server",
                "8000", "--directory", str(ROOT)
            ],
            cwd=ROOT,
            creationflags=CREATE_NO_WINDOW
        ))

        if not wait_for_server():
            return

        webbrowser.open(URL)

        while not FLAG.exists():
            if any(p.poll() is not None for p in processes):
                break
            time.sleep(0.5)

    finally:
        FLAG.touch(exist_ok=True)

        if len(processes) > 1 and processes[1].poll() is None:
            try:
                processes[1].wait(timeout=6)
            except subprocess.TimeoutExpired:
                pass

        for process in processes:
            stop_process(process)

        FLAG.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
