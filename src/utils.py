import datetime
import os
import sys


class Logger:
    """Dual-output logger: writes to both terminal and a timestamped log file.

    Creates run_YYYYMMDD_HHMMSS.txt in log_dir and maintains a latest_run.txt
    symlink (or plain file on systems where symlinks are unavailable) for quick
    access to the most recent log.
    """

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(log_dir, f"run_{timestamp}.txt")
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
        latest = os.path.join(log_dir, "latest_run.txt")
        try:
            if os.path.islink(latest):
                os.unlink(latest)
            os.symlink(filename, latest)
        except OSError:
            # Fallback for systems without symlink support (e.g. some Windows configs)
            with open(latest, "w", encoding="utf-8") as f:
                f.write(f"Latest run: {filename}\n")

    def write(self, message):
        self.terminal.write(message)  # Writes in the terminal
        self.log.write(message)       # Writes in the txt file

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def _safe_filename(title: str) -> str:
    for ch in [" ", "|", "(", ")", "-", "/", "+", "->"]:
        title = title.replace(ch, "_")
    return title
