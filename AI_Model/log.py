import sys
from multiprocessing import Queue
from typing import Optional
from datetime import datetime

# Debug mode flag
from .config import DEBUG_MODE as DEBUG

# Global log queue (optional)
LOG_QUEUE: Optional[Queue] = None


def init_log_queue(queue: Optional[Queue]):
    """
    Call ONCE from main process to enable centralized logging.
    """
    global LOG_QUEUE
    LOG_QUEUE = queue


def log(msg: str, task: str = "GENERAL"):
    if not DEBUG:
        return

    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted = f"[{timestamp}] [{task}] {msg}"

    # If queue exists → send to main process
    if LOG_QUEUE is not None:
        try:
            LOG_QUEUE.put(formatted)
            return
        except Exception:
            pass  # fall back to print

    # Fallback (safe)
    print(formatted, flush=True)


def log_drain_loop(queue, stop_event):
    while not stop_event.is_set() or not queue.empty():
        try:
            msg = queue.get(timeout=0.1)
            print(msg, flush=True)
        except:
            pass
