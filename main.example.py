from multiprocessing import Process, freeze_support, Queue, Pipe, Event
from threading import Thread
import time
from pathlib import Path
import os, sys
import subprocess

from AI_Model.agent import BrainAgent
from AI_Model.log import log, init_log_queue, log_drain_loop
from Visualizer.point_e_api_serve import main as point_e_main_server
from GUI.py_web import main as gui_main


VENV_DIR = Path(".venv")
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"

def ensure_running_in_venv():
    """
    Ensure this script is running inside .venv.
    If not:
      - re-run with .venv Python
      - or exit if .venv does not exist
    """

    # Already inside venv → good
    if sys.prefix.endswith(str(VENV_DIR)):
        return

    # Not inside venv
    if not VENV_PYTHON.exists():
        print("❌ Virtual environment '.venv' not found.")
        print("Running setup.py to create it...")
        subprocess.run(
            [sys.executable, "setup.py"],
            check=True
        )


    print("🔁 Restarting inside virtual environment...")
    subprocess.run(
        [str(VENV_PYTHON), *sys.argv],
        check=True
    )
    sys.exit(0)


def chat_loop(parent_conn,shutdownEvent):
    agent = BrainAgent()
    log("Iniya online. Type 'exit' to quit.")
    parent_conn.send({"event":"system","data":"online"})

    while True:
        msg = parent_conn.recv()
        if msg["event"] == "message":
            user = msg["data"]["content"]
        elif msg["event"] == "system":
            if msg["data"] == "shutdown":
                agent.cleanup()
                log("Iniya shutting down. Goodbye!")
                shutdownEvent.set()
                break
        else:
            continue 
        if user:
            start = time.perf_counter()
            reply = agent.process(user)
            elapsed = time.perf_counter() - start
            user = ""
            try: log(f"Cleaned Text: {reply['clean_text']}", "Text"); 
            except: pass

            log(f"Iniya: {reply['text']}")
            data = {
                "type": "message",
                "source": "python",
                "fullReply":reply["text"],
                "content": reply['clean_text']['raw'],
                "timestamp":""
            }
            parent_conn.send({"event":"message","data":data})
            log(f"⏱ Time Taken:  {elapsed:.2f}s\n")


if __name__ == "__main__":
    ensure_running_in_venv()  
    freeze_support()  # REQUIRED on Windows

    parent_conn , child_conn = Pipe()
    flask_event = Event()
    shutDown_event = Event()
    log_queue = Queue()


    init_log_queue(log_queue)  # Simple logging to console

    # Flask in separate process
    flask_proc = Process(
        target=point_e_main_server,
        args=(flask_event,),
        daemon=True
    )
    flask_proc.start()

    log_stop_event = Event()
    log_drain = Thread(
      target=log_drain_loop,
      args=(log_queue, log_stop_event),
      daemon=True
    )
    log_drain.start()

    Process(target=gui_main, args=(child_conn,), daemon=True).start()

    print("Waiting for systems to be ready...")

    flask_event.wait()

    print("All systems ready. Starting chat.")

    Thread(target=chat_loop,args=(parent_conn,shutDown_event),daemon=True).start()
    shutDown_event.wait()
    log("Shutting down...", "SHUTDOWN")

    # Stop accepting logs
    log_stop_event.set()

    # Stop Flask
    flask_proc.terminate()
    flask_proc.join(timeout=2)

    if flask_proc.is_alive():
        log("Flask did not exit cleanly", "SHUTDOWN")

    # Flush remaining logs
    log_drain.join(timeout=2)

    print("Shutdown complete.")


