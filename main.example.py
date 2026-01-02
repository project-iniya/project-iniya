from multiprocessing import Process, freeze_support, Queue
from threading import Thread, Event
import time

from AI_Model.agent import BrainAgent
from AI_Model.log import log, init_log_queue, log_drain_loop
from Visualizer.point_e_api_serve import main as point_e_main_server

def chat_loop():
    agent = BrainAgent()
    log("Iniya online. Type 'exit' to quit.")

    while True:
        user = input("You: ")
        if user.strip().lower() in {"exit", "quit"}:
            agent.cleanup()
            log("Iniya shutting down. Goodbye!")
            break

        start = time.perf_counter()
        reply = agent.process(user)
        elapsed = time.perf_counter() - start

        log(f"Iniya: {reply['text']}")
        log(f"⏱ {elapsed:.2f}s\n")


if __name__ == "__main__":
    freeze_support()  # REQUIRED on Windows
    log_queue = Queue()
    init_log_queue(log_queue)  # Simple logging to console

    # Flask in separate process
    flask_proc = Process(
        target=point_e_main_server,
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


    # Agent runs in MAIN THREAD
    chat_loop()
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


