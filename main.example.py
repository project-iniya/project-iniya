from multiprocessing import Process, freeze_support, Queue, Pipe, Event, Manager
import threading
import time
from pathlib import Path
import sys
import subprocess

from AI_Model.agent import BrainAgent
from AI_Model.memory.memory_chat import MainChatHistory, CurrentChatHistory
from AI_Model.config import SharedState
from AI_Model.log import log, init_log_queue, log_drain_loop
from Visualizer.point_e_api_serve import main as point_e_main_server
from GUI.py_web import main as gui_main


# ================= VENV =================

VENV_DIR = Path(".venv")
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"


def ensure_running_in_venv():
    if sys.prefix.endswith(str(VENV_DIR)):
        return

    if not VENV_PYTHON.exists():
        print("❌ .venv not found. Running setup...")
        subprocess.run([sys.executable, "setup.py"], check=True)

    print("🔁 Restarting inside .venv...")
    subprocess.run([str(VENV_PYTHON), *sys.argv], check=True)
    sys.exit(0)


# ================= CHAT LOOP =================

def receiver_thread(conn, gui_conn, stop_event, chatHistory):
    while not stop_event.is_set():
        if conn.poll(0.1):  # non-blocking check every 100ms
            msg = conn.recv()
            if msg and not msg.get("event") == "finalReply":
                data = {"type": msg["event"], "source": "agent", "content": msg["data"]}
                chatHistory.appendMsg(data)
                gui_conn.send({"event": "message", "function_used":True, "data": data})
            # handle incoming msg while agent is processing
            # e.g. cancellation signal, etc.

def chat_loop(state, conn, shutdown_event, main_chat_history):
    msg_event_pipe = Pipe()
    agent = BrainAgent(chatID=state.get_chat(), audio_mode=state.get_audio_mode(), msgConn=msg_event_pipe[1])
    chatHistory = CurrentChatHistory(state.get_chat())
    log("Iniya online.")

    conn.send({"event": "system", "data": "online"})

    while not shutdown_event.is_set():
        msg = conn.recv()

        if msg["event"] == "system" and msg["data"] == "shutdown":
            agent.cleanup()
            shutdown_event.set()
            break
        if msg["event"] == "chatChange":
            try:
                if msg.get("chatID"):
                    chatInfo = main_chat_history.getChatInfo(msg["chatID"])
                    if not chatInfo:
                        log(f"Failed to get chat info for ID {msg['chatID']}", "ERROR")
                        continue
                    state.set_chat(msg["chatID"])   
                    agent = BrainAgent(chatID=state.get_chat(), audio_mode=chatInfo.get("type", "text"), msgConn=msg_event_pipe[1])
                    chatHistory = CurrentChatHistory(state.get_chat())
                else:
                    id = main_chat_history.addNewChat(type=msg.get("type", "text"))
                    chatInfo = main_chat_history.getChatInfo(id)
                    if not chatInfo:
                        log(f"Failed to get chat info for new chat ID {id}", "ERROR")
                        continue
                    state.set_chat(id)
                    agent = BrainAgent(chatID=state.get_chat(), audio_mode=chatInfo.get("type", "text"), msgConn=msg_event_pipe[1])
                    chatHistory = CurrentChatHistory(state.get_chat())
                conn.send({"event":"chatChange", "status":"success", "chatID": state.get_chat()})
            except Exception as e:
                log(f"Error changing chat: {e}", "ERROR")
                continue

        if msg["event"] != "message":
            continue

        user_input = msg["data"]["content"]
        if not user_input:
            continue

        stop_event = threading.Event()
        t = threading.Thread(target=receiver_thread, args=(msg_event_pipe[0], conn, stop_event,chatHistory), daemon=True)
        t.start()

        start = time.perf_counter()
        reply = agent.process(user_input)
        elapsed = time.perf_counter() - start

        stop_event.set()  # signals thread to stop
        t.join()          # wait for it to cleanly exit

        try:
            log(f"Cleaned: {reply['clean_text']}", "TEXT")
        except:
            pass

        log(f"Iniya: {reply['text']}")
        log(f"Uncertainty: {reply['uncertainty']}")
        log(f"⏱ {elapsed:.2f}s\n")

        data={
                "type": "message",
                "source": "python",
                "fullReply": reply["text"],
                "content": reply["clean_text"]["raw"],
                "timestamp": ""
            }
        
        chatHistory.appendMsg(msg["data"])
        chatHistory.appendMsg(data)

        conn.send({
            "event": "message",
            "data": data
        })


# ================= MAIN =================

if __name__ == "__main__":
    ensure_running_in_venv()
    freeze_support()

    # Shared State
    state = SharedState(manager=Manager())

    # IPC
    gui_parent_conn, gui_child_conn = Pipe()

    # Chat system
    main_chat_history = MainChatHistory()

    # Events
    flask_ready = Event()
    shutdown_event = Event()

    # Logging
    log_queue = Queue()
    init_log_queue(log_queue)

    log_stop_event = Event()
    log_thread = threading.Thread(
        target=log_drain_loop,
        args=(log_queue, log_stop_event),
        daemon=True
    )
    log_thread.start()

    # Flask (Point-E server)
    flask_proc = Process(
        target=point_e_main_server,
        args=(flask_ready,),
        daemon=True
    )
    flask_proc.start()

    # GUI
    Process(target=gui_main, args=(gui_child_conn,), daemon=True).start()

    print("Waiting for systems...")
    flask_ready.wait()
    print("Systems ready.")

    # Chat thread
    threading.Thread(
        target=chat_loop,
        args=(state,gui_parent_conn, shutdown_event, main_chat_history),
        daemon=True
    ).start()

    shutdown_event.wait()
    log("Shutting down...", "SHUTDOWN")

    # Stop logging
    log_stop_event.set()
    log_thread.join(timeout=2)

    # Stop Flask
    flask_proc.terminate()
    flask_proc.join(timeout=2)

    if flask_proc.is_alive():
        log("Flask did not exit cleanly", "SHUTDOWN")

    print("Shutdown complete.")
