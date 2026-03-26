import webview
import base64
from threading import Thread
from pathlib import Path
import json
from AI_Model.log import log
from AI_Model.memory.memory_chat import CurrentChatHistory

BASE_DIR = Path(__file__).resolve().parent.parent

class API:
    def __init__(self):
        self._window = None
        self.info_conn = None
        self.msg_conn = None
        self.chatManager = None

    def setWindow(self, window):
        self._window = window
    
    def setChildConn(self, info_conn, msg_conn):
        self.info_conn = info_conn
        self.msg_conn = msg_conn

    def setChatManager(self, chatID):
        self.chatManager = CurrentChatHistory(chatID)

    def startLister(self):
        Thread(target=self.receiveReply, daemon=True).start()
        Thread(target=self.receiveInfo, daemon=True).start()

    def test_method(self, msg):
        print(f"[FROM VUE] {msg}")
        return f"Python received: {msg}"

    def change_chat(self, payload):
        new = payload.get("new", False)
        chatID = payload.get("chatID", None)
        log(f"change_chat called: new={new}, chatID={chatID}", "PYWEBVIEW")
        if not (new or chatID):
            return {"status": "error", "message": "ChatID not Present"}

        self.info_conn.send(
            {"event": "chatChange", "type": "text"} if new
            else {"event": "chatChange", "chatID": str(chatID)}
        )
        log("change_chat event sent to child", "PYWEBVIEW")

    def handle_change_chat_response(self, msg):
        if msg.get("status") == "success":
            self.chatManager = CurrentChatHistory(msg["chatID"])
            log(f"Chat changed to {msg['chatID']}", "PYWEBVIEW")
        else:
            log(f"Chat change error: {msg.get('error')}", "PYWEBVIEW ERROR")

    def read_chat_history(self):
        return self.chatManager.getChatHistory()

    def receive_image(self, payload):
        return self.chatManager.saveImage(payload)

    def receive_file(self, payload):
        return self.chatManager.saveFile(payload)

    def sendQuestion(self, payload):
        try:
            log(f"sendQuestion received: {payload}", "PYWEBVIEW")
            self.info_conn.send({"event": "message", "data": payload})
            self._window.evaluate_js("window.startBotMessage()")
            log("startBotMessage fired", "PYWEBVIEW")
            return {"status": "ok", "message": "Payload Received"}
        except Exception as e:
            log(f"sendQuestion error: {e}", "PYWEBVIEW ERROR")
            return {"status": "error", "message": str(e)}
    def receiveInfo(self):
        log("receiveInfo listener started", "PYWEBVIEW")
        try:
            while True:
                msg = self.info_conn.recv()
                log(f"INFO MSG RECEIVED: {msg}", "PYWEBVIEW DEBUG")
                event = msg.get("event")

                if event == "chatChange":
                    log(f"chatChange response: {msg}", "PYWEBVIEW DEBUG")
                    self.handle_change_chat_response(msg)

                elif event == "system":
                    log(f"System event: {msg.get('data')}", "PYWEBVIEW")

                else:
                    log(f"Unhandled info event: {event}", "PYWEBVIEW WARN")

        except Exception as e:
            log(f"receiveInfo crashed: {e}", "PYWEBVIEW ERROR")

    def receiveReply(self):
        log("receiveReply listener started", "PYWEBVIEW")
        try:
            while True:
                msg = self.msg_conn.recv()
                log(f"RAW MSG RECEIVED: {msg}", "PYWEBVIEW DEBUG")

                event = msg.get("event")

                if event == "message":
                    data = msg.get("data", {})
                    msg_type = data.get("type")
                    content = data.get("content", {})
                    log(f"event=message | type={msg_type}", "PYWEBVIEW DEBUG")

                    if msg_type == "finalReply":
                        text = content.get("clean_text", {}).get("raw") or content.get("text", "")
                        log(f"finalReply content: {repr(text)}", "PYWEBVIEW DEBUG")
                        self._window.evaluate_js(f"window.finalizeResponse({json.dumps({'content': text})})")

                    elif msg_type == "toolUse":
                        x = {
                            "label": content.get("name", ""),
                            "detail": content.get("agent_reply", "")[:60],
                            "fullDetail": content.get("agent_reply", ""),
                            "attempt": content.get("attempt", 1),
                            "input": content.get("input"),
                        }
                        self._window.evaluate_js(f"window.addStep({json.dumps(x)})")

                    elif msg_type == "toolDone":
                        x = {
                            "label": content.get("name", ""),
                            "failed": content.get("failed", False),
                            "result": content.get("result"),
                        }
                        try:
                            if content.get("failed", False):
                                self._window.evaluate_js(f"window.failStep({json.dumps(x)})")
                            else:
                                self._window.evaluate_js(f"window.completeStep({json.dumps(x)})")
                        except Exception as js_err:
                            log(f"JS call failed (toolDone): {js_err}", "PYWEBVIEW WARN")

                    elif msg_type == "toolFailed":
                        x = {
                            "label": content.get("name", ""),
                            "reason": content.get("reason", ""),
                            "attempts": content.get("attempts", 3),
                        }
                        log(f"toolFailed: {x}", "PYWEBVIEW DEBUG")
                        self._window.evaluate_js(f"window.failStep({json.dumps(x)})")

                    elif msg_type in ("memoryUpdate", "taskUpdate"):
                        log(f"{msg_type}: {content}", "PYWEBVIEW DEBUG")

                    else:
                        log(f"Unhandled message type: {msg_type}", "PYWEBVIEW WARN")

                else:
                    log(f"Unhandled event on msg_conn: {event}", "PYWEBVIEW WARN")

        except Exception as e:
            log(f"receiveReply crashed: {e}", "PYWEBVIEW ERROR")
            import traceback
            traceback.print_exc()
            self._window.evaluate_js(f"window.Error({json.dumps({'status': 'error', 'error': str(e)})})")


def main(info_conn=None, msg_conn=None):
    UI_PATH = BASE_DIR / "GUI" / "ui_vue" / "dist" / "index.html"

    if not info_conn:
        print("Info pipeline connection not provided!")
    
    if not msg_conn:
        print("Message pipeline connection not provided!")

    try:
        api = API()
        window = webview.create_window(
            "Iniya",
            UI_PATH.as_uri(),
            width=1200,
            height=800,
            js_api=api,
        )
        api.setWindow(window)
        api.setChildConn(info_conn, msg_conn)
        api.startLister()
        webview.start(debug=True)
        info_conn.send({"event": "system", "data": "shutdown"})
    except Exception as e:
        print("WebView Exception:", e)


if __name__ == "__main__":
    main()