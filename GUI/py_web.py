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
        self._child_conn = None
        self.chatManager = None

    def setWindow(self, window):
        self._window = window
    
    def setChildConn(self, conn):
        self._child_conn = conn

    def setChatManager(self, chatID):
        self.chatManager = CurrentChatHistory(chatID)

    def startLister(self):
        Thread(target=self.receiveReply, daemon=True).start()

    def test_method(self, msg):
        """
        A test method to demonstrate communication from Vue to Python
        """
        print(f"[FROM VUE] {msg}")
        return f"Python received: {msg}"

    def change_chat(self, payload):
        new = payload.get("new", False)
        chatID = payload.get("chatID", None)
        log(f"test0A {new}, {chatID} ")
        if not (new or chatID):
            return {"status": "error", "message": "ChatID not Present"}

        self._child_conn.send({"event":"chatChange", "type":"text"} if new else {"event":"chatChange", "chatID": str(chatID)})
        log("sent msg")
        
    def handle_change_chat_response(self, msg):
      if msg["status"] == "success":
          self.chatManager = CurrentChatHistory(msg["chatID"])
          log(f"Chat Changed to {msg['chatID']}", "PYWEBVIEW INFO")
      else :
          log(f"Error : {msg['error']}", "PYWEBVIEW ERROR")

    def read_chat_history(self,):
        return self.chatManager.getChatHistory()

    def receive_image(self, payload):
        return self.chatManager.saveImage(payload)

    def receive_file(self, payload):
        return self.chatManager.saveFile(payload)

    def sendQuestion(self, payload):
        try:
            print(f"Recieved Payload: {payload}")
            self._child_conn.send({"event":"message", "data":payload})
            return {"status": "ok", "message":"Payload Recieved"}
        except Exception as e:
            return {"status": "error", "message":str(e)}
    
    def receiveReply(self):
        try:
            while True:
                msg = self._child_conn.recv()
                if msg["event"] == "message":
                    fncUsed = msg.get("function_used")
                    if not fncUsed:
                      self._window.evaluate_js(
                          f"window.sendResponse({json.dumps(msg["data"])})"
                      )
                    else:
                        log(f"Functions used in response: {msg.get("data")}", "PYWEBVIEW INFO")
                elif msg["event"] == "chatChange":
                    self.handle_change_chat_response(msg)
                  
        except Exception as e:
            data = {"status":"error","error":e}
            self._window.evaluate_js(
                f"window.Error({json.dumps(data)})"
            )


def main(child_conn = None):
    UI_PATH = BASE_DIR / "GUI" / "ui_vue" / "dist" / "index.html"

    if not child_conn :
      print("Pipeline connection not provided!")

    try:
        api = API()
        window = webview.create_window(
            "Iniya",
            UI_PATH.as_uri(),
            width=1200,
            height=800,
            js_api= api,
        )
        api.setWindow(window)
        api.setChildConn(child_conn)
        api.startLister()
        webview.start(debug=True)
        child_conn.send({"event":"system", "data":"shutdown"})
    except Exception as e :
        print("WebView Expection:", e)


if __name__ == "__main__":
    main()
