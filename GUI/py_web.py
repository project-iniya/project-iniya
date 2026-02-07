import webview
import base64
from threading import Thread
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
IMAGE_DIR = UPLOAD_DIR / "images"
FILE_DIR = UPLOAD_DIR / "files"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
FILE_DIR.mkdir(parents=True, exist_ok=True)


class API:
    def __init__(self):
        self._window = None
        self._child_conn = None

    def setWindow(self, window):
        self._window = window
    
    def setChildConn(self, conn):
        self._child_conn = conn

    def startLister(self):
        Thread(target=self.receiveReply, daemon=True).start()

    def test_method(self, msg):
        """
        A test method to demonstrate communication from Vue to Python
        """
        print(f"[FROM VUE] {msg}")
        return f"Python received: {msg}"

    def receive_image(self, payload):
        """
        Receives base64 image from Vue
        """
        try:
            name = payload["name"]
            data_url = payload["data"]

            header, encoded = data_url.split(",", 1)
            binary = base64.b64decode(encoded)

            path = IMAGE_DIR / name
            with open(path, "wb") as f:
                f.write(binary)

            print(f"[IMAGE SAVED] {path}")
            return {"status": "ok", "path": str(path)}
        except Exception as e:
            print(f"[ERROR] {e}")
            return {"status": "error", "message": str(e)}

    def receive_file(self, payload):
        """
        Receives any file from Vue
        """
        try:
            name = payload["name"]
            data_url = payload["data"]

            header, encoded = data_url.split(",", 1)
            binary = base64.b64decode(encoded)

            path = FILE_DIR / name
            with open(path, "wb") as f:
                f.write(binary)

            print(f"[FILE SAVED] {path}")
            return {"status": "ok", "path": str(path)}
        except Exception as e:
            print(f"[ERROR] {e}")
            return {"status": "error", "message": str(e)}

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
                    self._window.evaluate_js(
                        f"window.sendResponse({json.dumps(msg["data"])})"
                    )
        except Exception as e:
            data = {"status":"error","error":e}
            self._window.evaluate_js(
                f"window.Error({json.dumps(data)})"
            )


def main(child_conn):
    UI_PATH = BASE_DIR / "ui_vue" / "dist" / "index.html"

    api = API()
    window = webview.create_window(
        "My Vue App",
        UI_PATH.as_uri(),
        width=1200,
        height=800,
        js_api= api
    )
    api.setWindow(window)
    api.setChildConn(child_conn)
    api.startLister()
    webview.start(debug=True)
    child_conn.send({"event":"system", "data":"shutdown"})


if __name__ == "__main__":
  
    main()
