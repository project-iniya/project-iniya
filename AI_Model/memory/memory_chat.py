from AI_Model import log
from pathlib import Path
import json
import datetime
import shutil
import uuid
import base64

BASE_PATH = Path(__file__).resolve().parent.parent.parent
CHAT_MEMORY_PATH = BASE_PATH / "AI_Model" / "memory_cache"
CHAT_LIST_FILE = BASE_PATH / "AI_Model" / "memory_cache" / "chat.jsonl"


class MainChatHistory:
    def __init__(self):

        CHAT_LIST_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not CHAT_LIST_FILE.exists():
            open (CHAT_LIST_FILE, 'w').close()

    def listAllChats(self):
        data_list = []
        with open(CHAT_LIST_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data_list.append(json.loads(line))
                except json.JSONDecodeError:
                    pass   # skip broken line

        return data_list
  
    def getChatInfo(self, id):
        chat_dir = CHAT_MEMORY_PATH / str(id)
        info_file = chat_dir / "chat.json"

        if not info_file.exists():
            log.log(f"Chat info not found for ID {id}", "CHAT")
            return None

        try:
            with open(info_file, "r") as f:
                return json.load(f)
        except Exception as e:
            log.log(f"Error reading chat info for ID {id}: {e}", "ERROR")
            return None

    def addNewChat(self, id:str = uuid.uuid4(), name:str="Default", type:str="text"):
        
        id = str(id)
        data = {
            "name":name,
            "chatID":id,
            "type":type
        }
        data_ext = {
            "name":name,
            "chatID":id,
            "type":type,
            "createdAt": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        newChat = CHAT_MEMORY_PATH / f"{id}"
        newChat.mkdir(parents=True, exist_ok=True)
        with open(newChat / "chat.json", "w") as f:
            json.dump(data_ext, f)

        open(newChat / f"{id}.jsonl", 'a').close()
        
        with open(CHAT_LIST_FILE, "a") as f:
            f.write(json.dumps(data) + "\n")
      
        return str(id)

    def deleteChat(self, id):
        try:
            base = Path(CHAT_MEMORY_PATH).resolve()
            target = (base / str(id)).resolve()

            # --- Validate UUID ---
            try:
                uuid.UUID(str(id))
            except ValueError:
                log.log(f"Deletion refused: Invalid ID {id}", "CHAT")
                return

            # --- Safety: ensure inside chat memory folder ---
            if base not in target.parents:
                log.log(f"Blocked deletion outside base path: {target}", "ERROR")
                return

            # --- Check existence ---
            if not target.exists() or not target.is_dir():
                log.log(f"Chat not found: {id}", "CHAT")
                return

            # --- Delete ---
            shutil.rmtree(target)
            log.log(f"Successfully removed chat {id}", "CHAT")

        except Exception as e:
            log.log(f"Error removing chat {id}: {e}", "ERROR")



class CurrentChatHistory:
    def __init__(self, id):
        self.currentChatID = id
        self.uploadDir = CHAT_MEMORY_PATH / f"{id}" / "uploads"
        self.imageDir = self.uploadDir / "images"
        self.fileDir = self.uploadDir / "files"
        self.chatInfoFile = CHAT_MEMORY_PATH / f"{id}" / "chat.json"
        self.msgFile = CHAT_MEMORY_PATH / f"{id}" / f"{id}.jsonl"

        self.imageDir.mkdir(parents=True, exist_ok=True)
        self.fileDir.mkdir(parents=True, exist_ok=True)
    
    def appendMsg(self, msg):
        with open(self.msgFile, 'a') as f:
            f.write(json.dumps(msg) + "\n")

    def getChatHistory(self):
        history = []
        with open(self.msgFile, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # skip empty lines
                    history.append(json.loads(line))
        return history
    
    def saveImage(self, payload):
        """
        Receives base64 image from Vue
        """
        try:
            name = payload["name"]
            data_url = payload["data"]

            header, encoded = data_url.split(",", 1)
            binary = base64.b64decode(encoded)

            path = self.imageDir / name
            with open(path, "wb") as f:
                f.write(binary)

            print(f"[IMAGE SAVED] {path}")
            return {"status": "ok", "path": str(path)}
        except Exception as e:
            print(f"[ERROR] {e}")
            return {"status": "error", "message": str(e)}
        
    def saveFile(self, payload):
        """
        Receives any file from Vue
        """
        try:
            name = payload["name"]
            data_url = payload["data"]

            header, encoded = data_url.split(",", 1)
            binary = base64.b64decode(encoded)

            path = self.fileDir / name
            with open(path, "wb") as f:
                f.write(binary)

            print(f"[FILE SAVED] {path}")
            return {"status": "ok", "path": str(path)}
        except Exception as e:
            print(f"[ERROR] {e}")
            return {"status": "error", "message": str(e)}