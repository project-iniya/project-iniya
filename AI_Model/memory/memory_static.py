import json
import os
from datetime import datetime
from threading import Lock
from ..log import log


class StaticMemory:
    def __init__(self, chat_id: str, base_dir="AI_Model/memory_cache"):
        self.chat_id = str(chat_id)
        self.memory_file = os.path.join(base_dir, self.chat_id, "static_memory.json")
        self._lock = Lock()

    # ================= FILE SAFETY =================

    def _ensure_dir(self):
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)

    def _default_memory(self):
        return {
            "facts": [],
            "preferences": {}
        }

    def load(self):
        log("Loading static memory...", "STATIC MEMORY")
        self._ensure_dir()

        if not os.path.exists(self.memory_file):
            log("No memory file found. Creating new structure.", "STATIC MEMORY")
            return self._default_memory()

        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "facts" not in data or "preferences" not in data:
                    return self._default_memory()
                return data
        except Exception as e:
            log(f"Memory corrupted, auto-repair: {e}", "STATIC MEMORY")
            return self._default_memory()

    def _atomic_save(self, memory):
        self._ensure_dir()
        tmp = self.memory_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.memory_file)

    # ================= CORE =================

    def save(self, memory: dict):
        with self._lock:
            self._atomic_save(memory)
        log("Static memory saved.", "STATIC MEMORY")

    def add_fact(self, fact: str):
        fact = fact.strip()
        if not fact:
            return

        with self._lock:
            memory = self.load()

            if any(f["text"].lower() == fact.lower() for f in memory["facts"]):
                log("Fact already exists. Skipping.", "STATIC MEMORY")
                return

            memory["facts"].append({
                "text": fact,
                "created_at": datetime.utcnow().isoformat() + "Z"
            })

            self._atomic_save(memory)

        log(f"Added fact: {fact}", "STATIC MEMORY")

    def set_preference(self, key: str, value):
        with self._lock:
            memory = self.load()
            memory["preferences"][key] = {
                "value": value,
                "updated_at": datetime.utcnow().isoformat() + "Z"
            }
            self._atomic_save(memory)

        log(f"Set preference: {key} = {value}", "STATIC MEMORY")

    def delete_fact(self, keyword: str):
        keyword = keyword.lower().strip()

        with self._lock:
            memory = self.load()
            original = len(memory["facts"])

            memory["facts"] = [
                f for f in memory["facts"]
                if keyword not in f["text"].lower()
            ]

            removed = original - len(memory["facts"])
            self._atomic_save(memory)

        if removed:
            log(f"Deleted {removed} facts containing '{keyword}'", "STATIC MEMORY")
            return f"Removed {removed} fact(s)."
        else:
            return "No matching memory found."
