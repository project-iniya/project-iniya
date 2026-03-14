import json
import os
import math
import ollama
import re
from datetime import datetime, timedelta
from threading import Lock


class TaskMemory:

    def __init__(
        self,
        chat_id: str,
        base_dir="AI_Model/memory_cache",
        embed_model="mxbai-embed-large",
        similarity_threshold=0.55,
    ):
        self.chat_id = str(chat_id)
        self.tasks_file = os.path.join(base_dir, self.chat_id, "tasks_memory.json")

        self.embed_model = embed_model
        self.similarity_threshold = similarity_threshold

        self._lock = Lock()

    # ================= FILE SAFETY =================

    def _ensure_dir(self):
        os.makedirs(os.path.dirname(self.tasks_file), exist_ok=True)

    def _load_raw(self):
        self._ensure_dir()

        if not os.path.exists(self.tasks_file):
            return {"tasks": []}

        try:
            with open(self.tasks_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "tasks" not in data:
                    return {"tasks": []}
                return data
        except Exception:
            return {"tasks": []}

    def _atomic_save(self, data):
        self._ensure_dir()
        tmp = self.tasks_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.tasks_file)

    # ================= EMBEDDING =================

    def _embed(self, text):
        try:
            res = ollama.embeddings(model=self.embed_model, prompt=text[:2000])
            return res["embedding"]
        except Exception:
            return None

    def _cosine(self, a, b):
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    # ================= INTELLIGENCE =================

    def _infer_priority(self, text):
        t = text.lower()
        if any(w in t for w in ["urgent", "asap", "immediately", "important"]):
            return "high"
        if any(w in t for w in ["later", "someday", "not urgent"]):
            return "low"
        return "normal"

    def _parse_due_date(self, text):
        t = text.lower()
        now = datetime.utcnow()

        if "tomorrow" in t:
            return (now + timedelta(days=1)).isoformat() + "Z"

        if "today" in t or "tonight" in t:
            return (now + timedelta(hours=6)).isoformat() + "Z"

        m = re.search(r"in (\d+) hours?", t)
        if m:
            return (now + timedelta(hours=int(m.group(1)))).isoformat() + "Z"

        m = re.search(r"in (\d+) days?", t)
        if m:
            return (now + timedelta(days=int(m.group(1)))).isoformat() + "Z"

        return None

    # ================= CORE =================

    def add_task(self, description: str, priority=None):
        description = description.strip()
        if not description:
            return "Task description empty."

        with self._lock:
            data = self._load_raw()
            tasks = data["tasks"]

            for t in tasks:
                if t["description"].lower() == description.lower() and t["status"] != "done":
                    return f"Task already exists: #{t['id']}"

            next_id = max((t.get("id", 0) for t in tasks), default=0) + 1

            emb = self._embed(description)

            if priority is None:
                priority = self._infer_priority(description)

            due = self._parse_due_date(description)

            task = {
                "id": next_id,
                "description": description,
                "embedding": emb,
                "status": "pending",
                "priority": priority,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "completed_at": None,
                "due_at": due,
                "tags": [],
                "importance": 0.5,
            }

            tasks.append(task)
            self._atomic_save(data)

        return f"Task #{next_id} added."

    def list_tasks(self, include_done=True):
        data = self._load_raw()
        tasks = data["tasks"]

        if not include_done:
            tasks = [t for t in tasks if t["status"] != "done"]

        if not tasks:
            return "No tasks."

        tasks.sort(
            key=lambda t: (
                {"high": 0, "normal": 1, "low": 2}.get(t["priority"], 1),
                t.get("due_at") or "9999",
            )
        )

        lines = []
        for t in tasks:
            due = f", due: {t['due_at']}" if t.get("due_at") else ""
            status = "✓" if t["status"] == "done" else "•"
            lines.append(
                f"{status} [#{t['id']}] {t['description']} (priority: {t['priority']}{due})"
            )

        return "\n".join(lines)

    # ================= MATCHING =================

    def _find_best_match(self, tasks, text):
        text = text.strip()

        if text.isdigit():
            tid = int(text)
            for t in tasks:
                if t["id"] == tid:
                    return t
            return None

        q_emb = self._embed(text)
        if not q_emb:
            return None

        best = None
        best_score = 0

        for t in tasks:
            emb = t.get("embedding")
            if not emb:
                continue
            s = self._cosine(q_emb, emb)
            if s > best_score:
                best_score = s
                best = t

        return best if best_score >= self.similarity_threshold else None

    # ================= ACTIONS =================

    def complete_task(self, text):
        with self._lock:
            data = self._load_raw()
            task = self._find_best_match(data["tasks"], text)
            if not task:
                return "No matching task."

            if task["status"] == "done":
                return "Task already completed."

            task["status"] = "done"
            task["completed_at"] = datetime.utcnow().isoformat() + "Z"
            self._atomic_save(data)

        return f"Task #{task['id']} completed."

    def delete_task(self, text):
        with self._lock:
            data = self._load_raw()
            task = self._find_best_match(data["tasks"], text)
            if not task:
                return "No matching task."

            data["tasks"].remove(task)
            self._atomic_save(data)

        return f"Task #{task['id']} deleted."
