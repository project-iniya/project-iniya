import json
import os
from datetime import datetime
from difflib import SequenceMatcher

TASKS_FILE = "AI_Model/memory/tasks_memory.json"


def _load_raw():
    if not os.path.exists(TASKS_FILE):
        return {"tasks": []}

    try:
        with open(TASKS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"tasks": []}


def _save_raw(data: dict):
    with open(TASKS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def add_task(description: str, priority: str = "normal"):
    description = description.strip()
    if not description:
        return "Task description was empty. Nothing added."

    data = _load_raw()
    tasks = data.get("tasks", [])

    # simple numeric ID
    next_id = (max((t.get("id", 0) for t in tasks), default=0) + 1)

    task = {
        "id": next_id,
        "description": description,
        "status": "pending",
        "priority": priority,
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
    tasks.append(task)
    data["tasks"] = tasks
    _save_raw(data)

    return f"Task #{next_id} added: {description} (priority: {priority})."


def list_tasks(include_done: bool = True):
    data = _load_raw()
    tasks = data.get("tasks", [])

    if not tasks:
        return "You currently have no tasks on record."

    lines = []
    for t in tasks:
        if not include_done and t.get("status") == "done":
            continue
        status_symbol = "✅" if t.get("status") == "done" else "🕒"
        lines.append(
            f"{status_symbol} [#{t.get('id')}] {t.get('description')} "
            f"(priority: {t.get('priority')}, status: {t.get('status')})"
        )

    return "\n".join(lines) if lines else "No matching tasks found."


def load_tasks():
    with open(TASKS_FILE, "r") as f:
        return json.load(f)

def save_tasks(data):
    with open(TASKS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def complete_task(user_input):
    data = load_tasks()
    tasks = data["tasks"]

    # 1. Try matching exact ID
    try:
        numeric_id = int(user_input.strip())
        matches = [t for t in tasks if t["id"] == numeric_id]
        if matches:
            matches[0]["status"] = "done"
            save_tasks(data)
            return f"Task '{matches[0]['description']}' marked as completed."
    except:
        pass

    # 2. Match by name (similarity)
    matches = sorted(
        tasks,
        key=lambda t: SequenceMatcher(None, t["description"].lower(), user_input.lower()).ratio(),
        reverse=True
    )

    top_match = matches[0]

    # Check if there are duplicates with same name
    duplicates = [t for t in tasks if t["description"].lower() == top_match["description"].lower() and t["status"] != "done"]

    if len(duplicates) > 1:
        options = "\n".join([f"{t['id']}: {t['description']} ({t['status']})" for t in duplicates])
        return f"There are multiple matching tasks. Which one do you want to complete?\n\n{options}"

    # Only one → complete it
    top_match["status"] = "done"
    save_tasks(data)
    return f"Task '{top_match['description']}' marked completed."
