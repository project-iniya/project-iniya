import json
import os
from datetime import datetime

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


def complete_task(identifier: str):
    """
    identifier: either numeric ID as string, or part of description.
    """
    ident = identifier.strip()
    if not ident:
        return "Provide a task ID or part of the task description to complete."

    data = _load_raw()
    tasks = data.get("tasks", [])

    if not tasks:
        return "You have no tasks to complete."

    # Try numeric ID first
    task = None
    if ident.isdigit():
        tid = int(ident)
        for t in tasks:
            if t.get("id") == tid:
                task = t
                break

    # Fallback: fuzzy match by description substring
    if task is None:
        lowered = ident.lower()
        matches = [t for t in tasks if lowered in t.get("description", "").lower()]
        if len(matches) == 1:
            task = matches[0]
        elif len(matches) > 1:
            ids = ", ".join(f"#{m['id']}" for m in matches)
            return f"Multiple tasks matched that phrase. Be more specific. Possible IDs: {ids}"
        else:
            return "I couldn't find any task matching that."

    task["status"] = "done"
    task["completed_at"] = datetime.utcnow().isoformat() + "Z"
    _save_raw(data)

    return f"Marked task #{task['id']} as completed: {task['description']}"
