import json
import os
from .log import log

MEMORY_FILE = "AI_Model/memory/static_memory.json"

def load_static_memory():
    log("Loading static memory...", "STATIC MEMORY")
    if not os.path.exists(MEMORY_FILE):
        log("No memory file found. Creating new memory structure.", "STATIC MEMORY")
        return {"facts": [], "preferences": {}}

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            log(f"Loaded: {data}")
            return data
    except Exception as e:
        log(f"Error reading file: {e}")
        return {"facts": [], "preferences": {}}


def save_static_memory(memory: dict):
    log(f"Saving static memory: {memory}")
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)


def add_fact(fact: str):
    fact = fact.strip()
    if not fact:
        return
    memory = load_static_memory()
    if fact not in memory["facts"]:
        memory["facts"].append(fact)
        save_static_memory(memory)
        log(f"Added fact: {fact}")
    else:
        log("Fact already exists. Skipping.", "STATIC MEMORY")


def set_preference(key: str, value):
    memory = load_static_memory()
    memory["preferences"][key] = value
    save_static_memory(memory)
    log(f"Set preference: {key} = {value}")

def delete_fact(keyword: str):
    keyword = keyword.lower().strip()
    memory = load_static_memory()

    original_count = len(memory["facts"])
    memory["facts"] = [
        f for f in memory["facts"] if keyword not in f.lower()
    ]

    removed_count = original_count - len(memory["facts"])

    save_static_memory(memory)

    if removed_count > 0:
        log(f"Deleted {removed_count} facts containing '{keyword}'")
        return f"Removed {removed_count} stored fact(s) related to '{keyword}'."
    else:
        return f"No stored memory matched '{keyword}'."
