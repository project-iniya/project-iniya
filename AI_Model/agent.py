from .memory import memory_chat
from .memory.memory_static import StaticMemory
from .memory.memory_vector import VectorMemory
from .personality import PERSONALITY, ASSISTANT_NAME
from .llm_wrapper import ask_model, preload_model, unload_all_models
from .tools.tools import ToolManager
from .text_cleaner import clean_text
from .log import log

import re, json
from pathlib import Path


# ================= INIT =================

BASE_PATH = Path(__file__).resolve().parent.parent

with open(BASE_PATH / "AI_Model" / "tools" / "directives" / "protocol_registry.json", "r") as f:
    registry = json.load(f)


def extract_keywords(text: str):
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    common = {"and", "or", "the", "is", "my", "your", "of", "to", "with", "in", "on", "for"}
    keywords = [w for w in words if w not in common and len(w) > 3]
    return list(set(keywords))[:6]


def clean_response(text: str):
    text = text.strip()
    if text.lower().startswith(ASSISTANT_NAME.lower()):
        text = text[len(ASSISTANT_NAME):].strip(" :,-")
    return text


def make_reply(text: str, audio_mode: bool = False):
    cleaned = clean_text(text)
    return {
        "assistant": ASSISTANT_NAME,
        "text": text,
        "keywords": extract_keywords(text),
        "speak": audio_mode,
        "clean_text": cleaned if isinstance(cleaned, dict) else {"raw": text},
    }


# ================= AGENT =================

class BrainAgent:

    def __init__(self, chatID, audio_mode: bool = False, msgConn = None):
        log("Initializing agent...", "AGENT")

        # Memory
        self.staMem = StaticMemory(chatID)
        self.static_memory = self.staMem.load()
        self.vector_memory = VectorMemory(chatID)
        self.currentChatManager = memory_chat.CurrentChatHistory(chatID)

        # Tool runtime
        self.tool_manager = ToolManager(chatID)

        self.history = []
        self.max_history = 10
        self.audio_mode = audio_mode
        self._msgConn = msgConn

        log("Agent initialized.", "AGENT")

        log("Preloading LLM model...", "AGENT")
        preload_model()

    # ================= HISTORY =================

    def trim_history(self):
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-(self.max_history * 2):]
            log("History trimmed.", "AGENT")

    # ================= PROMPT =================

    def build_prompt(self, query: str):
        relevant = self.vector_memory.search_relevant(query)

        prefs = self.static_memory.get("preferences", {})
        facts = self.static_memory.get("facts", [])

        mem_text = "\n".join(relevant) if relevant else "No relevant memory."
        pref_text = "\n".join([f"{k}: {v}" for k, v in prefs.items()]) if prefs else "None"

        return f"""
{PERSONALITY}

== USER FACTS ==
{facts if facts else "None recorded"}

== USER PREFERENCES ==
{pref_text}

== MEMORY SEARCH ==
{mem_text}

== protocol List ==
{registry}

IMPORTANT:
- You may perform multiple helpful actions
- Decide yourself if external information is required
- If information is missing or outdated, you may use the search tool
- Use tools only when necessary
- Prefer reasoning before searching
- Answer naturally

You MUST respond ONLY in valid JSON.

JSON FORMAT:
{{
  "answer": string,
  "uncertainty": number,
  "memory_add": list of strings,
  "memory_delete": list of strings,
  "tasks_add": list of strings,
  "tools": list of {{ "name": tool_name, "input": tool_input }}
}}
"""

    # ================= MAIN =================

    def process(self, user_input: str):
        log(f"User: {user_input}", "AGENT")

        system_prompt = self.build_prompt(user_input)
        self.trim_history()

        self.history.append({"role": "user", "content": user_input})
        raw_reply = ask_model(system_prompt, self.history)
        self.history.append({"role": "assistant", "content": raw_reply})

        try:
            data = json.loads(raw_reply)
        except Exception:
            data = {
                "answer": raw_reply,
                "uncertainty": 0.5,
                "memory_add": [],
                "memory_delete": [],
                "tasks_add": [],
                "tools": [],
            }

        responses = []

        # ===== MEMORY DELETE =====
        for item in data.get("memory_delete", []):
            self.staMem.delete_fact(item)
            self.vector_memory.delete(item)
            responses.append(f"Memory '{item}' erased.")
            if self._msgConn:
                self._msgConn.send({
                    "event": "memoryUpdate",
                    "data": {"action": "delete", "content": item}
                })

        # ===== MEMORY ADD =====
        for item in data.get("memory_add", []):
            self.staMem.add_fact(item)
            self.vector_memory.store_message(f"FACT: {item}")
            if self._msgConn:
                self._msgConn.send({
                    "event": "memoryUpdate",
                    "data": {"action": "add", "content": item}
                })


        # ===== TASK ADD =====
        for task in data.get("tasks_add", []):
            self.tool_manager.execute("task_add", task)
            if self._msgConn:
                self._msgConn.send({
                    "event": "taskUpdate",
                    "data": {"action": "add", "content": task}
                })

        # ===== TOOL CHAIN =====
        used_tools = set()

        for _ in range(5):
            tools_list = data.get("tools", [])
            if not tools_list:
                break

            executed = False

            for tool in tools_list:
                name = tool.get("name")
                tool_input = tool.get("input")

                if name in used_tools:
                    continue
                
                if self._msgConn:
                    self._msgConn.send({
                        "event": "toolUse",
                        "data": {"name": name, "input": tool_input, "agent_reply": data.get("answer", "")}
                    })

                used_tools.add(name)
                executed = True

                log(f"Executing tool: {name}", "AGENT")

                result = self.tool_manager.execute(name, tool_input)

                self.history.append({
                    "role": "tool",
                    "content": f"{name} → {result}"
                })

            if not executed:
                break

            raw_reply = ask_model(
                "Continue reasoning using tool results. Respond ONLY in valid JSON.",
                self.history,
            )

            self.history.append({"role": "assistant", "content": raw_reply})

            try:
                data = json.loads(raw_reply)
            except Exception:
                break

        # ===== FINAL =====
        final_text = data.get("answer", "")
        uncertainty = float(data.get("uncertainty", 0.3))

        self.vector_memory.store_message(f"USER:{user_input}")
        self.vector_memory.store_message(f"{ASSISTANT_NAME}:{final_text}")

        final_text = clean_response(final_text)

        if responses:
            final_text = " ".join(responses) + "\n\n" + final_text

        reply = make_reply(final_text, audio_mode=self.audio_mode)
        reply["uncertainty"] = uncertainty

        if self._msgConn:
            self._msgConn.send({
                "event": "finalReply",
                "data": reply
            })
        return reply

    def cleanup(self):
        log("Unloading LLM model...", "AGENT")
        unload_all_models()
        log("Cleanup complete.", "AGENT")
