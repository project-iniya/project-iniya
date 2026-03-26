from .memory import memory_chat
from .memory.memory_static import StaticMemory
from .memory.memory_vector import VectorMemory
from .memory.memory_master import MasterMemory
from .personality import PERSONALITY, ASSISTANT_NAME
from .llm_wrapper import ask_model, preload_model, unload_all_models
from .tools.tools import ToolManager
from .text_cleaner import clean_text
from .log import log

import re, json
from pathlib import Path 
import platform
import sys

#======= HELPERS =======

def get_system_info() -> str:
    return f"""
== SYSTEM INFO ==
OS: {platform.system()} {platform.release()} ({platform.version()})
Platform: {platform.platform()}
Python: {sys.version.split()[0]}
Python executable: {sys.executable}
Shell: {"cmd.exe (Windows)" if platform.system() == "Windows" else "bash"}
""".strip()

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


def extract_json(raw: str) -> dict | None:
    """Strip thinking tokens / markdown and extract the JSON object."""
    # remove <think>...</think> blocks (qwen3 thinking mode)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # strip markdown code fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw).strip()
    raw = re.sub(r"\s*```$", "", raw).strip()
    # find first { ... } block
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(raw[start:end + 1])
    except Exception:
        return None


# ================= AGENT =================

class BrainAgent:

    def __init__(self, chatID, audio_mode: bool = False, msgConn = None):
        log("Initializing agent...", "AGENT")

        # Memory
        self.staMem = StaticMemory(chatID)
        self.static_memory = self.staMem.load()
        self.vector_memory = VectorMemory(chatID)
        self.master_memory = MasterMemory()
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

    def _check_tool_failure(self, name: str, result) -> tuple[bool, str]:
      """Returns (is_failure, reason). Catches unknown tool, status:error, status:blocked."""
      if isinstance(result, str):
          if result.startswith("Unknown tool:"):
              return True, result
          if result.startswith("Tool execution failed:"):
              return True, result
          return False, ""
      if isinstance(result, dict):
          status = result.get("status", "")
          if status in ("error", "blocked"):
              return True, result.get("error") or result.get("reason") or status
          # non-zero exit code is not a tool failure, that's program output
      return False, ""

    # ================= HISTORY =================

    def trim_history(self):
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-(self.max_history * 2):]
            log("History trimmed.", "AGENT")

    # ================= PROMPT =================

    def build_prompt(self, query: str):
        relevant = self.vector_memory.search_relevant(query)
        master_relevant = self.master_memory.search(query)

        prefs = self.static_memory.get("preferences", {})
        facts = self.static_memory.get("facts", [])

        mem_text = "\n".join(relevant) if relevant else "No relevant memory."
        pref_text = "\n".join([f"{k}: {v}" for k, v in prefs.items()]) if prefs else "None"
        master_text = "\n".join(master_relevant) if master_relevant else "None"

        return f"""
{PERSONALITY}

{get_system_info()}

== USER FACTS ==
{facts if facts else "None recorded"}

== USER PREFERENCES ==
{pref_text}

== MASTER MEMORY (cross-chat, long-term facts) ==
{master_text}

== MEMORY SEARCH ==
{mem_text}

== PROTOCOL LIST ==
{registry}

== AVAILABLE CODING TOOLS ==
{json.dumps(self.tool_manager.coding_protocols, indent=2)}


IMPORTANT:
- You may perform multiple helpful actions
- Decide yourself if external information is required
- If information is missing or outdated, you may use the search tool
- Use tools only when necessary
- Prefer reasoning before searching
- Answer naturally
- You may perform multiple helpful actions
- Set "done": false if you still have more steps to do — NEVER say "I'll now do X" and set done: true
- Set "done": true only when you have a complete final answer
- Use tools only when necessary
- OS: {platform.system()} {platform.release()} | Python: {sys.version.split()[0]}
{"- Use 'python' not 'python3' | Use 'dir' not 'ls' | Do NOT use 'echo -e', it does not exist on Windows" if platform.system() == "Windows" else "- Use 'python3'"}
- ALWAYS prefer code_run with 'inputs' list over shell piping for interactive programs
- For stdin on Windows via shell: (echo 10 & echo 20 & echo 30) | python script.py
- Answer naturally

MEMORY RULES:
- memory_add: facts relevant only to this chat session
- memory_add_global: facts important across ALL future chats (user's name, job, preferences, permanent facts)
- Only add to global if it would genuinely be useful in a different unrelated conversation

You MUST respond ONLY in valid JSON.

JSON FORMAT:
{{
  "answer": string,
  "done": boolean,
  "uncertainty": number,
  "memory_add": list of strings,
  "memory_delete": list of strings,
  "memory_add_global": list of strings,
  "memory_delete_global": list of strings,
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

        # ── initial parse ──
        data = extract_json(raw_reply)
        if not data:
            log(f"Initial JSON parse failed, raw: {raw_reply[:200]}", "AGENT WARN")
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
            self._msgConn.send({
                "event": "message",
                "data": {"type": "memoryUpdate", "source": "agent", "content": {"action": "delete", "content": item}}
            })

        # ===== MEMORY ADD =====
        for item in data.get("memory_add", []):
            self.staMem.add_fact(item)
            self.vector_memory.store_message(f"FACT: {item}")
            self._msgConn.send({
                "event": "message",
                "data": {"type": "memoryUpdate", "source": "agent", "content": {"action": "add", "content": item}}
            })

        # ===== GLOBAL MEMORY ADD =====
        for item in data.get("memory_add_global", []):
            self.master_memory.store(item, source_chat_id=self.staMem.chat_id)
            self.vector_memory.store_message(f"GLOBAL FACT: {item}")
            if self._msgConn:
                self._msgConn.send({
                    "event": "message",
                    "data": {"type": "memoryUpdate", "source": "agent", "content": {
                        "action": "add_global", "content": item
                    }}
                })

        # ===== GLOBAL MEMORY DELETE =====
        for item in data.get("memory_delete_global", []):
            self.master_memory.delete(item)
            if self._msgConn:
                self._msgConn.send({
                    "event": "message",
                    "data": {"type": "memoryUpdate", "source": "agent", "content": {
                        "action": "delete_global", "content": item
                    }}
                })
        
        # ===== TASK ADD =====
        for task in data.get("tasks_add", []):
            self.tool_manager.execute("task_add", task)
            self._msgConn.send({
                "event": "message",
                "data": {"type": "taskUpdate", "source": "agent", "content": {"action": "add", "content": task}}
            })

        # ===== TOOL CHAIN =====
        MAX_TOOL_RETRIES = 3
        used_tools = set()

        for iteration in range(8):
            tools_list = data.get("tools", [])
            is_done = data.get("done", True)

            if is_done and not tools_list:
                break

            if not tools_list:
                log("Agent not done but no tools — re-prompting.", "AGENT")
                raw_reply = ask_model(
                    (
                        "You set done=false but provided no tools. "
                        "You MUST now output a tool call to continue. "
                        "Do NOT write explanations or code in 'answer'. "
                        f"Available coding tools: {list(self.tool_manager.coding_protocols.keys())}. "
                        "Respond ONLY in valid JSON with at least one tool in 'tools'."
                    ),
                    self.history,
                )
                self.history.append({"role": "assistant", "content": raw_reply})
                parsed = extract_json(raw_reply)
                if not parsed or not parsed.get("tools"):
                    log("Re-prompt produced no tools, finalizing.", "AGENT WARN")
                    break
                data = parsed
                continue

            executed = False

            for tool in tools_list:
                name = tool.get("name")
                tool_input = tool.get("input")

                if name in used_tools:
                    continue

                executed = True
                used_tools.add(name)

                result = None
                success = False

                for attempt in range(MAX_TOOL_RETRIES):
                    self._msgConn.send({
                        "event": "message",
                        "data": {"type": "toolUse", "source": "agent", "content": {
                            "name": name,
                            "input": tool_input,
                            "agent_reply": data.get("answer", ""),
                            "attempt": attempt + 1,
                        }}
                    })

                    log(f"Executing tool: {name} (attempt {attempt + 1}/{MAX_TOOL_RETRIES})", "AGENT")
                    result = self.tool_manager.execute(name, tool_input)
                    is_failure, failure_reason = self._check_tool_failure(name, result)

                    self._msgConn.send({
                        "event": "message",
                        "data": {"type": "toolDone", "source": "agent", "content": {
                            "name": name,
                            "result": str(result)[:500],
                            "failed": is_failure,
                        }}
                    })

                    if not is_failure:
                        success = True
                        self.history.append({
                            "role": "tool",
                            "content": f"{name} → {result}"
                        })
                        break

                    log(f"Tool {name} failed (attempt {attempt + 1}): {failure_reason}", "AGENT WARN")

                    if attempt + 1 >= MAX_TOOL_RETRIES:
                        log(f"Tool {name} failed after {MAX_TOOL_RETRIES} attempts. Giving up.", "AGENT WARN")
                        self._msgConn.send({
                            "event": "message",
                            "data": {"type": "toolFailed", "source": "agent", "content": {
                                "name": name,
                                "reason": failure_reason,
                                "attempts": MAX_TOOL_RETRIES,
                            }}
                        })
                        self.history.append({
                            "role": "tool",
                            "content": f"{name} → FAILED after {MAX_TOOL_RETRIES} attempts: {failure_reason}"
                        })
                        break

                    retry_prompt = (
                        f"Tool '{name}' failed with: {failure_reason}\n"
                        f"Input was: {json.dumps(tool_input)}\n"
                        f"Retry with a corrected tool call. "
                        f"Available coding tools: {list(self.tool_manager.coding_protocols.keys())}. "
                        f"Respond ONLY in valid JSON with a single corrected tool in 'tools'."
                    )
                    self.history.append({"role": "user", "content": retry_prompt})

                    raw_retry = ask_model(
                        "You are retrying a failed tool call. Respond ONLY in valid JSON.",
                        self.history,
                    )
                    self.history.append({"role": "assistant", "content": raw_retry})

                    retry_data = extract_json(raw_retry)
                    if retry_data and retry_data.get("tools"):
                        corrected = retry_data["tools"][0]
                        name = corrected.get("name", name)
                        tool_input = corrected.get("input", tool_input)
                        log(f"LLM self-corrected to: {name}", "AGENT")
                    else:
                        log("LLM failed to produce a corrected tool call.", "AGENT WARN")
                        break

            if not executed:
                if data.get("done", True):
                    break

            raw_reply = ask_model(
                "Continue reasoning using tool results. Respond ONLY in valid JSON.",
                self.history,
            )
            self.history.append({"role": "assistant", "content": raw_reply})

            parsed = extract_json(raw_reply)
            if not parsed:
                log("Tool chain JSON parse failed, stopping loop", "AGENT WARN")
                # preserve last known done state — don't force exit
                data = {"tools": [], "done": data.get("done", True), "answer": data.get("answer", "")}
            else:
                data = parsed

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

        self._msgConn.send({
            "event": "message",
            "data": {"type": "finalReply", "source": "agent", "content": reply}
        })
        return reply

    def cleanup(self):
        log("Unloading LLM model...", "AGENT")
        unload_all_models()
        log("Cleanup complete.", "AGENT")