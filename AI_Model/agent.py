from .memory_static import load_static_memory, add_fact, set_preference, delete_fact
from .memory_vector import store_message, search_relevant, delete_vector_memory
from .personality import PERSONALITY, ASSISTANT_NAME
from .llm_wrapper import ask_model
from .tools import TOOLS, parse_tool_call

DEBUG = True

def log(msg):
    if DEBUG:
        print(f"[AGENT] {msg}")


class BrainAgent:

    def __init__(self):
        log("Initializing agent...")
        self.static_memory = load_static_memory()

    # ---------------- Preference Command ----------------
    def detect_preference_command(self, text: str):
        text_lower = text.strip().lower()

        # STRICT match — no more triggering on "preferably"
        if text_lower.startswith("i prefer "):
            return text_lower.replace("i prefer", "").strip()

        if text_lower.startswith("set my preference to"):
            return text_lower.replace("set my preference to", "").strip()

        if text_lower.startswith("preference:"):
            return text_lower.replace("preference:", "").strip()

        return None

    # ---------------- Memory Deletion ----------------
    def detect_memory_delete(self, text: str):
        text = text.lower()

        triggers = ["forget ", "erase ", "delete memory ", "remove memory of "]

        for t in triggers:
            if text.startswith(t):
                return text.replace(t, "").strip()

        return None

    # ---------------- Prompt Builder -------------------
    def build_prompt(self, query: str):
        relevant = search_relevant(query)

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

TOOL EXECUTION RULES:
-----------------------------------
- You NEVER execute tasks yourself.
- If the user asks for something that can be handled by a tool, you MUST return ONLY a JSON tool call.

FORMAT MUST BE:
{"action": "<tool_name>", "input": "<arguments>"}

- DO NOT reply with text, explanations, confirmations, or polite responses.
- DO NOT assume the tool ran automatically.
- DO NOT mark tasks as complete yourself — ONLY the tool does that.
- After a tool result is returned to you, THEN you may respond naturally.

If unsure whether a tool applies, ALWAYS default to calling one.
-----------------------------------
"""

    # ---------------- Main Thinking Loop ----------------
    def process(self, user_input: str):
        log(f"User: {user_input}")

        # ==== Memory Delete Handling ====
        delete_topic = self.detect_memory_delete(user_input)
        if delete_topic:
            delete_fact(delete_topic)   # delete static memory
            delete_vector_memory(delete_topic)    # delete vector memory
            return f"{ASSISTANT_NAME}: Memory related to '{delete_topic}' has been erased."

        # ==== Preference Handling ====
        pref = self.detect_preference_command(user_input)
        if pref:
            set_preference("response_style", pref)
            store_message(f"SET_PREFERENCE: {pref}")
            return f"{ASSISTANT_NAME}: Preference updated to '{pref}'."

        # ==== Memory Store ====
        if user_input.lower().startswith("remember"):
            fact = user_input.replace("remember", "").strip()
            add_fact(fact)
            store_message(f"FACT: {fact}")
            return f"{ASSISTANT_NAME}: Stored and acknowledged."

        # ==== LLM Response ====
        system_prompt = self.build_prompt(user_input)
        reply = ask_model(system_prompt, user_input)

        action, tool_input = parse_tool_call(reply)

        # ==== Tool Path ====
        if action:
            log(f"Tool triggered → {action}")

            if action not in TOOLS:
                return f"{ASSISTANT_NAME}: Invalid tool call."

            result = TOOLS[action](tool_input)

            # Correct follow-up format to prevent new tool invocation
            tool_message = {
                "role": "tool",
                "name": action,
                "content": str(result)
            }

            # Now ask the model to continue the conversation
            final = ask_model(system_prompt, tool_message)

            store_message(f"RESULT: {result}")
            store_message(f"{ASSISTANT_NAME}:{final}")

            return f"{ASSISTANT_NAME}: {final}"

        # ==== Normal Chat ====
        store_message(f"USER:{user_input}")
        store_message(f"{ASSISTANT_NAME}:{reply}")
        return reply
