from .memory_static import load_static_memory, add_fact, set_preference, delete_fact
from .memory_vector import store_message, search_relevant, delete_vector_memory
from .personality import PERSONALITY, ASSISTANT_NAME
from .llm_wrapper import ask_model, preload_model, unload_all_models
from .tools.tools import TOOLS, parse_tool_call
from .text_cleaner import clean_text
from .log import log
import re

#===== Functions =====

def extract_keywords(text):
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    common = {"and","or","the","is","my","your","of","to","with","in","on","for"}
    keywords = [w for w in words if w not in common and len(w) > 3]
    return list(set(keywords))[:6]

def clean_response(text: str):
    text = text.strip()
    
    # Remove duplicate assistant prefix if present
    if text.lower().startswith(ASSISTANT_NAME.lower()):
        text = text[len(ASSISTANT_NAME):].strip(" :,-")
    
    return text

class BrainAgent:

    def __init__(self):
        log("Initializing agent...", "AGENT")
        self.static_memory = load_static_memory()
        self.history = []
        self.max_history = 10  # ✅ ADD THIS: Limit conversation history
        log("Agent initialized.", "AGENT")
        log("Preloading LLM model...", "AGENT")
        if preload_model():
          log("LLM model preloaded.", "AGENT")
        else:
          log("LLM model preload failed.", "AGENT")

    # ✅ ADD THIS METHOD: Trim conversation history
    def trim_history(self):
        """Keep only the last N exchanges to prevent context overflow"""
        if len(self.history) > self.max_history * 2:  # *2 because user+assistant = 2 messages
            self.history = self.history[-(self.max_history * 2):]
            log(f"History trimmed to last {self.max_history} exchanges", "AGENT")

    # ---------------- Preference Command ----------------
    def detect_preference_command(self, text: str):
        text_lower = text.strip().lower()
        
        # STRICT match – no more triggering on "preferably"
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

TOOL RULES:
- Call a tool ONLY when the user explicitly asks to add, search, list, or complete something.
- If calling a tool, respond ONLY with valid JSON: {{"action":"<name>","input":"<value>"}} 
- After a tool runs, respond normally and do NOT call another tool unless requested again.

IMPORTANT:
- Answer the user's current question directly and naturally
- Don't repeat previous responses
- Be conversational and helpful

-----------------------------------
"""

    # ---------------- Main Thinking Loop ----------------
    def process(self, user_input: str):
        log(f"User: {user_input}", "AGENT")
        
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
        
        # ✅ MODIFIED: Build fresh prompt for each query
        system_prompt = self.build_prompt(user_input)
        
        # ✅ ADD: Trim history before processing
        self.trim_history()
        
        # Add user message to history
        self.history.append({"role": "user", "content": user_input})
        
        # ✅ MODIFIED: Pass system prompt separately
        reply = ask_model(system_prompt, self.history)
        
        # Add assistant reply to history
        self.history.append({"role": "assistant", "content": reply})
        
        action, tool_input = parse_tool_call(reply)
        
        # ==== Tool Path ====
        if action:
            log(f"Tool triggered → {action}", "AGENT")
            
            if action not in TOOLS:
                return f"{ASSISTANT_NAME}: Invalid tool call."
            
            result = TOOLS[action](tool_input)
            
            # Add tool execution into history
            self.history.append({
                "role": "tool",
                "content": f"{action} → {result}"
            })
            
            # ✅ MODIFIED: Get final response without repeating system
            log(f"History length: {len(self.history)}", "DEBUG")
            log(f"Last user message: {self.history[-1] if self.history else 'None'}", "DEBUG")
            final = ask_model(
                "Respond naturally to the user based on the tool result. Don't repeat yourself.",
                [
                    *self.history,
                    {"role": "user", "content": "The tool has completed. Respond normally, do NOT call another tool."}
                ]
            )
            
            self.history.append({"role": "assistant", "content": final})
            return {
                "assistant": ASSISTANT_NAME,
                "text": final,
                "keywords": extract_keywords(final),
                "speak": True,
                "clean_text" : clean_text(final)
            }
        
        # ==== Normal Chat ====
        store_message(f"USER:{user_input}")
        store_message(f"{ASSISTANT_NAME}:{reply}")
        final_response = clean_response(reply)
        
        return {
            "assistant": ASSISTANT_NAME,
            "text": final_response,
            "keywords": extract_keywords(final_response),
            "speak": True,
            "clean_text" : clean_text(final_response)
        }
    
    def cleanup(self):
        log("Unloading LLM model...", "AGENT")
        unload_all_models()
        log("Agent cleanup complete.", "AGENT")