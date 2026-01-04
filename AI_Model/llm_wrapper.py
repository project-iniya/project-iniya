import ollama # type: ignore
import subprocess
from log import log



# Default Local model for your assistant (can be auto-switched later)
DEFAULT_MODEL = "qwen2.5:7b-instruct-q4_0"

#Cloud Models
#DEFAULT_MODEL = "qwen3-vl:235b-cloud"

_MODEL_AVAILABLE_CACHE = set()

def ensure_model_available(model_name: str):
    if model_name in _MODEL_AVAILABLE_CACHE:
        return True

    log(f"Checking if model '{model_name}' is installed...", "LLM")
    result = subprocess.getoutput("ollama list")

    if model_name in result:
        _MODEL_AVAILABLE_CACHE.add(model_name)
        log(f"Model '{model_name}' is available.", "LLM")
        return True

    log(f"⚠ Model '{model_name}' NOT found.", "LLM")
    return False



def chat(messages, model: str = DEFAULT_MODEL) -> str:
    """
    Core function to send messages to the model.
    """
    if not ensure_model_available(model):
        return "Model not installed. Run: ollama pull mixtral:instruct"

    try:
        log("Sending request to Ollama...", "LLM")
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"num_ctx": 4096}
        )
        result = response["message"]["content"]
        log(f"Response received: {result[:120]}...", "LLM")
        return result

    except Exception as e:
        log(f"❌ Error communicating with Ollama: {e}", "LLM")
        return f"Error: {e}"


def ask_model(system_prompt, messages, model: str = DEFAULT_MODEL) -> str:
    """
    Flexible message handler:
    - Accepts string, dict, or list for `messages`
    - Allows custom roles (tool, function, memory, etc.)
    """

    # Start with system prompt only ONCE
    final_messages = [{"role": "system", "content": system_prompt}]

    # 🔹 If messages is a string → treat as user message
    if isinstance(messages, str):
        final_messages.append({"role": "user", "content": messages})

    # 🔹 If messages is a dict → assume it's a valid message object
    elif isinstance(messages, dict):
        final_messages.append(messages)

    # 🔹 If messages is a list → assume it's fully constructed message history
    elif isinstance(messages, list):
        final_messages.extend(messages)

    else:
        raise TypeError(f"Unsupported message type: {type(messages)}. Must be str, dict, or list.")

    # Call the existing chat function
    return chat(final_messages, model=model)

def preload_model(model: str = DEFAULT_MODEL, keep_alive: int = 3600):
    if not ensure_model_available(model):
        return False

    try:
        log(f"Preloading model '{model}' (full warmup)...", "LLM")

        ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Reply with OK."}
            ],
            options={"num_ctx": 4096},
            keep_alive=keep_alive
        )

        log("Model fully warmed and pinned in memory.", "LLM")
        return True

    except Exception as e:
        log(f"❌ Preload failed: {e}", "LLM")
        return False


def unload_model(model: str = DEFAULT_MODEL):
    """
    Forces Ollama to unload the model from RAM/VRAM immediately.
    """

    try:
        log(f"Unloading model '{model}' from memory...", "LLM")

        ollama.chat(
            model=model,
            messages=[{"role": "system", "content": "unload"}],
            keep_alive=0
        )

        log(f"Model '{model}' unloaded successfully.", "LLM")
        return True

    except Exception as e:
        log(f"❌ Failed to unload model '{model}': {e}", "LLM")
        return False
