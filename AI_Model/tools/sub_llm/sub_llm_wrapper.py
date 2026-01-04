import ollama # type: ignore
import subprocess
import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

from AI_Model.log import log



# Default Local model for Sub Model (can be auto-switched later)
DEFAULT_MODEL = "qwen2.5:0.5b"


def ensure_model_available(model_name: str):
    """
    Checks if the model exists locally. If not, instructs user to pull it.
    This prevents runtime crashes.
    """
    log(f"Checking if model '{model_name}' is installed...", "SUB LLM")

    result = subprocess.getoutput("ollama list")

    if model_name in result:
        log(f"Model '{model_name}' is available.", "SUB LLM")
        return True
    
    log(f"⚠ Model '{model_name}' NOT found.", "SUB LLM")
    print(f"\n>>> You need to install the model first:\n    ollama pull {model_name}\n")
    return False



def chat(messages, model: str = DEFAULT_MODEL) -> str:
    """
    Core function to send messages to the model.
    """
    if not ensure_model_available(model):
        return "Model not installed. Run: ollama pull mixtral:instruct"

    try:
        log("Sending request to Ollama...", "SUB LLM")
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"num_ctx": 4096}
        )
        result = response["message"]["content"]
        log(f"Response received: {result[:120]}...", "SUB LLM")
        return result

    except Exception as e:
        log(f"❌ Error communicating with Ollama: {e}", "SUB LLM")
        return f"Error: {e}"


def ask_sub_model(system_prompt, messages, model: str = DEFAULT_MODEL) -> str:
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
