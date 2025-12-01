import ollama
import subprocess

DEBUG = True

def log(msg: str):
    if DEBUG:
        print(f"[LLM] {msg}")


# Default Local model for your assistant (can be auto-switched later)
DEFAULT_MODEL = "qwen2.5:7b-instruct-q4_0"

#Cloud Models
#DEFAULT_MODEL = "qwen3-vl:235b-cloud"

def ensure_model_available(model_name: str):
    """
    Checks if the model exists locally. If not, instructs user to pull it.
    This prevents runtime crashes.
    """
    log(f"Checking if model '{model_name}' is installed...")

    result = subprocess.getoutput("ollama list")

    if model_name in result:
        log(f"Model '{model_name}' is available.")
        return True
    
    log(f"⚠ Model '{model_name}' NOT found.")
    print(f"\n>>> You need to install the model first:\n    ollama pull {model_name}\n")
    return False



def chat(messages, model: str = DEFAULT_MODEL) -> str:
    """
    Core function to send messages to the model.
    """
    if not ensure_model_available(model):
        return "Model not installed. Run: ollama pull mixtral:instruct"

    try:
        log("Sending request to Ollama...")
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"num_ctx": 4096}
        )
        result = response["message"]["content"]
        log(f"Response received: {result[:120]}...")
        return result

    except Exception as e:
        log(f"❌ Error communicating with Ollama: {e}")
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
