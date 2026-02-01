import ollama
import subprocess
from .log import log

from .config import DEFAULT_MODEL

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
    """
    Preload a model and keep it in memory.
    
    Args:
        model: Model name to load
        keep_alive: How long to keep in memory (seconds)
                    Use -1 to keep indefinitely
                    Use 0 to unload immediately after
    """
    if not ensure_model_available(model):
        return False

    try:
        log(f"Preloading model '{model}' (keep_alive={keep_alive}s)...", "LLM")

        # ✅ FIX: Use generate() for faster loading with less overhead
        ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "This is to Start You UP For the Usage of the User So. get Ready.. (dont reply to this msg , reply with a max OK)"},
            ],
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
    Uses generate() with empty prompt for instant unload (<1 second).
    """
    try:
        log(f"Unloading model '{model}' from memory...", "LLM")

        # ✅ FIX: Use generate() instead of chat() - much faster!
        # Empty prompt means no processing, just immediate unload
        ollama.generate(
            model=model,
            prompt="",  # Empty prompt - no processing needed
            keep_alive=0  # Unload immediately
        )

        log(f"Model '{model}' unloaded successfully.", "LLM")
        return True

    except Exception as e:
        log(f"❌ Failed to unload model '{model}': {e}", "LLM")
        return False


def unload_all_models():
    """
    Unload ALL currently loaded Ollama models.
    Useful when you need to free maximum VRAM.
    """
    try:
        log("Checking for loaded Ollama models...", "LLM")
        
        # ✅ Use ollama.ps() to get RUNNING models (not just downloaded ones)
        ps_result = ollama.ps()
        running_models = ps_result.get('models', [])
        
        if not running_models:
            log("No Ollama models currently loaded", "LLM")
            return True
        
        log(f"Found {len(running_models)} loaded model(s)", "LLM")
        
        # Unload each model
        for model_info in running_models:
            model_name = model_info.get('name', model_info.get('model', 'unknown'))
            log(f"Unloading: {model_name}...", "LLM")
            
            try:
                ollama.generate(
                    model=model_name,
                    prompt="",
                    keep_alive=0
                )
                log(f"✓ Unloaded: {model_name}", "LLM")
            except Exception as e:
                log(f"✗ Failed to unload {model_name}: {e}", "LLM")
        
        log("All Ollama models unloaded", "LLM")
        return True
        
    except Exception as e:
        log(f"❌ Error checking/unloading models: {e}", "LLM")
        return False


def get_loaded_models():
    """
    Get list of currently loaded Ollama models.
    Returns: List of model names
    """
    try:
        ps_result = ollama.ps()
        running_models = ps_result.get('models', [])
        
        model_names = []
        for m in running_models:
            name = m.get('name', m.get('model', ''))
            if name:
                model_names.append(name)
        
        return model_names
        
    except Exception as e:
        log(f"Could not get loaded models: {e}", "LLM")
        return []