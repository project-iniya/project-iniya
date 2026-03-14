import json
from AI_Model.tools.directives import protocol_functions  # file with your functions
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent.parent.parent.parent

# Load registry
with open(BASE_PATH / "AI_Model" / "tools" / "directives" /  "protocol_registry.json", "r") as f:
    registry = json.load(f)


# Build lookup: command -> function
FUNCTION_MAP = {
    name: func
    for name, func in vars(protocol_functions).items()
    if callable(func)
}


def execute_protocol_by_code(code):
    try:
        code = int(code)   # <-- FIX: normalize type
    except:
        print("Invalid protocol code:", code)
        return

    for protocol in registry:
        if protocol["code"] == code:
            command_name = protocol["command"]

            func = FUNCTION_MAP.get(command_name)

            if func:
                print(f"[EXECUTING] {protocol['name']}")
                func()
            else:
                print("Function not found:", command_name)
            return

    print("Protocol not found.")


# Example
if __name__ == "__main__":
    execute_protocol_by_code(17)
