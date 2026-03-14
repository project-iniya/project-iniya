"""
Configuration file for AI Model settings.
Contains default model settings and other configurations.
"""
# Default Local model for your assistant (can be auto-switched later)
# Not Finalized - you can choose your own preferred model here
# DEFAULT_MODEL = "qwen2.5:7b-instruct-q4_0"
# DEFAULT_MODEL = "qwen3-vl:4b-instruct-q4_K_M"
# DEFAULT_MODEL = "qwen3-vl:4b-thinking-q4_K_M"
# DEFAULT_MODEL = "qwen3-vl:235b-cloud"
# DEFAULT_MODEL = "qwen3-vl:235b-instruct-cloud"
DEFAULT_MODEL = "qwen3-coder-next:cloud"

DEBUG_MODE = True

class SharedState:
    def __init__(self, manager):
        self.data = manager.dict()

    def set_chat(self, chat_id):
        self.data["CURRENT_CHAT_ID"] = chat_id

    def set_audio_mode(self, audio_mode):
        self.data["AUDIO_MODE"] = audio_mode

    def get_chat(self):
        return self.data.get("CURRENT_CHAT_ID", None)
    
    def get_audio_mode(self):
        return self.data.get("AUDIO_MODE", False)