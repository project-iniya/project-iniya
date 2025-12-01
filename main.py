from AI_Model.agent import BrainAgent
#from Audio.elevenlabs_tts import speak
from Audio.silero_tts import speak, list_voices
import time

agent = BrainAgent()

def chat_loop():
    print("Iniya online. Type 'exit' to quit.")
    while True:
        user = input("You: ")
        if user.strip().lower() in {"exit", "quit"}:
            break
        
        start_time = time.perf_counter()   # ⏱ Start timing
        
        reply = agent.process(user)

        end_time = time.perf_counter()     # ⏱ End timing
        elapsed = end_time - start_time

        speak(reply["text"])
        print(f"Iniya: {reply}")
        print(f"⏱ Response time: {elapsed:.2f} seconds\n")  # formatted result

if __name__ == "__main__":
    chat_loop()
