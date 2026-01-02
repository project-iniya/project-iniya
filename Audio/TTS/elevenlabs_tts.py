import os,dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.play  import play

dotenv.load_dotenv()

# Loads API key from system env
ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not ELEVEN_API_KEY:
    raise ValueError("❌ ELEVENLABS_API_KEY not found.")

elevenlabs = ElevenLabs(api_key=ELEVEN_API_KEY)

VOICE_NAME = "Jessica"  # you can change later

def speak(text: str):
    """
    Speak text using ElevenLabs Flash v2.5 model (fast and natural).
    """
    if not text or not text.strip():
        return

    # Generate speech using ElevenLabs Flash model
    audio_stream = elevenlabs.text_to_speech.convert(
        text=text,
        voice_id="1qEiC6qsybMkmnNdVMbK",
        model_id="eleven_flash_v2",   # <-- HERE is the Flash engine
        output_format="mp3_44100_128"
    )

    play(audio_stream)

if __name__ == "__main__":
    test_text = input("Enter text to speak: ")
    speak(test_text)