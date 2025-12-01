#  Iniya AI — Personal Autonomous Assistant (Early Prototype)

###  Status: In Early Development

This project is an experimental attempt to build a **fully autonomous AI personal assistant** — one that, in the future, may be capable of interacting with the physical world, rendering holographic objects, and functioning as a real-world companion system.

Right now, this repository contains only the **core AI "brain" layer**, which will later serve as the control center for:

- Speech understanding and natural conversation  
- Cognitive reasoning and memory handling  
- Interaction with hardware and holographic display systems  
- Modular software capabilities (vision, mobility, automation, etc.)

The goal is not just to make another chatbot — but to create a **long-term scalable assistant architecture** that can grow far beyond text-only interaction.

---

##  Software Requirments

- Python 3.13 (Anaconda Python used for Development)
- ffmpeg, mpv for Audio
- Nvidia GPU with Cuda for offline ollama models
- Ollama Cloud model also available (requires messing with llm_wrapper.py for now to enable)

---

##  Vision (Long-Term Goals)

- 🗣️ Natural human-level interaction  
- 🎭 Personality and emotion simulation  
- 📡 Multi-device awareness  
- 🧩 Modular expansion for skills and tools  
- 🔮 Future projection: hologram-style visual interface  
- 🤖 Integration with robotics, AR, smart environments  

This is **not just software — it's a platform in progress.**

---

##  Current Stage

✔ AI reasoning engine (foundation)  
✔ Prompt framework and architecture  
✔ Experimental memory layer  
✔ Voice TTS Added(only the TTS Scripts and Models , No integration)

❌ Voice interface  
❌ Physical holographic output  
❌ Hardware integration  

Those systems will be added in future iterations.

---

##  Tech Stack (Current)

- Python
- Large Language Model logic and assistant framework using Ollama
- Memory and context handling modules
- Modular architecture for future expansion

---

## Audio Stack (Current, TTS Only)

- Main Audio TTS interface is Elevenlabs API with its Flash v2.5 Model
- secondary offline TTS is Silero TTS with its en_v5 TTS Model
- Audio Play using mpv through subprocess module

---

## License! 

This repository is protected under the **Custom Restricted License (CRL-1.0)**.  
Use, modification, execution, or redistribution of the code is **prohibited** without  
explicit written permission from the author.

See the `LICENSE` file for full terms.
