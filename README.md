# Project Iniya — The Autonomous AI Assistant (Early Prototype)

###  Status: In Early Development

This project is an experimental attempt to build a **fully autonomous AI personal assistant** — one that, in the future, may be capable of interacting with the physical world, rendering holographic objects, and functioning as a real-world companion system.

Right now, this repository contains only the **core AI "brain" layer**, which will later serve as the control center for:

- Speech understanding and natural conversation  
- Cognitive reasoning and memory handling  
- Interaction with hardware and holographic display systems  
- Modular software capabilities (vision, mobility, automation, etc.)

The goal is not just to make another chatbot — but to create a **long-term scalable assistant architecture** that can grow far beyond text-only interaction.

---
## Installation and Usage

- Make sure to have Python and Nvidia Drivers(if using nvidia GPU) installed
- Make sure you have winget, otherwise you have to install mpv, ffmpeg, ollama manually
- Install this Project to your Device Using the Following commands:

```
git clone https://github.com/project-iniya/project-iniya
cd project-iniya
python setup.py
```
- To Try/Test the Project run this command: `python main.example.py`

---
##  Software Requirments

- Python 3.13
- ffmpeg, mpv for Audio
- Nvidia GPU with Cuda for offline ollama models
- Ollama Cloud model also Usable
- Edit `AI_Model/llm_wrapper.py` To Change or Try/Test Models. Defualt Model is `qwen3-coder-next:cloud`
- This is a Windows Only Project. No plans for adding Linux Support

#### CUDA
- This Project Contains Cuda Runtime DLLS which are a Property of Nvidia Coorporation
- This Project Ships with Cuda runtime Dlls of verision v13.0

#### HF Assets
- The Assets such as Models, DLLS, Binaries are Hosted separately on Hugging Face as a Dataset
- Assets Dataset Link - [Project-Iniya-Assets](https://huggingface.co/datasets/night-games-20/project-Iniya-Assets/tree/main)

---

##  Vision (Long-Term Goals)

- 🗣️ Natural human-level interaction  
- 🎭 Personality and emotion simulation  
- 📡 Multi-device awareness  
- 🧩 Modular expansion for skills and tools  
- 🔮 Future projection: hologram-style visual interface  
- 🤖 Integration with robotics, AR, smart environments  

---

##  Current Stage

✔ AI reasoning engine (foundation)  
✔ Prompt framework and architecture  
✔ Experimental memory layer  
✔ Voice TTS Added(only the TTS Scripts and Models, No integration)  
✔ Voice STT Added(only the STT Scripts and Models, No integration)  
✔ Added Visualization(only Scripts)  

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

## Audio Stack (Current)

- Main Audio TTS interface is Elevenlabs API with its Flash v2.5 Model
- secondary offline TTS is Silero TTS with its en_v5 TTS Model
- Audio Play using mpv through subprocess module
- STT using faster-whisper library 
- STT Models used whisper-tiny

---

## Visualiztion

- This Project Uses Point-E which is a Ai Model Provided by OpenAI
- Uses a Custom Build cloudPoint to SDF encoder
- Uses Three.js to use SDF encodings to display Objects 

---

## GUI

- Work in Progress 
- Pywebview based on edge WebView2
- Web built on Vite + Vue With Javascript
- GUI added but not integrated with The Main Script 

---

## License! 

This repository is protected under the **Custom Restricted License (CRL-1.0)**.  
Modification or redistribution of the code is **prohibited** without  
explicit written permission from the author.

See the `LICENSE` file for full terms.
