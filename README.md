# 🛡️ AI Guard Agent

**NOTE**: final.py is the full verison of the AI Guard. Other files are milestone specific files because work was divided among teammates and hence, this github was used to share files.
SO, SEE THE FINAL.PY VERISON ONLY FOR FULL WORKING

***EE782-ASSIGN2.IPYNB FILE CONTAINS THE COMMAND ACCURACY TESTING PART IN ADDITION TO FULL MODEL OF FINAL.PY, SO IF NEEDED TO SEE COMMAND ACCURACY TESTING, RUN THE .I[PYNB VERSION, IT IS ALSO COMPLETE.***

An intelligent room monitoring system that combines voice activation, face recognition, and escalating AI dialogue to deter intruders. Built for EE782 Advanced Machine Learning.
This project implements an integrated AI guard agent designed to monitor hostel rooms or personal spaces autonomously. The system leverages pre-trained models across multiple modalities:

Voice Activation: Responds to spoken commands like "Guard my room" to activate/deactivate monitoring
Face Recognition: Identifies trusted users (roommates, friends) vs. unknown individuals using face embeddings
Escalating Dialogue: Engages intruders with progressively stern warnings (4 escalation levels) using LLM-generated responses
Multi-lingual Support: Operates in English, Hindi, and Gujarati with culturally appropriate responses
Speaker Verification (Bonus): Optional voice biometrics to ensure only the owner can deactivate the system

## ✨ Features

- 🎤 **Voice Activation** - "Guard my room" to start/stop
- 👤 **Face Recognition** - Identifies trusted users vs intruders
- 💬 **Escalating Dialogue** - 4 levels from friendly to police alert
- 🌍 **Multi-lingual** - English, Hindi, Gujarati support
- 📸 **Intruder Capture** - Auto-saves timestamped photos
- 🔊 **Speaker Verification** - Optional voice biometrics (BONUS)

## 🚀 Quick Start

### Installation

```bash
# Clone repo
git clone https://github.com/yourusername/ai-guard-agent.git
cd ai-guard-agent

# Install dependencies
pip install face_recognition opencv-python numpy pillow
pip install SpeechRecognition pyaudio gtts pygame
pip install google-generativeai

# Optional: Speaker verification
pip install speechbrain torch torchaudio
```

### Setup API Key

Get a free Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey) and add it to `final.py`:

```python
GEMINI_API_KEY = "your_key_here"
```

### Run

```bash
python final.py
```

## 📖 Usage

1. **Enroll trusted users** - Add 2 photos per person
2. **Activate guard mode** - Say "Guard my room" or use menu
3. **Monitor** - System greets trusted users, challenges intruders
4. **Deactivate** - Say "Stop guard"

### Example Flow

```
# Enroll face
Menu > 1. Enroll trusted user
Name: Alice
Photo 1: photos/alice_front.jpg
Photo 2: photos/alice_side.jpg

# Voice activation
Menu > 5. Listen for voice activation
[Speak]: "Guard my room"
🛡️ GUARD MODE ACTIVATED

# System monitors and responds
✅ Trusted user detected: "Welcome back, Alice!"
🚨 Intruder detected: "Hey! New face alert. Got a name?"
```

## 🎭 Escalation Levels

| Level | Trigger | Response Style |
|-------|---------|----------------|
| 1 | First detection | Curious, friendly |
| 2 | 2+ interactions | Firm warning |
| 3 | 3+ interactions | Police threat |
| 4 | 4+ interactions | Emergency lockdown |

## 🛠️ Troubleshooting

**Webcam not opening?**
- Close other apps using camera
- Check system permissions

**PyAudio errors?**
```bash
# macOS
brew install portaudio

# Linux
sudo apt-get install portaudio19-dev

# Windows - download wheel from
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
```

**Face not detected?**
- Ensure good lighting
- Face should be front-facing
- Use images > 640×480 resolution

## 📁 Project Structure

```
ai-guard-agent/
├── final.py                 # Main system
├── photos/                  # Training photos
├── intruders/              # Captured intruder photos (auto-generated)
├── trusted_users.pkl       # Face database (auto-generated)
└── speaker_model.pkl       # Voice model (auto-generated)
```

## 📦 Requirements

- Python 3.8+
- Webcam + Microphone
- Internet (for Gemini API)

## 🎓 Academic Context

**Assignment**: EE782 Programming Assignment 2  
**Focus**: Multi-modal AI integration (vision + speech + language)  
**Milestones**: Voice activation → Face recognition → Escalating dialogue

## 📝 License

MIT License - Educational project for IIT coursework

## 👥 Authors

Built by Dhrumil Lotiya and Krisha Shah 
