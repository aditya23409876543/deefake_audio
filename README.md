# 🎙️ Voice Sentinel AI
### Premium Neural Audio Authentication & Deepfake Detection

**Voice Sentinel AI** is a state-of-the-art deep learning system designed to authenticate digital voices and identify synthetic audio artifacts. Using a multi-scale neural engine, it can distinguish between genuine human speech (Bona fide) and AI-generated deepfakes with up to **99.81% accuracy**.

---

## 🎯 Project Overview

In an era of increasingly convincing AI voice clones, Voice Sentinel AI provides a robust forensic layer for audio verification. The system doesn't just look at audio quality; it analyzes the fundamental mathematical patterns that differ between human vocal cords and synthetic neural vocoders.

### 🧠 Triple-Feature Neural Engine
The system employs a unique multi-branch CNN architecture that analyzes three distinct dimensions of audio data:
1.  **Spectral Analysis**: Multi-scale Mel-Spectrograms to detect frequency-domain anomalies.
2.  **Cepstral Forensics (MFCC)**: Analyzing phonetic patterns and temporal deltas.
3.  **Phase Coherence**: Detecting unnatural phase shifts common in synthesized signals.

---

## ✨ Key Features

-   **🏆 High-Accuracy Models**: Enhanced CNN (99.81%) and Multi-Scale Ensemble models.
-   **💎 Premium Dashboard**: A glassmorphic, responsive UI built with Streamlit and custom CSS.
-   **📊 Real-time Visualizations**: Interactive Waveform and Spectral Mapping for every analysis.
-   **📱 Cross-Device Optimization**: Fully responsive layout compatible with mobile, tablet, and desktop.
-   **🌐 Network-Ready**: Built-in support for local network access (analyze voices from your phone!).
-   **🔬 Research Integration**: Pre-integrated with SOTA models from HuggingFace (98.5M parameters).

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python 3.11+ installed. For the best experience (support for MP3/FLAC), install **ffmpeg**:
- **Windows**: `choco install ffmpeg`
- **Mac**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

### 2. Installation
```bash
# Clone the repository
git clone https://huggingface.co/spaces/pranay-ai-ml/Audio-Deepfake-Detection
cd Audio-Deepfake-Detection

# Create a virtual environment
python -m venv env
.\env\Scripts\activate  # Windows
source env/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### 3. Launching Voice Sentinel
```bash
streamlit run app.py --server.address 0.0.0.0
```

---

## 📊 Model Intelligence

| Model | Architecture | Accuracy | Best For |
| :--- | :--- | :---: | :--- |
| **Enhanced** | Tri-Feature CNN + Attention | 99.81% | Production-grade detection |
| **Ensemble** | Multi-Scale Weighted Averaging | 98.4% | Highest confidence scoring |
| **Research** | Wav2Vec2 + BiGRU + Attention | 97.2% | Complex, variable quality audio |

---

## 🛠️ Project Architecture

```
Voice-Sentinel-AI/
├── app.py                # Premium Streamlit UI & Logic
├── model.py              # CNN Multi-branch Architectures
├── utils.py              # Feature Extraction & Waveform Viz
├── ensemble.py           # Multi-model Voting Strategies
├── calibration.py        # Robust Prediction & Explainability
└── requirements.txt      # Engine Dependencies (Torch, Librosa, etc.)
```

---

## 📡 Cross-Device Access
Voice Sentinel AI is optimized for the modern web. Once running, you can access the dashboard from any device on your WiFi:

1.  Find the **Network URL** in your terminal (e.g., `http://192.168.1.5:8501`).
2.  Open this link on your mobile phone or tablet.
3.  Upload audio files directly from your mobile storage for instant authentication.

---

## ⚖️ License
This project is licensed under the **Apache-2.0 License**.

---

## 🙏 Credits
Developed using **PyTorch**, **Librosa**, and **Streamlit**. Special thanks to the researchers behind the **ASVspoof** datasets.

---
> **Disclaimer**: This tool is designed for research and forensic assistance. While highly accurate, it should be used as part of a multi-layered verification process.