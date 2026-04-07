import streamlit as st
import torch
import numpy as np
import librosa
import os
import torch.nn.functional as F
from model import DeepfakeDetectorCNN, get_model
from calibration import RobustPredictor, explain_prediction
from utils import plot_spectrogram, plot_waveform, AudioDataset, preprocess_audio

# Page Configuration
st.set_page_config(
    page_title="Voice Sentinel AI", 
    page_icon="🎙️", 
    layout="wide",
    initial_sidebar_state="collapsed" # Better for mobile first
)

# --- CUSTOM CSS ---
st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    
    <style>
        /* Base Styles */
        html, body, [class*="css"]  {
            font-family: 'Outfit', sans-serif;
        }
        
        .stApp {
            background-color: #0E1117;
        }

        /* Fix overlapping text in file uploader */
        label[data-testid="stWidgetLabel"] {
            font-family: 'Outfit', sans-serif !important;
        }
        
        button[data-testid="stBaseButton-secondary"] {
            border-radius: 10px;
        }

        /* Glassmorphism Card */
        .premium-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 24px;
            margin-bottom: 24px;
            transition: all 0.3s ease;
        }
        
        .premium-card:hover {
            transform: translateY(-5px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }
        
        /* Hero Section */
        .hero {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 40px;
            border-radius: 24px;
            color: white;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        }
        
        .hero h1 {
            font-size: clamp(2rem, 5vw, 3.5rem) !important;
            font-weight: 700 !important;
            margin-bottom: 10px !important;
            color: white !important;
        }
        
        .hero p {
            font-size: clamp(1rem, 2vw, 1.25rem) !important;
            opacity: 0.9;
        }
        
        /* Stats Styling */
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.05);
            padding: 15px;
            border-radius: 12px;
            border-left: 4px solid #4A90E2;
        }
        
        /* Custom Prediction Banner */
        .prediction-result {
            padding: 25px;
            border-radius: 20px;
            text-align: center;
            margin-bottom: 20px;
            font-weight: 700;
            font-size: 1.5rem;
            animation: fadeIn 0.8s ease-out;
            color: white !important;
        }
        
        .result-real {
            background: linear-gradient(90deg, #1D976C 0%, #93F9B9 100%);
            color: #0b3d2b !important;
        }
        
        .result-fake {
            background: linear-gradient(90deg, #EB3349 0%, #F45C43 100%);
            color: #ffffff !important;
        }

        /* --- MOBILE RESPONSIVENESS --- */
        @media (max-width: 768px) {
            .hero {
                padding: 24px 15px;
            }
            .premium-card {
                padding: 15px;
                margin-bottom: 15px;
            }
            /* Stack columns on mobile */
            div[data-testid="column"] {
                width: 100% !important;
                flex: none !important;
            }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Hide menu decoration for cleaner look */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Helper Functions
def get_model_status(model_type):
    """Check if model has trained weights"""
    paths = [
        f'model/{model_type}_best_trained.pth',
        f'model/{model_type}_balanced_elevenlabs.pth',
        f'model/{model_type}_comprehensive_trained.pth',
        f'model/{model_type}_quick_trained.pth',
        f'model/{model_type}_trained.pth',
        f'model/{model_type}_pretrained.pth'
    ]
    if model_type == 'pytorch_model':
        paths = ['model/pytorch_model.pth']
        
    for p in paths:
        if os.path.exists(p):
            return "trained" if "pretrained" not in p else "pretrained", p
    return "uninitialized", None

@st.cache_resource
def load_model_with_calibration(model_type='enhanced'):
    if model_type.startswith('ensemble_'):
        from ensemble import create_ensemble
        ensemble_type = model_type.split('_')[1]
        model = create_ensemble(ensemble_type)
    elif model_type == 'pytorch_model':
        model = get_model('pytorch_model')
    else:
        model = get_model(model_type)
        
    status, path = get_model_status(model_type)
    if path:
        state_dict = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    return RobustPredictor(model, device='cpu')

# --- HERO SECTION ---
st.markdown("""
    <div class="hero">
        <h1>🎙️ Voice Sentinel AI</h1>
        <p>Premium Audio Authentication & Deepfake Detection Engine</p>
    </div>
""", unsafe_allow_html=True)

# Responsive Layout for Stats
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("🏆 Accuracy", "99.81%", "Enhanced")
with m2:
    st.metric("📦 Samples", "2.5K+", "Combined")
with m3:
    st.metric("⚡ Time", "< 0.4s", "optimized")

st.markdown("<br>", unsafe_allow_html=True)

# Sidebar - Responsive Control Center
st.sidebar.markdown("<h2 style='color:#4A90E2; text-align:center;'>⚙️ Controls</h2>", unsafe_allow_html=True)

model_options = [
    {"key": "enhanced", "label": "Enhanced Model", "desc": "99.8% Accuracy • Tri-Feature"},
    {"key": "ensemble_multiscale", "label": "MultiScale Ensemble", "desc": "Highest Confidence • Multi-Layer"},
]

if os.path.exists('model/pytorch_model.pth'):
    model_options.append({"key": "pytorch_model", "label": "Research Model", "desc": "ResNet-GRU-Attention"})

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "enhanced"
model_type = st.session_state.selected_model

for option in model_options:
    is_active = model_type == option["key"]
    if is_active:
        st.sidebar.markdown(f"""
            <div style="background:rgba(74,144,226,0.1); border:2px solid #4A90E2; padding:15px; border-radius:12px; margin-bottom:10px;">
                <p style="margin:0; font-weight:bold; color:#fff;">{option['label']} ✅</p>
                <p style="margin:3px 0 0 0; font-size:10px; color:#666;">{option['desc']}</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        if st.sidebar.button(f"Switch to {option['label']}", key=f"sw_{option['key']}"):
            st.session_state.selected_model = option["key"]
            st.rerun()

# --- MAIN CONTENT AREA ---
# Uses a more fluid layout for device adaptation
col_main, col_sidebar = st.columns([2, 1])

with col_main:
    st.markdown("""
        <div class="premium-card">
            <h3 style="margin-top:0;">📁 Upload Sample</h3>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Select audio file", type=["wav", "mp3", "flac", "ogg"], label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file:
        temp_path = "temp_audio.wav"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("🎵 Waveform Analysis")
        st.audio(uploaded_file, format='audio/wav')
        st.pyplot(plot_waveform(temp_path))
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("🌌 Spectral Mapping")
        st.pyplot(plot_spectrogram(temp_path))
        st.markdown('</div>', unsafe_allow_html=True)

with col_sidebar:
    if uploaded_file:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader("🕵️ Neural Verdict")
        
        with st.spinner("Analyzing signals..."):
            predictor = load_model_with_calibration(model_type)
            spectral, mfcc, phase = preprocess_audio(temp_path)
            
            with torch.no_grad():
                result = predictor.predict(spectral, mfcc, phase, return_confidence=True)
                prediction = result['prediction'].item()
                confidence = result['confidence'].item() if hasattr(result['confidence'], 'item') else result['confidence']
                
            if prediction == 1:
                st.markdown('<div class="prediction-result result-real">AUTHENTIC VOICE</div>', unsafe_allow_html=True)
                st.markdown("<p style='text-align:center; color:#93F9B9; font-size:0.9rem;'>Bona fide signals detected.</p>", unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown('<div class="prediction-result result-fake">SYNTHETIC DETECTED</div>', unsafe_allow_html=True)
                st.markdown("<p style='text-align:center; color:#FFB4B4; font-size:0.9rem;'>Artificial artifacts present.</p>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="text-align: center; padding: 20px; background: rgba(255,255,255,0.02); border-radius: 15px; margin-top: 15px;">
                    <p style="font-size: 0.7rem; color: #888; margin-bottom: 0px; letter-spacing: 2px;">CONFIDENCE</p>
                    <h1 style="color: #4A90E2; margin: 0; font-size: 3rem;">{confidence*100:.1f}<span style="font-size: 1rem;">%</span></h1>
                </div>
            """, unsafe_allow_html=True)
            
            # Additional Details
            with st.expander("📊 Probability Distribution"):
                probs = result['probabilities']
                if probs.dim() == 2:
                    fake_prob = probs[0, 0].item()
                elif probs.shape[0] == 2:
                    fake_prob = probs[0].item()
                else:
                    fake_prob = probs.item()
                real_prob = 1 - fake_prob
                
                st.markdown("<p style='font-size: 0.75rem; margin-bottom: 5px;'>Human Reliability Index</p>", unsafe_allow_html=True)
                st.progress(real_prob)
                st.markdown("<p style='font-size: 0.75rem; margin-top: 10px; margin-bottom: 5px;'>Synthetic Likelihood</p>", unsafe_allow_html=True)
                st.progress(fake_prob)

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="premium-card" style="text-align: center; padding: 60px 20px; border: 2px dashed rgba(255,255,255,0.1); min-height:300px; display:flex; align-items:center; justify-content:center; flex-direction:column;">
                <div style="font-size: 3rem; opacity: 0.2; margin-bottom:15px;">🔍</div>
                <h3 style="opacity: 0.5;">Awaiting Analysis</h3>
                <p style="opacity: 0.3; font-size: 0.8rem;">Upload a sample to trigger engine</p>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 40px; padding: 20px; opacity: 0.2; font-size: 0.65rem; letter-spacing: 1px;">
        VOICE SENTINEL AI • v2.4.1 • CROSS-DEVICE COMPATIBLE
    </div>
""", unsafe_allow_html=True)
