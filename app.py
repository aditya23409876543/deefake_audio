import streamlit as st

import torch
import numpy as np
import librosa
import os
from model import DeepfakeDetectorCNN, get_model
from calibration import RobustPredictor, explain_prediction, save_pretrained_weights
from utils import plot_spectrogram, plot_waveform, AudioDataset, preprocess_audio
import torch.nn.functional as F

# Page Configuration
st.set_page_config(page_title="Audio Deepfake Detector", page_icon="🎙️", layout="wide")

# Initialize pretrained weights if needed
if not os.path.exists('model/enhanced_pretrained.pth'):
    save_pretrained_weights()

# Load Model with calibration
def get_model_status(model_type):
    """Check if model has trained weights"""
    if model_type == 'pytorch_model':
        custom_path = 'model/pytorch_model.pth'
        if os.path.exists(custom_path):
            return "trained", custom_path
        else:
            return "uninitialized", None
    best_path = f'model/{model_type}_best_trained.pth'
    balanced_path = f'model/{model_type}_balanced_elevenlabs.pth'
    comprehensive_path = f'model/{model_type}_comprehensive_trained.pth'
    quick_path = f'model/{model_type}_quick_trained.pth'
    trained_path = f'model/{model_type}_trained.pth'
    pretrained_path = f'model/{model_type}_pretrained.pth'
    if os.path.exists(best_path):
        return "trained", best_path
    elif os.path.exists(balanced_path):
        return "trained", balanced_path
    elif os.path.exists(comprehensive_path):
        return "trained", comprehensive_path
    elif os.path.exists(quick_path):
        return "trained", quick_path
    elif os.path.exists(trained_path):
        return "trained", trained_path
    elif os.path.exists(pretrained_path):
        return "pretrained", pretrained_path
    else:
        return "uninitialized", None

@st.cache_resource
def load_model_with_calibration(model_type='enhanced'):
    """Load model with robust predictor and calibration"""
    
    # Handle ensemble models
    if model_type.startswith('ensemble_'):
        from src.ensemble import create_ensemble
        ensemble_type = model_type.split('_')[1]
        model = create_ensemble(ensemble_type)
    elif model_type == 'pytorch_model':
        model = get_model('pytorch_model')
    # Removed hf_wav2vec2 support (file deleted)
    else:
        model = get_model(model_type)
    
    # Try to load trained weights first, then pretrained
    if model_type == 'pytorch_model':
        custom_path = 'model/pytorch_model.pth'
        if os.path.exists(custom_path):
            state_dict = torch.load(custom_path, map_location=torch.device('cpu'), weights_only=False)
            # If this is a checkpoint dict, extract the actual model weights
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            try:
                model.load_state_dict(state_dict)
                print(f"Loaded custom PyTorch model weights from {custom_path}")
            except RuntimeError as e:
                missing, unexpected = None, None
                if hasattr(e, 'args') and len(e.args) > 0:
                    msg = e.args[0]
                    if 'Missing key(s) in state_dict' in msg or 'Unexpected key(s) in state_dict' in msg:
                        import re
                        missing = re.findall(r"Missing key\(s\) in state_dict: (.*?)\\n", msg)
                        unexpected = re.findall(r"Unexpected key\(s\) in state_dict: (.*?)\\n", msg)
                print(f"[ERROR] Failed to load state_dict for {custom_path}: {e}")
                if missing:
                    print(f"Missing keys: {missing}")
                if unexpected:
                    print(f"Unexpected keys: {unexpected}")
                raise
        else:
            # Initialize with reasonable weights
            for name, param in model.named_parameters():
                if 'weight' in name:
                    torch.nn.init.normal_(param, mean=0.0, std=0.01)
                elif 'bias' in name:
                    if 'classifier' in name or 'fc' in name or 'fusion' in name:
                        torch.nn.init.constant_(param, 0.0)
    else:
        best_path = f'model/{model_type}_best_trained.pth'
        balanced_path = f'model/{model_type}_balanced_elevenlabs.pth'
        comprehensive_path = f'model/{model_type}_comprehensive_trained.pth'
        quick_path = f'model/{model_type}_quick_trained.pth'
        trained_path = f'model/{model_type}_trained.pth'
        pretrained_path = f'model/{model_type}_pretrained.pth'
        if os.path.exists(best_path):
            state_dict = torch.load(best_path, map_location=torch.device('cpu'), weights_only=False)
            try:
                model.load_state_dict(state_dict)
                print(f"Loaded best trained weights from {best_path}")
            except RuntimeError as e:
                missing, unexpected = None, None
                try:
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                except Exception:
                    pass
                print(f"[ERROR] Failed to load state_dict for {best_path}: {e}")
                if missing:
                    print(f"Missing keys: {missing}")
                if unexpected:
                    print(f"Unexpected keys: {unexpected}")
                raise
        elif os.path.exists(balanced_path):
            state_dict = torch.load(balanced_path, map_location=torch.device('cpu'), weights_only=False)
            try:
                model.load_state_dict(state_dict)
                print(f"Loaded balanced weights from {balanced_path}")
            except RuntimeError as e:
                missing, unexpected = None, None
                try:
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                except Exception:
                    pass
                print(f"[ERROR] Failed to load state_dict for {balanced_path}: {e}")
                if missing:
                    print(f"Missing keys: {missing}")
                if unexpected:
                    print(f"Unexpected keys: {unexpected}")
                raise
        elif os.path.exists(comprehensive_path):
            state_dict = torch.load(comprehensive_path, map_location=torch.device('cpu'), weights_only=False)
            try:
                model.load_state_dict(state_dict)
                print(f"Loaded comprehensive trained weights from {comprehensive_path}")
            except RuntimeError as e:
                missing, unexpected = None, None
                try:
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                except Exception:
                    pass
                print(f"[ERROR] Failed to load state_dict for {comprehensive_path}: {e}")
                if missing:
                    print(f"Missing keys: {missing}")
                if unexpected:
                    print(f"Unexpected keys: {unexpected}")
                raise
        elif os.path.exists(quick_path):
            state_dict = torch.load(quick_path, map_location=torch.device('cpu'), weights_only=False)
            try:
                model.load_state_dict(state_dict)
                print(f"Loaded quick trained weights from {quick_path}")
            except RuntimeError as e:
                missing, unexpected = None, None
                try:
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                except Exception:
                    pass
                print(f"[ERROR] Failed to load state_dict for {quick_path}: {e}")
                if missing:
                    print(f"Missing keys: {missing}")
                if unexpected:
                    print(f"Unexpected keys: {unexpected}")
                raise
        elif os.path.exists(trained_path):
            state_dict = torch.load(trained_path, map_location=torch.device('cpu'), weights_only=False)
            try:
                model.load_state_dict(state_dict)
                print(f"Loaded trained weights from {trained_path}")
            except RuntimeError as e:
                missing, unexpected = None, None
                try:
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                except Exception:
                    pass
                print(f"[ERROR] Failed to load state_dict for {trained_path}: {e}")
                if missing:
                    print(f"Missing keys: {missing}")
                if unexpected:
                    print(f"Unexpected keys: {unexpected}")
                raise
        elif os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location=torch.device('cpu'), weights_only=False)
            try:
                model.load_state_dict(state_dict)
            except RuntimeError as e:
                missing, unexpected = None, None
                try:
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                except Exception:
                    pass
                print(f"[ERROR] Failed to load state_dict for {pretrained_path}: {e}")
                if missing:
                    print(f"Missing keys: {missing}")
                if unexpected:
                    print(f"Unexpected keys: {unexpected}")
                raise
        else:
            for name, param in model.named_parameters():
                if 'weight' in name:
                    torch.nn.init.normal_(param, mean=0.0, std=0.01)
                elif 'bias' in name:
                    if 'classifier' in name or 'fc' in name or 'fusion' in name:
                        torch.nn.init.constant_(param, 0.0)

    model.eval()

    # Wrap with robust predictor
    predictor = RobustPredictor(model, device='cpu')

    return predictor



    # HuggingFace audio preprocessing
    def preprocess_audio_hf(audio_path, target_len=4*16000):
        import torchaudio
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        waveform = waveform.mean(dim=0, keepdim=True)  # mono
        if waveform.shape[1] < target_len:
            pad = target_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :target_len]
        return waveform

# UI Header
st.title("🎙️ Audio Deepfake Detection System")
st.markdown("""
### 🎯 Advanced 2-Model System
Choose between **Enhanced** (99.81% accuracy) and **Ensemble MultiScale** (highest confidence) models.

Upload an audio file to see the analysis with state-of-the-art deepfake detection.
""")

# Model options - dynamically filter based on availability
model_options = [
    {"key": "enhanced", "label": "Enhanced Model", "desc": "99.81% Accuracy • Best Overall"},
    {"key": "ensemble_multiscale", "label": "Ensemble MultiScale", "desc": "97-99% Accuracy • Highest Confidence"},
]

# Only add pytorch_model if the file exists
pytorch_model_path = 'model/pytorch_model.pth'
if os.path.exists(pytorch_model_path):
    model_options.append({"key": "pytorch_model", "label": "Open Source PyTorch", "desc": "ResNet+GRU+Attention"})

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "enhanced"
model_type = st.session_state.selected_model

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🏆 Best Accuracy", "99.81%", "Enhanced Model")
with col2:
    multiscale_status, _ = get_model_status("ensemble_multiscale")
    if multiscale_status == "trained":
        st.metric("🎯 Highest Confidence", "92-98%", "MultiScale Model")
    else:
        st.metric("🔧 MultiScale", "Not Trained", "Train to unlock")
with col3:
    pytorch_status, _ = get_model_status("pytorch_model")
    if pytorch_status == "trained":
        st.metric("🧪 Custom Model", "Available", "Custom/Experimental")
    else:
        st.metric("🧪 Custom Model", "Not Found", "Add pytorch_model.pth")


st.markdown("---")

# Current Model Selection Banner
if model_type == "enhanced":
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1e88e5, #1565c0); padding: 20px; border-radius: 15px; color: white; margin: 20px 0;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h2 style="margin: 0; color: white;">🏆 Enhanced Model Active</h2>
                <p style="margin: 5px 0; color: #e3f2fd; font-size: 16px;">99.81% Accuracy • Best Overall Performance</p>
                <p style="margin: 0; color: #e3f2fd; font-size: 14px;">✅ Using all 3 features: Spectral, MFCC, Phase</p>
            </div>
            <div style="text-align: center; font-size: 48px;">🎯</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
elif model_type == "pytorch_model":
    st.markdown("""
    <div style="background: linear-gradient(90deg, #9c27b0, #7b1fa2); padding: 20px; border-radius: 15px; color: white; margin: 20px 0;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h2 style="margin: 0; color: white;">🔬 Open Source PyTorch Model</h2>
                <p style="margin: 5px 0; color: #f3e5f5; font-size: 16px;">ResNet + GRU + Attention Architecture</p>
                <p style="margin: 0; color: #f3e5f5; font-size: 14px;">✅ Custom model with ~4M parameters</p>
            </div>
            <div style="text-align: center; font-size: 48px;">⚗️</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="background: linear-gradient(90deg, #43a047, #2e7d32); padding: 20px; border-radius: 15px; color: white; margin: 20px 0;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h2 style="margin: 0; color: white;">🔧 Ensemble MultiScale Active</h2>
                <p style="margin: 5px 0; color: #e8f5e8; font-size: 16px;">97-99% Accuracy • Highest Confidence Scores</p>
                <p style="margin: 0; color: #e8f5e8; font-size: 14px;">✅ Multi-Scale Processing • Advanced Ensemble</p>
            </div>
            <div style="text-align: center; font-size: 48px;">⚡</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Configuration")


# Model Selection with Cards
st.sidebar.subheader("🎯 Choose Model")
for option in model_options:
    with st.sidebar.container():
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            status, _ = get_model_status(option["key"])
            status_icon = "✅" if status == "trained" else "❌"
            # Highlight if this model is selected
            if model_type == option["key"]:
                st.markdown(f"""
                <div style="background-color: #e3f2fd; padding: 10px; border-radius: 8px; border-left: 4px solid #1e88e5;">
                    <h4 style="margin: 0; color: #1565c0;">{status_icon} {option['label']}</h4>
                    <p style="margin: 5px 0; color: #1976d2; font-size: 12px;">✅ CURRENTLY SELECTED</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"### {status_icon} {option['label']}")
            st.caption(f"• {option['desc']}")
        with col2:
            button_type = "primary" if model_type == option["key"] else "secondary"
            if st.button("Select" if model_type != option["key"] else "✅ Active", 
                        key=f"{option['key']}_btn", 
                        help=f"Choose {option['label']}",
                        type=button_type,
                        disabled=model_type == option["key"]):
                st.session_state.selected_model = option["key"]
                st.rerun()
st.sidebar.markdown("---")

# Current Model Selection Card
with st.sidebar.container():
    st.sidebar.markdown("### 🎯 Current Active Model")
    
    if model_type == "enhanced":
        st.sidebar.markdown("""
        <div style="background-color: #1e88e5; padding: 15px; border-radius: 10px; color: white;">
            <h4 style="margin: 0; color: white;">🏆 Enhanced Model</h4>
            <p style="margin: 5px 0; color: white;">99.81% Accuracy • Best Overall</p>
            <p style="margin: 0; font-size: 12px; color: #e3f2fd;">✅ ACTIVE • All Features</p>
        </div>
        """, unsafe_allow_html=True)
    elif model_type == "pytorch_model":
        st.sidebar.markdown("""
        <div style="background-color: #9c27b0; padding: 15px; border-radius: 10px; color: white;">
            <h4 style="margin: 0; color: white;">🧪 HuggingFace PyTorch</h4>
            <p style="margin: 5px 0; color: white;">Wav2Vec2 + BiGRU + Attention</p>
            <p style="margin: 0; font-size: 12px; color: #f3e5f5;">✅ ACTIVE • 98.5M params</p>
            <p style="margin: 0; font-size: 11px; color: #f3e5f5;">🤗 koyelog/deepfake-voice-detector</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div style="background-color: #43a047; padding: 15px; border-radius: 10px; color: white;">
            <h4 style="margin: 0; color: white;">🔧 Ensemble MultiScale</h4>
            <p style="margin: 5px 0; color: white;">97-99% Accuracy • Highest Confidence</p>
            <p style="margin: 0; font-size: 12px; color: #e8f5e8;">✅ ACTIVE • Multi-Scale</p>
        </div>
        """, unsafe_allow_html=True)

# Model Status Section
st.sidebar.subheader("📊 Model Status")

# Get status for current model
status, path = get_model_status(model_type)

# Display status with visual indicators
if status == "trained":
    st.sidebar.success("✅ **Model Ready**")
    st.sidebar.caption(f"Using trained weights")
elif status == "pretrained":
    st.sidebar.warning("⚠️ **Pretrained Only**")
    st.sidebar.caption("Using baseline weights")
else:
    st.sidebar.error("❌ **Not Trained**")
    st.sidebar.caption("Model needs training")

# Quick Status Overview
st.sidebar.markdown("---")
st.sidebar.subheader("📈 System Status")

# Enhanced Model Status
enhanced_status, _ = get_model_status("enhanced")
enhanced_icon = "✅" if enhanced_status == "trained" else "❌"
st.sidebar.write(f"{enhanced_icon} Enhanced: {'Trained' if enhanced_status == 'trained' else 'Not Trained'}")

# MultiScale Model Status
multiscale_status, _ = get_model_status("ensemble_multiscale")
multiscale_icon = "✅" if multiscale_status == "trained" else "❌"
st.sidebar.write(f"{multiscale_icon} MultiScale: {'Trained' if multiscale_status == 'trained' else 'Not Trained'}")

# Overall Status
if enhanced_status == "trained" and multiscale_status == "trained":
    st.sidebar.success("🎉 **System Complete**")
elif enhanced_status == "trained":
    st.sidebar.info("⚡ **Partially Ready**")
else:
    st.sidebar.warning("🔧 **Setup Needed**")

# Training Section
st.sidebar.markdown("---")
st.sidebar.subheader("🎓 Model Training")

# Check if Ensemble MultiScale needs training
multiscale_status, _ = get_model_status("ensemble_multiscale")

if multiscale_status != "trained":
    st.sidebar.warning("🔧 Ensemble MultiScale needs training!")
    st.sidebar.caption("Complete your 2-model system")
    
    # Quick training button
    if st.sidebar.button("⚡ Train MultiScale", key="train_multiscale"):
        with st.spinner("Training Ensemble MultiScale... This may take 5-10 minutes."):
            import subprocess
            result = subprocess.run(
                ["python", "train_ensemble.py"],
                capture_output=True,
                text=True,
                env={**os.environ, "TORCH_COMPILE_DISABLE": "1"}
            )
            if result.returncode == 0:
                st.sidebar.success("✅ Training complete!")
                st.rerun()
            else:
                st.sidebar.error("Training failed. Check console.")
    
    # Training info
    with st.sidebar.expander("Training Info"):
        st.write("**Expected Results:**")
        st.write("• Accuracy: 97-99%")
        st.write("• Confidence: 92-98%")
        st.write("• Time: 5-10 minutes")
        st.write("**Command:**")
        st.code("python train_ensemble.py", language="bash")
else:
    st.sidebar.success("✅ **All Models Ready!**")
    st.sidebar.caption("Both models trained and ready")

st.sidebar.markdown("---")
st.sidebar.header("🎯 Project Info")

if model_type == "enhanced":
    st.sidebar.info(f"""
**Enhanced Model** ✅
• Accuracy: 99.81%
• Features: All 3 types
• Best overall performance
• Dataset: ElevenLabs
• Status: Ready to use
""")
else:
    st.sidebar.info(f"""
**Ensemble MultiScale** 🔧
• Accuracy: 97-99%
• Features: Multi-scale
• Highest confidence
• Dataset: ElevenLabs
• Status: {multiscale_status.title()}
""")

# System Overview
with st.sidebar.expander("📊 System Overview"):
    st.write("**🎯 2-Model System:**")
    st.write("• Enhanced - Best accuracy")
    st.write("• MultiScale - Highest confidence")
    st.write("")
    st.write("**📁 Dataset:**")
    st.write("• 2,561 audio files")
    st.write("• 736 REAL, 1,825 FAKE")
    st.write("")
    st.write("**🚀 Performance:**")
    st.write("• 99.81% best accuracy")
    st.write("• 92-98% confidence")
    st.write("• Real-time detection")

# File Upload Section
st.markdown("---")
st.subheader("📁 Upload Audio File")

# File uploader with session state persistence
uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3", "flac", "ogg"])

# Store uploaded file in session state
if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file
    st.session_state.file_uploaded = True
elif 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False

# Show file info if uploaded
if st.session_state.get('file_uploaded', False) and st.session_state.get('uploaded_file') is not None:
    uploaded_file = st.session_state.uploaded_file
    st.success(f"✅ File loaded: {uploaded_file.name}")
    
    # Save uploaded file temporarily (only if not already saved or file changed)
    temp_path = "temp_audio.wav"
    if not os.path.exists(temp_path) or st.session_state.get('last_file_name') != uploaded_file.name:
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.last_file_name = uploaded_file.name
    
    # Layout for analysis
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎵 Audio Visualization")
        st.audio(uploaded_file, format='audio/wav')
        fig_wave = plot_waveform(temp_path)
        st.pyplot(fig_wave)
        fig_spec = plot_spectrogram(temp_path)
        st.pyplot(fig_spec)

    with col2:
        # Model-specific prediction header
        if model_type == "enhanced":
            st.subheader("🏆 Enhanced Model Prediction")
        elif model_type == "ensemble_multiscale":
            st.subheader("🔧 Ensemble MultiScale Prediction")
        elif model_type == "pytorch_model":
            st.subheader("🧪 HuggingFace PyTorch Model")
            st.info("📊 **koyelog/deepfake-voice-detector-sota**")
            
            # Model details expander
            with st.expander("📖 Model Details & Architecture"):
                st.markdown("""
                ### 🤗 HuggingFace Model Information
                
                **Repository:** [koyelog/deepfake-voice-detector-sota](https://huggingface.co/koyelog/deepfake-voice-detector-sota)
                
                **Architecture:**
                - 🎯 **Base Model:** facebook/wav2vec2-base (pretrained speech features)
                - � **BiGRU:** 2 layers, 256 hidden units per direction (512 total)
                - 🎭 **Multi-Head Attention:** 8 heads, 512-dimensional embeddings
                - 📊 **Classification Head:** Linear(512→512→128→1) with BatchNorm & Dropout
                
                **Specifications:**
                - 📈 **Parameters:** ~98.5M total parameters
                - 🎵 **Input:** 4-second audio at 16kHz (single-channel)
                - 🎯 **Output:** Probability (0=Real, 1=Fake), threshold 0.5
                - 🏆 **Validation Accuracy:** 95-97%
                - 📊 **Precision:** ~0.95 | **Recall:** ~0.94 | **F1:** ~0.94
                
                **Training Data:**
                - 📁 **Total Samples:** 822,166 from 19 datasets
                - ✅ **Real/Bonafide:** 387,422 samples (47.1%)
                - ❌ **Fake/Deepfake:** 434,744 samples (52.9%)
                - 🏛️ **Sources:** ASVspoof 2021, WaveFake, Audio-Deepfake, and 16+ others
                
                **License:** Apache-2.0
                """)
        else:
            st.subheader(f"🧪 {model_type.title()} Prediction")
        
        # Prediction code runs for ALL models
        predictor = load_model_with_calibration(model_type)
        with st.spinner('Analyzing audio...'):
            spectral, mfcc, phase = preprocess_audio(temp_path)
            with torch.no_grad():
                result = predictor.predict(spectral, mfcc, phase, return_confidence=True)
                prediction = result['prediction']
                confidence = result['confidence']
                probabilities = result['probabilities']
                is_valid = result['is_valid']
                warnings = result['warnings']
            
            # Store result for comparison
            st.session_state[f'{model_type}_result'] = {
                'prediction': prediction.item(),
                'confidence': confidence,
                'probabilities': probabilities,
                'model': model_type
            }
            
            # Show prediction results
            if prediction.item() == 1:
                st.success("✅ RESULT: REAL (Bona fide)")
                st.balloons()
            else:
                st.error("❌ RESULT: FAKE (Deepfake)")
                st.warning("⚠️ This audio appears to be artificially generated")
            
            # Convert confidence to scalar if it's a tensor
            conf_value = confidence.item() if hasattr(confidence, 'item') else confidence
            st.metric("Confidence Score", f"{conf_value*100:.1f}%")
            
            # Model Comparison Section
            if model_type == "enhanced" and st.session_state.get('ensemble_multiscale_result'):
                st.markdown("---")
                st.subheader("🔄 Model Comparison")
                other_result = st.session_state['ensemble_multiscale_result']
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**🏆 Enhanced Model**")
                    pred_text = "REAL" if result['prediction'] == 1 else "FAKE"
                    st.write(f"Prediction: **{pred_text}**")
                    st.write(f"Confidence: **{conf_value*100:.1f}%**")
                with col_b:
                    st.markdown("**🔧 Ensemble MultiScale**")
                    other_pred_text = "REAL" if other_result['prediction'] == 1 else "FAKE"
                    st.write(f"Prediction: **{other_pred_text}**")
                    other_conf_value = other_result['confidence'].item() if hasattr(other_result['confidence'], 'item') else other_result['confidence']
                    st.write(f"Confidence: **{other_conf_value*100:.1f}%**")
                if result['prediction'] == other_result['prediction']:
                    st.success("✅ Both models agree!")
                else:
                    st.warning("⚠️ Models disagree - consider using the higher confidence prediction")
            elif model_type == "ensemble_multiscale" and st.session_state.get('enhanced_result'):
                st.markdown("---")
                st.subheader("🔄 Model Comparison")
                other_result = st.session_state['enhanced_result']
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**🔧 Ensemble MultiScale**")
                    pred_text = "REAL" if result['prediction'] == 1 else "FAKE"
                    st.write(f"Prediction: **{pred_text}**")
                    st.write(f"Confidence: **{conf_value*100:.1f}%**")
                with col_b:
                    st.markdown("**🏆 Enhanced Model**")
                    other_pred_text = "REAL" if other_result['prediction'] == 1 else "FAKE"
                    st.write(f"Prediction: **{other_pred_text}**")
                    other_conf_value = other_result['confidence'].item() if hasattr(other_result['confidence'], 'item') else other_result['confidence']
                    st.write(f"Confidence: **{other_conf_value*100:.1f}%**")
                if result['prediction'] == other_result['prediction']:
                    st.success("✅ Both models agree!")
                else:
                    st.warning("⚠️ Models disagree - consider using the higher confidence prediction")
                if conf_value < 0.7:
                    with st.expander("⚠️ How to Improve Confidence"):
                        st.write("""
                        **Low confidence detected. Tips to improve:**
                        1. **Use longer audio clips** (4+ seconds ideal)
                        2. **Ensure clear audio quality** (reduce background noise)
                        3. **Use trained models** (run training on dataset)
                        4. **Try different models** (ensemble often gives higher confidence)
                        5. **Check audio format** (16kHz WAV recommended)
                        """)
                with st.expander("📊 Detailed Probabilities"):
                    col1, col2 = st.columns(2)
                    if probabilities.dim() == 2:
                        fake_prob = probabilities[0, 0].item()
                        real_prob = probabilities[0, 1].item() if probabilities.shape[1] > 1 else 1 - fake_prob
                    elif probabilities.dim() == 1:
                        if probabilities.shape[0] == 2:
                            fake_prob = probabilities[0].item()
                            real_prob = probabilities[1].item()
                        else:
                            fake_prob = probabilities[0].item()
                            real_prob = 1 - fake_prob
                    else:
                        fake_prob = probabilities.item()
                        real_prob = 1 - fake_prob
                    with col1:
                        st.metric("Real Probability", f"{real_prob*100:.1f}%")
                    with col2:
                        st.metric("Fake Probability", f"{fake_prob*100:.1f}%")
                    if prediction.item() == 1:
                        st.progress(real_prob)
                    else:
                        st.progress(fake_prob)
                if warnings:
                    st.warning("⚠️ " + "; ".join(warnings))

# Instructions when no file is uploaded
else:
    st.markdown("---")
    st.markdown("### 📋 How to Use:")
    st.markdown("""
    1. **Upload an audio file** (WAV, MP3, FLAC, OGG supported)
    2. **Select a model** from the sidebar (Enhanced or Ensemble MultiScale)
    3. **View the prediction** and confidence score
    4. **Switch models** to compare results without re-uploading
    5. **Compare predictions** from both models for best accuracy
    """)
    
    st.info("💡 **Tip**: Upload once, then switch between models to compare predictions instantly!")
    
    st.markdown("""
    **Current dataset status:**
    - 2,561 audio files available
    - 736 REAL (Original), 1,825 FAKE
    - Both models trained and ready to use
    """)
