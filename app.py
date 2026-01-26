"""
Bowel Sound Detection - Streamlit Application
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import io
import json

# Add modules to path
sys.path.append(str(Path(__file__).parent))

from config import UIConfig, AudioConfig, ModelConfig, validate_environment
from modules import (
    AudioProcessor,
    load_model_file,
    get_available_models,
    BowelSoundPredictor,
    AudioVisualizer,
    ResultsVisualizer,
    MetricsCalculator
)
from modules.model_builder import load_standardization_params, load_optimal_threshold


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Bowel Sound Detection",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# CUSTOM CSS - WORKING VERSION
# =============================================================================

st.markdown("""
<style>
    /* Import Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;900&family=Orbitron:wght@900&display=swap');
    
    /* Dark Background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: #ffffff;
    }
    
    /* Main Title */
    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(45deg, #ff0080, #ff8c00, #40e0d0, #7b68ee, #ff1493);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: rainbow 6s ease infinite;
        margin: 2rem 0;
    }
    
    @keyframes rainbow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Section Headers */
    .section-header {
        color: #667eea;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    /* Info Box */
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #ffffff;
    }
    
    /* Success Box */
    .success-box {
        background: rgba(16, 185, 129, 0.15);
        border: 2px solid rgba(16, 185, 129, 0.5);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #ffffff;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 25px;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Metrics */
    .stMetric {
        background: rgba(30, 30, 50, 0.6);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Text Colors */
    p, span, label, div {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(20, 20, 40, 0.95);
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================

if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = False


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown("---")
    
    # Model selection
    st.subheader("🤖 AI Model")
    
    available_models = get_available_models()
    model_options = []
    
    if available_models['crnn']:
        model_options.append('CRNN (Recommended)')
    if available_models['cdnn']:
        model_options.append('CDNN (Faster)')
    
    if not model_options:
        st.error("⚠️ No models found!")
        st.stop()
    
    selected_model_display = st.selectbox("Select Model", model_options)
    selected_model = 'crnn' if 'CRNN' in selected_model_display else 'cdnn'
    
    st.markdown("---")
    
    # Threshold info
    try:
        threshold = load_optimal_threshold()
        st.info(f"🎯 Auto Threshold: **{threshold:.3f}**")
    except:
        threshold = 0.3
        st.info(f"🎯 Default Threshold: **{threshold:.3f}**")
    
    st.markdown("---")
    
    st.subheader("ℹ️ About")
    st.write("AI-powered bowel sound detection using deep learning.")
    st.caption(f"Model: {selected_model.upper()}")


# =============================================================================
# MAIN CONTENT
# =============================================================================

# Title
st.markdown('<h1 class="main-title">🔊 BOWEL SOUND DETECTION AI</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <strong>🚀 Welcome!</strong> Upload a WAV audio file to analyze bowel sounds using AI.
</div>
""", unsafe_allow_html=True)

# Environment check
issues = validate_environment()
if issues:
    st.error("⚠️ **System Issues:**")
    for issue in issues:
        st.error(f"• {issue}")
    st.stop()


# =============================================================================
# FILE UPLOAD
# =============================================================================

st.markdown('<h2 class="section-header">📁 Step 1: Upload Audio</h2>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a WAV file",
    type=['wav'],
    help="Upload a WAV audio file (max 10MB)"
)

if uploaded_file is not None:
    # Save file
    temp_path = Path("uploads") / uploaded_file.name
    temp_path.parent.mkdir(exist_ok=True)
    
    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    st.markdown(f"""
    <div class="success-box">
        ✓ <strong>File uploaded:</strong> {uploaded_file.name}
    </div>
    """, unsafe_allow_html=True)
    
    # File info
    col1, col2, col3 = st.columns(3)
    col1.metric("📄 Filename", uploaded_file.name[:20])
    col2.metric("💾 Size", f"{uploaded_file.size / 1024:.2f} KB")
    col3.metric("🎵 Format", "WAV")
    
    
    # =============================================================================
    # AUDIO PROCESSING
    # =============================================================================
    
    st.markdown('<h2 class="section-header">📊 Step 2: Audio Analysis</h2>', unsafe_allow_html=True)
    
    processor = AudioProcessor()
    
    try:
        # Load audio
        with st.spinner("Loading audio..."):
            audio, metadata = processor.load_audio(str(temp_path))
        
        # Audio player
        st.subheader("🎧 Playback")
        st.audio(str(temp_path), format='audio/wav')
        
        # Visualizations
        visualizer = AudioVisualizer()
        audio_normalized = processor.normalize_length(audio)
        spectrogram, freqs = processor.compute_spectrogram(audio_normalized)
        
        # Waveform
        st.subheader("📈 Waveform")
        fig_waveform = visualizer.plot_waveform(audio_normalized)
        st.pyplot(fig_waveform)
        
        # Spectrogram
        st.subheader("🌈 Spectrogram")
        fig_spec = visualizer.plot_spectrogram(spectrogram)
        st.pyplot(fig_spec)
        
        st.success("✓ Audio processed successfully!")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.stop()
    
    
    # =============================================================================
    # PREDICTION
    # =============================================================================
    
    st.markdown('<h2 class="section-header">🤖 Step 3: AI Prediction</h2>', unsafe_allow_html=True)
    
    if st.button("🚀 Run AI Analysis", type="primary", use_container_width=True):
        try:
            # Load model
            with st.spinner("Loading AI model..."):
                model = load_model_file(selected_model)
                mean, std = load_standardization_params()
            
            # Preprocess
            with st.spinner("Preprocessing..."):
                sequences, info = processor.preprocess_for_prediction(str(temp_path), mean, std)
            
            # Predict
            with st.spinner("Running inference..."):
                predictor = BowelSoundPredictor(model, threshold)
                predictions = predictor.predict(sequences, threshold)
                stats = predictor.get_detection_stats(predictions)
            
            st.success("✓ Analysis complete!")
            
            # Store results
            st.session_state.predictions_made = True
            st.session_state.prediction_results = {
                'predictions': predictions,
                'stats': stats
            }
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
    
    
    # =============================================================================
    # RESULTS
    # =============================================================================
    
    if st.session_state.predictions_made:
        st.markdown('<h2 class="section-header">📈 Step 4: Results</h2>', unsafe_allow_html=True)
        
        results = st.session_state.prediction_results
        predictions = results['predictions']
        stats = results['stats']
        
        # Key metrics
        st.subheader("🎯 Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("⚡ Frames", f"{stats['n_total_frames']:,}")
        col2.metric("🔊 Detections", f"{stats['n_detected']:,}", f"{stats['detection_percentage']:.1f}%")
        col3.metric("⏱️ /Minute", f"{stats['bowel_sounds_per_minute']:.1f}")
        col4.metric("🎚️ Confidence", f"{stats['mean_probability']:.3f}")
        
        st.markdown("---")
        
        # Detailed stats
        with st.expander("📊 Detailed Statistics"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Detection Metrics:**")
                st.write(f"• Total: {stats['n_total_frames']:,}")
                st.write(f"• Detected: {stats['n_detected']:,}")
                st.write(f"• Noise: {stats['n_noise']:,}")
            
            with col2:
                st.write("**Confidence Metrics:**")
                st.write(f"• Mean: {stats['mean_probability']:.4f}")
                st.write(f"• Max: {stats['max_probability']:.4f}")
                st.write(f"• Min: {stats['min_probability']:.4f}")
        
        # Timeline
        st.subheader("📉 Detection Timeline")
        results_viz = ResultsVisualizer()
        fig_timeline = results_viz.plot_prediction_timeline(
            predictions['probabilities'],
            predictions['predictions'],
            predictions['threshold']
        )
        st.pyplot(fig_timeline)
        
        # Export
        st.subheader("💾 Export")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            df = pd.DataFrame({
                'Frame': range(len(predictions['predictions'])),
                'Probability': predictions['probabilities'],
                'Prediction': predictions['predictions']
            })
            csv = df.to_csv(index=False)
            st.download_button("📥 CSV", csv, f"predictions.csv")
        
        with col2:
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.integer, np.int64)):
                        return int(obj)
                    if isinstance(obj, (np.floating, np.float64)):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super().default(obj)
            
            stats_json = json.dumps(stats, indent=2, cls=NumpyEncoder)
            st.download_button("📥 JSON", stats_json, f"stats.json")
        
        with col3:
            buf = io.BytesIO()
            fig_timeline.savefig(buf, format='png', dpi=150)
            buf.seek(0)
            st.download_button("📥 PNG", buf, f"timeline.png")


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; color: rgba(255,255,255,0.7);'>
    <h3>🔊 Bowel Sound Detection AI</h3>
    <p>Powered by Deep Learning & Neural Networks</p>
    <p style='font-size: 0.9rem;'>Based on: Ficek et al. (2021)</p>
</div>
""", unsafe_allow_html=True)