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
import requests

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
# CUSTOM CSS - ENHANCED WITH TABS
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
    
    /* Warning Box */
    .warning-box {
        background: rgba(251, 191, 36, 0.15);
        border: 2px solid rgba(251, 191, 36, 0.5);
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
        transition: all 0.3s ease;
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
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
        border-color: rgba(102, 126, 234, 0.6);
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 30, 50, 0.5);
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(50, 50, 70, 0.5);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 8px;
        color: rgba(255, 255, 255, 0.7);
        padding: 12px 24px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.2);
        border-color: rgba(102, 126, 234, 0.5);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-color: transparent;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.5rem;
    }
    
    /* Text Colors */
    p, span, label, div {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(20, 20, 40, 0.95);
    }
    
    /* Placeholder Content */
    .placeholder-content {
        background: rgba(50, 50, 70, 0.3);
        border: 2px dashed rgba(102, 126, 234, 0.4);
        border-radius: 15px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 2rem 0;
    }
    
    .placeholder-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }
    
    .placeholder-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 1rem;
    }
    
    .placeholder-description {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.6);
        line-height: 1.6;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(50, 50, 70, 0.5);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(102, 126, 234, 0.2);
        border-color: rgba(102, 126, 234, 0.5);
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================

if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = False
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None


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
    
    st.markdown("---")
    
    # Quick Stats
    if st.session_state.predictions_made:
        st.subheader("📊 Quick Stats")
        stats = st.session_state.prediction_results['stats']
        st.metric("Detections", f"{stats['n_detected']:,}")
        st.metric("Rate", f"{stats['detection_percentage']:.1f}%")


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
            st.session_state.audio_data = {
                'audio': audio,
                'metadata': metadata,
                'filepath': str(temp_path)
            }
        
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
            # Check if inference server is running
            server_url = "http://127.0.0.1:8502"
            use_server = False
            
            try:
                health_check = requests.get(f"{server_url}/health", timeout=2)
                if health_check.status_code == 200:
                    use_server = True
                    st.info("🌐 Using FastAPI inference server")
            except:
                st.warning("⚠️ Inference server not detected. Using local inference.")
            
            # Load model and parameters
            with st.spinner("Loading AI model..."):
                if not use_server:
                    model = load_model_file(selected_model)
                mean, std = load_standardization_params()
            
            # Preprocess
            with st.spinner("Preprocessing..."):
                sequences, info = processor.preprocess_for_prediction(str(temp_path), mean, std)
            
            # Predict
            with st.spinner("Running inference..."):
                if use_server:
                    # Use inference server
                    response = requests.post(
                        f"{server_url}/predict",
                        json={
                            "sequences": sequences.tolist(),
                            "model_type": selected_model,
                            "threshold": threshold
                        },
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        predictions = {
                            'probabilities': np.array(result['probabilities']),
                            'predictions': np.array(result['predictions']),
                            'threshold': result['threshold']
                        }
                    else:
                        st.error(f"Server error: {response.status_code}")
                        st.stop()
                else:
                    # Local inference
                    predictor = BowelSoundPredictor(model, threshold)
                    predictions = predictor.predict(sequences, threshold)
                
                # Calculate stats
                predictor = BowelSoundPredictor(None, threshold)
                stats = predictor.get_detection_stats(predictions)
            
            st.success("✓ Analysis complete!")
            
            # Store results
            st.session_state.predictions_made = True
            st.session_state.prediction_results = {
                'predictions': predictions,
                'stats': stats,
                'sequences': sequences,
                'info': info
            }
            
            # Force rerun to show tabs
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
    
    
    # =============================================================================
    # RESULTS - 4 TABS
    # =============================================================================
    
    if st.session_state.predictions_made:
        st.markdown('<h2 class="section-header">📈 Step 4: Analysis Results</h2>', unsafe_allow_html=True)
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Classification",
            "🔬 Signal Processing",
            "🤖 Agent",
            "📄 Report"
        ])
        
        results = st.session_state.prediction_results
        predictions = results['predictions']
        stats = results['stats']
        
        
        # =====================================================================
        # TAB 1: CLASSIFICATION
        # =====================================================================
        
        with tab1:
            st.markdown("### 🎯 Classification Results")
            st.markdown("Complete AI-powered bowel sound detection and analysis.")
            
            st.markdown("---")
            
            # Key metrics
            st.subheader("📊 Key Performance Indicators")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "⚡ Frames Analyzed",
                    f"{stats['n_total_frames']:,}",
                    help="Total number of audio frames processed"
                )
            
            with col2:
                st.metric(
                    "🔊 Bowel Sounds",
                    f"{stats['n_detected']:,}",
                    f"{stats['detection_percentage']:.1f}%"
                )
            
            with col3:
                st.metric(
                    "⏱️ Sounds/Minute",
                    f"{stats['bowel_sounds_per_minute']:.1f}",
                    help="Estimated bowel sounds per minute"
                )
            
            with col4:
                st.metric(
                    "🎚️ Confidence",
                    f"{stats['mean_probability']:.3f}",
                    help="Average prediction confidence"
                )
            
            st.markdown("---")
            
            # Detailed statistics
            with st.expander("📊 Detailed Statistical Analysis", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🔍 Detection Metrics**")
                    st.write(f"• Total Frames: **{stats['n_total_frames']:,}**")
                    st.write(f"• Bowel Sound Frames: **{stats['n_detected']:,}**")
                    st.write(f"• Background Noise: **{stats['n_noise']:,}**")
                    st.write(f"• Detection Rate: **{stats['detection_percentage']:.2f}%**")
                    st.write(f"• Sounds/Minute: **{stats['bowel_sounds_per_minute']:.1f}**")
                
                with col2:
                    st.markdown("**📊 Confidence Metrics**")
                    st.write(f"• Mean Probability: **{stats['mean_probability']:.4f}**")
                    st.write(f"• Std Deviation: **{stats['std_probability']:.4f}**")
                    st.write(f"• Maximum: **{stats['max_probability']:.4f}**")
                    st.write(f"• Minimum: **{stats['min_probability']:.4f}**")
                    st.write(f"• Mean Detected: **{stats['mean_detected_probability']:.4f}**")
            
            st.markdown("---")
            
            # Visualizations
            st.subheader("📉 Visual Analysis")
            
            results_viz = ResultsVisualizer()
            
            # Detection timeline
            st.markdown("**📍 Temporal Detection Timeline**")
            fig_timeline = results_viz.plot_prediction_timeline(
                predictions['probabilities'],
                predictions['predictions'],
                predictions['threshold']
            )
            st.pyplot(fig_timeline)
            
            # Probability histogram
            with st.expander("📊 Probability Distribution Analysis", expanded=False):
                fig_hist = results_viz.plot_probability_histogram(
                    predictions['probabilities'],
                    predictions['predictions'],
                    predictions['threshold']
                )
                st.pyplot(fig_hist)
            
            st.markdown("---")
            
            # Export options
            st.subheader("💾 Export Classification Data")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # CSV Export
                df = pd.DataFrame({
                    'Frame': range(len(predictions['predictions'])),
                    'Probability': predictions['probabilities'],
                    'Prediction': predictions['predictions'],
                    'Time_seconds': np.arange(len(predictions['predictions'])) * (AudioConfig.HOP_SAMPLES / AudioConfig.SR)
                })
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    f"classification_{uploaded_file.name.replace('.wav', '.csv')}",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # JSON Export
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
                st.download_button(
                    "📥 Download JSON",
                    stats_json,
                    f"stats_{uploaded_file.name.replace('.wav', '.json')}",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col3:
                # PNG Export
                buf = io.BytesIO()
                fig_timeline.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                st.download_button(
                    "📥 Download PNG",
                    buf,
                    f"timeline_{uploaded_file.name.replace('.wav', '.png')}",
                    mime="image/png",
                    use_container_width=True
                )
        
        
        # =====================================================================
        # TAB 2: SIGNAL PROCESSING
        # =====================================================================
        
        with tab2:
            st.markdown("""
            <div class="placeholder-content">
                <div class="placeholder-icon">🔬</div>
                <div class="placeholder-title">Signal Processing Module</div>
                <div class="placeholder-description">
                    This section will contain advanced signal processing features:<br/><br/>
                    • Frequency domain analysis<br/>
                    • Time-frequency representations<br/>
                    • Filter design and application<br/>
                    • Noise reduction algorithms<br/>
                    • Feature extraction visualizations<br/><br/>
                    <strong>Status:</strong> Under Development
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Placeholder for future implementation
            st.info("💡 **Coming Soon**: Advanced signal processing tools for detailed acoustic analysis.")
            
            # Example structure for future implementation
            with st.expander("🔧 Planned Features", expanded=False):
                st.markdown("""
                **Feature Roadmap:**
                1. **Spectral Analysis**
                   - Power spectral density
                   - Spectral centroid tracking
                   - Bandwidth analysis
                
                2. **Filtering**
                   - Adaptive filtering
                   - Wavelet denoising
                   - Band-pass filtering
                
                3. **Feature Extraction**
                   - MFCC visualization
                   - Zero-crossing rate
                   - Energy contours
                
                4. **Time-Frequency Analysis**
                   - Wavelet transform
                   - Wigner-Ville distribution
                   - Gabor analysis
                """)
        
        
        # =====================================================================
        # TAB 3: AGENT
        # =====================================================================
        
        with tab3:
            st.markdown("""
            <div class="placeholder-content">
                <div class="placeholder-icon">🤖</div>
                <div class="placeholder-title">AI Agent Module</div>
                <div class="placeholder-description">
                    This section will feature an intelligent agent for interactive analysis:<br/><br/>
                    • Natural language queries about results<br/>
                    • Automated insights generation<br/>
                    • Comparative analysis with historical data<br/>
                    • Clinical recommendations<br/>
                    • Pattern recognition and alerts<br/><br/>
                    <strong>Status:</strong> Under Development
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("💡 **Coming Soon**: Conversational AI agent for intelligent result interpretation.")
            
            # Placeholder for future implementation
            with st.expander("🔧 Planned Features", expanded=False):
                st.markdown("""
                **Agent Capabilities:**
                1. **Conversational Interface**
                   - Ask questions about your results
                   - Get explanations in plain language
                   - Request specific analyses
                
                2. **Automated Insights**
                   - Anomaly detection
                   - Trend analysis
                   - Clinical correlations
                
                3. **Recommendations**
                   - Suggested follow-up tests
                   - Quality improvement tips
                   - Recording best practices
                
                4. **Knowledge Base**
                   - Medical literature references
                   - Similar case studies
                   - Diagnostic criteria
                """)
        
        
        # =====================================================================
        # TAB 4: REPORT
        # =====================================================================
        
        with tab4:
            st.markdown("""
            <div class="placeholder-content">
                <div class="placeholder-icon">📄</div>
                <div class="placeholder-title">Report Generation Module</div>
                <div class="placeholder-description">
                    This section will provide comprehensive reporting capabilities:<br/><br/>
                    • Professional PDF reports<br/>
                    • Customizable templates<br/>
                    • Multi-patient comparisons<br/>
                    • Longitudinal tracking<br/>
                    • Clinical documentation<br/><br/>
                    <strong>Status:</strong> Under Development
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("💡 **Coming Soon**: Automated report generation for clinical documentation.")
            
            # Placeholder for future implementation
            with st.expander("🔧 Planned Features", expanded=False):
                st.markdown("""
                **Report Features:**
                1. **PDF Generation**
                   - Professional layout
                   - Embedded visualizations
                   - Customizable branding
                
                2. **Report Types**
                   - Single session report
                   - Multi-session comparison
                   - Longitudinal analysis
                   - Batch processing reports
                
                3. **Clinical Documentation**
                   - SOAP note format
                   - ICD-10 coding suggestions
                   - Evidence-based references
                
                4. **Export Options**
                   - PDF, DOCX, HTML
                   - EMR integration formats
                   - Research data exports
                """)


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; color: rgba(255,255,255,0.7);'>
    <h3>🔊 Bowel Sound Detection AI</h3>
    <p>Powered by Deep Learning & Neural Networks</p>
    <p style='font-size: 0.9rem;'>Based on: Ficek et al. (2021) - Sensors, 21(22), 7602</p>
    <p style='font-size: 0.85rem; margin-top: 1rem;'>
        Made with ❤️ using Streamlit, TensorFlow & Librosa
    </p>
</div>
""", unsafe_allow_html=True)