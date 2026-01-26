"""
Configuration file for Bowel Sound Detection Application
Contains all constants and parameters used throughout the app
"""

import os
from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

# Base directory
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"
TEMP_DIR = BASE_DIR / "temp"

# Create directories if they don't exist
UPLOADS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# =============================================================================
# AUDIO PROCESSING PARAMETERS (from paper)
# =============================================================================

class AudioConfig:
    """Audio preprocessing configuration"""
    SR = 44100                      # Sampling rate (Hz)
    DURATION = 2.0                  # Audio clip duration (seconds)
    FRAME_SAMPLES = 441             # Frame size (10ms at 44.1kHz)
    HOP_SAMPLES = 110               # Hop length (25% of frame)
    MAX_FREQ = 1500.0               # Maximum frequency (Hz)
    MIN_FREQ = 0.0                  # Minimum frequency (Hz)
    WINDOW_TYPE = 'hann'            # Window function for STFT
    
    # Derived parameters
    FRAME_DURATION_MS = (FRAME_SAMPLES / SR) * 1000  # 10ms
    HOP_DURATION_MS = (HOP_SAMPLES / SR) * 1000      # 2.5ms
    TARGET_SAMPLES = int(DURATION * SR)               # 88200 samples


class ModelConfig:
    """Model architecture configuration"""
    SEQ_LEN = 9                     # Number of frames in sequence
    HALF_SEQ = SEQ_LEN // 2         # 4 frames on each side
    
    # Model file names
    CRNN_MODEL_NAME = "crnn_best.keras"
    CDNN_MODEL_NAME = "cdnn_best.keras"
    
    # Fallback to .h5 if .keras not found
    CRNN_MODEL_H5 = "crnn_best.h5"
    CDNN_MODEL_H5 = "cdnn_best.h5"
    
    # Standardization files
    MEAN_FILE = "standardization_mean.npy"
    STD_FILE = "standardization_std.npy"
    THRESHOLD_FILE = "optimal_threshold.npy"
    
    # Default threshold (if file not found)
    DEFAULT_THRESHOLD = 0.3


class PredictionConfig:
    """Prediction configuration"""
    BATCH_SIZE = 1024
    VERBOSE = 0                     # Silent prediction
    
    # Threshold range for UI slider
    THRESHOLD_MIN = 0.1
    THRESHOLD_MAX = 0.9
    THRESHOLD_STEP = 0.05
    THRESHOLD_DEFAULT = 0.3


# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================

class VisualizationConfig:
    """Plotting and visualization settings"""
    
    # Figure sizes
    WAVEFORM_FIGSIZE = (12, 4)
    SPECTROGRAM_FIGSIZE = (12, 6)
    PREDICTION_FIGSIZE = (15, 5)
    METRICS_FIGSIZE = (10, 6)
    
    # DPI settings
    DPI = 100
    SAVE_DPI = 150
    
    # Colors
    PRIMARY_COLOR = '#1f77b4'
    SECONDARY_COLOR = '#ff7f0e'
    DETECTION_COLOR = '#2ca02c'
    THRESHOLD_COLOR = '#d62728'
    
    # Plot style
    PLOT_STYLE = 'seaborn-v0_8-darkgrid'
    GRID_ALPHA = 0.3
    LINE_WIDTH = 2


# =============================================================================
# UI SETTINGS
# =============================================================================

class UIConfig:
    """Streamlit UI configuration"""
    
    PAGE_TITLE = "Bowel Sound Detection"
    PAGE_ICON = "🔊"
    LAYOUT = "wide"
    INITIAL_SIDEBAR_STATE = "expanded"
    
    # File upload
    MAX_UPLOAD_SIZE_MB = 10
    ALLOWED_EXTENSIONS = ['.wav']
    
    # Sections
    SHOW_AUDIO_INFO = True
    SHOW_WAVEFORM = True
    SHOW_SPECTROGRAM = True
    SHOW_PREPROCESSING = True
    SHOW_PREDICTIONS = True
    SHOW_METRICS = True


# =============================================================================
# EVALUATION METRICS
# =============================================================================

class MetricsConfig:
    """Evaluation metrics configuration"""
    
    # Metrics to display
    DISPLAY_METRICS = [
        'Total Frames',
        'Bowel Sound Frames',
        'Noise Frames',
        'Detection Percentage',
        'Bowel Sounds per Minute',
        'Mean Confidence',
        'Max Confidence',
        'Min Confidence'
    ]
    
    # Decimal places for display
    PERCENTAGE_DECIMALS = 2
    CONFIDENCE_DECIMALS = 4


# =============================================================================
# MODEL PATHS (Constructed)
# =============================================================================

class ModelPaths:
    """Dynamically constructed model paths"""
    
    @staticmethod
    def get_model_path(model_type='crnn'):
        """Get model file path, checking .keras first, then .h5"""
        if model_type.lower() == 'crnn':
            keras_path = MODELS_DIR / ModelConfig.CRNN_MODEL_NAME
            h5_path = MODELS_DIR / ModelConfig.CRNN_MODEL_H5
        else:
            keras_path = MODELS_DIR / ModelConfig.CDNN_MODEL_NAME
            h5_path = MODELS_DIR / ModelConfig.CDNN_MODEL_H5
        
        if keras_path.exists():
            return keras_path
        elif h5_path.exists():
            return h5_path
        else:
            return None
    
    @staticmethod
    def get_standardization_paths():
        """Get paths for standardization files"""
        return {
            'mean': MODELS_DIR / ModelConfig.MEAN_FILE,
            'std': MODELS_DIR / ModelConfig.STD_FILE,
            'threshold': MODELS_DIR / ModelConfig.THRESHOLD_FILE
        }


# =============================================================================
# LOGGING
# =============================================================================

class LogConfig:
    """Logging configuration"""
    LEVEL = "INFO"
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_config_summary():
    """Get a summary of all configuration settings"""
    return f"""
    **Audio Configuration:**
    - Sample Rate: {AudioConfig.SR} Hz
    - Duration: {AudioConfig.DURATION} s
    - Frame Size: {AudioConfig.FRAME_SAMPLES} samples ({AudioConfig.FRAME_DURATION_MS:.1f} ms)
    - Hop Length: {AudioConfig.HOP_SAMPLES} samples ({AudioConfig.HOP_DURATION_MS:.1f} ms)
    - Frequency Range: {AudioConfig.MIN_FREQ}-{AudioConfig.MAX_FREQ} Hz
    
    **Model Configuration:**
    - Sequence Length: {ModelConfig.SEQ_LEN} frames
    - Default Threshold: {ModelConfig.DEFAULT_THRESHOLD}
    
    **Prediction Configuration:**
    - Batch Size: {PredictionConfig.BATCH_SIZE}
    """


def validate_environment():
    """Validate that all required directories and base files exist"""
    issues = []
    
    # Check directories
    if not MODELS_DIR.exists():
        issues.append(f"Models directory not found: {MODELS_DIR}")
    
    # Check for at least one model file
    crnn_path = ModelPaths.get_model_path('crnn')
    cdnn_path = ModelPaths.get_model_path('cdnn')
    
    if not crnn_path and not cdnn_path:
        issues.append("No model files found. Please add crnn_best.keras or cdnn_best.keras to models/")
    
    # Check standardization files
    std_paths = ModelPaths.get_standardization_paths()
    if not std_paths['mean'].exists():
        issues.append(f"Standardization mean file not found: {std_paths['mean']}")
    if not std_paths['std'].exists():
        issues.append(f"Standardization std file not found: {std_paths['std']}")
    
    return issues


if __name__ == "__main__":
    # Test configuration
    print("Configuration loaded successfully!")
    print(get_config_summary())
    
    # Validate environment
    issues = validate_environment()
    if issues:
        print("\n⚠️ Environment Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ Environment validated successfully!")