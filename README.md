# 🔊 Bowel Sound Detection AI

Advanced AI-powered system for detecting and analyzing bowel sounds from audio recordings using deep learning.

## 🌟 Features

- 🤖 AI-powered detection using CRNN/CDNN neural networks
- 📊 Real-time audio visualization (waveform & spectrogram)
- 🎯 Automated bowel sound detection
- 📈 Comprehensive performance metrics
- 💾 Export results (CSV, JSON, PNG)
- 🎨 Beautiful dark-themed UI
- ⚡ GPU-accelerated inference (Apple Silicon support)

## 🚀 Quick Start

### Prerequisites

- Python 3.10
- UV package manager (recommended) or pip
- 16GB RAM (recommended)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/bowel-sound-detection.git
cd bowel-sound-detection
```

2. **Create virtual environment:**
```bash
uv venv
source .venv/bin/activate  # Mac/Linux
# or
.venv\Scripts\activate     # Windows
```

3. **Install dependencies:**
```bash
uv pip install -r requirements.txt
```

4. **Add model files:**

Place these files in the `models/` directory:
- `crnn_best.keras` (trained model)
- `standardization_mean.npy` (preprocessing parameters)
- `standardization_std.npy` (preprocessing parameters)

5. **Run the application:**
```bash
streamlit run app.py
```

## 📁 Project Structure
```
bowel_sound_app_COMPLETE/
├── app.py                    # Main Streamlit application
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── pyproject.toml           # UV project configuration
├── modules/
│   ├── __init__.py          # Module initialization
│   ├── audio_processor.py   # Audio preprocessing
│   ├── model_builder.py     # Model loading
│   ├── predictor.py         # Prediction engine
│   ├── visualizer.py        # Visualization
│   └── evaluator.py         # Metrics calculation
├── models/                   # Model files (add your trained models here)
└── uploads/                  # Temporary upload directory
```

## 🎯 Usage

1. **Upload Audio File**: Drag and drop or browse for a WAV file (2 seconds recommended)
2. **View Visualizations**: Automatic waveform and spectrogram generation
3. **Run AI Analysis**: Click "Run AI Analysis" button
4. **Review Results**: View detection metrics, timeline, and statistics
5. **Export Data**: Download results as CSV, JSON, or PNG

## 🧠 Model Information

Based on research by Ficek et al. (2021):
- **Paper**: "Analysis of Gastrointestinal Acoustic Activity Using Deep Neural Networks"
- **Journal**: Sensors, 21(22), 7602
- **Models**: CRNN (77.3% sensitivity) and CDNN (71.1% sensitivity)

## 🔧 Configuration

Edit `config.py` to customize:
- Audio processing parameters (sample rate, frame size, etc.)
- Model settings (batch size, threshold)
- Visualization preferences
- UI configuration

## 📊 Technical Details

- **Audio Processing**: 44.1 kHz sampling, 0-1500 Hz frequency filtering
- **Feature Extraction**: STFT with 10ms frames, 2.5ms hop length
- **Model Architecture**: CNN + Bidirectional GRU
- **Input Format**: 9-frame sequences with temporal context
- **Output**: Binary classification (bowel sound vs. noise)

## 🛠️ Development

### Setup Development Environment
```bash
# Install development dependencies
uv pip install -r requirements.txt

# Run tests
python config.py

# Check module imports
python -c "from modules import AudioProcessor; print('OK')"
```

## 📝 License

This project is for research and educational purposes.

## 🙏 Acknowledgments

- Based on research by Ficek et al. (2021)
- Built with Streamlit, TensorFlow, and Librosa
- UI design inspired by modern dark themes

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Made with ❤️ using Python, TensorFlow & Streamlit**