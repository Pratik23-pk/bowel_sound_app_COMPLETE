# Setup Guide - Bowel Sound Classification App

This guide will walk you through setting up and running the Bowel Sound Classification application.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Model Setup](#model-setup)
4. [Running the Application](#running-the-application)
5. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- **Operating System**: Windows, macOS, or Linux
- **Python**: Version 3.11 or higher
- **RAM**: Minimum 4 GB (8 GB recommended)
- **Disk Space**: At least 2 GB free space

### Required Software

#### Option 1: Using UV (Recommended)
UV is a fast Python package manager. Install it from: https://docs.astral.sh/uv/

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Option 2: Using pip
Make sure you have Python 3.11+ installed:
```bash
python --version  # Should show 3.11 or higher
```

---

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/bowel_sound_app_COMPLETE
```

### Step 2: Create Virtual Environment

#### Using UV:
```bash
uv venv
```

#### Using Python venv:
```bash
python -m venv .venv
```

### Step 3: Activate Virtual Environment

#### macOS/Linux:
```bash
source .venv/bin/activate
```

#### Windows (Command Prompt):
```cmd
.venv\Scripts\activate
```

#### Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

#### Using UV:
```bash
uv pip install -r requirements.txt
```

#### Using pip:
```bash
pip install -r requirements.txt
```

### Step 5: Verify Installation

```bash
python -c "import streamlit; import tensorflow; import librosa; print('All packages installed successfully!')"
```

---

## Model Setup

### Required Model Files

You need to add your trained model files to the `models/` directory:

1. **crnn_best.keras** - Your trained Keras model
2. **standardization_mean.npy** - Mean values from training
3. **standardization_std.npy** - Standard deviation values from training
4. **optimal_threshold.npy** (optional) - Custom classification threshold

### How to Add Model Files

1. Locate your trained model files
2. Copy them to the `models/` directory:

```bash
# Example (adjust paths as needed)
cp /path/to/your/crnn_best.keras models/
cp /path/to/your/standardization_mean.npy models/
cp /path/to/your/standardization_std.npy models/
cp /path/to/your/optimal_threshold.npy models/  # Optional
```

3. Verify files are in place:

```bash
ls models/
# Should show:
# README.md
# crnn_best.keras
# standardization_mean.npy
# standardization_std.npy
# optimal_threshold.npy (if you added it)
```

### Model File Formats

- **crnn_best.keras**: Keras model file (HDF5 or SavedModel format)
- **standardization_mean.npy**: NumPy array file
- **standardization_std.npy**: NumPy array file
- **optimal_threshold.npy**: NumPy scalar or single-element array

---

## Running the Application

### Start the Application

With your virtual environment activated:

```bash
streamlit run app.py
```

### Expected Output

You should see output similar to:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.x:8501
```

### Access the Application

Open your web browser and navigate to:
```
http://localhost:8501
```

### First Run

On the first run:
1. The application will load the model (this may take 10-30 seconds)
2. You should see "✅ Model loaded successfully!"
3. The main interface will appear

---

## Using the Application

### 1. Upload Audio

- Click on "Browse files" or drag and drop an audio file
- Supported formats: WAV, MP3, FLAC, OGG
- Maximum file size: 10 MB

### 2. Analyze

- Click the "🔍 Analyze Audio" button
- Wait for processing (usually 2-5 seconds)
- View prediction results

### 3. Explore Results

- **Upload & Predict**: See prediction, probability, and confidence
- **Visualizations**: View waveforms and spectrograms
- **Analysis**: Explore detailed statistics and history
- **Help**: Read usage instructions and technical details

---

## Troubleshooting

### Issue: "Model file not found"

**Solution:**
- Verify model files are in `models/` directory
- Check file names match exactly (case-sensitive)
- Ensure files have correct extensions (.keras, .npy)

### Issue: "ImportError: No module named..."

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Or with UV
uv pip install -r requirements.txt --force-reinstall
```

### Issue: TensorFlow Installation Problems

**For Apple Silicon Macs (M1/M2/M3):**
```bash
pip install tensorflow-macos tensorflow-metal
```

**For CUDA-enabled GPUs:**
```bash
pip install tensorflow[and-cuda]
```

**For CPU only:**
```bash
pip install tensorflow-cpu
```

### Issue: "Port 8501 is already in use"

**Solution:**
```bash
# Use a different port
streamlit run app.py --server.port 8502

# Or kill the process using port 8501
# macOS/Linux:
lsof -ti:8501 | xargs kill -9

# Windows:
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

### Issue: Audio file won't upload

**Solution:**
- Check file format (must be WAV, MP3, FLAC, or OGG)
- Verify file size is under 10 MB
- Ensure file is not corrupted (try opening in media player)

### Issue: Slow performance

**Solutions:**
- Close other applications to free up RAM
- Use smaller audio files
- Ensure you're using a GPU-enabled TensorFlow build (if available)
- Consider upgrading hardware

### Issue: Librosa installation fails

**Solution:**
```bash
# Install system dependencies first (macOS)
brew install libsndfile

# Install system dependencies (Ubuntu/Debian)
sudo apt-get install libsndfile1

# Then reinstall librosa
pip install librosa --upgrade
```

---

## Advanced Configuration

### Changing Model Parameters

Edit `config.py` to modify:

```python
# Audio processing
SAMPLE_RATE = 22050  # Target sample rate
AUDIO_DURATION = 5   # Fixed duration in seconds

# Mel spectrogram
N_MELS = 128        # Number of Mel bands
N_FFT = 2048        # FFT size
HOP_LENGTH = 512    # Hop length

# Classification
DEFAULT_THRESHOLD = 0.5  # Classification threshold
```

### Custom Port

Run on a specific port:
```bash
streamlit run app.py --server.port 8080
```

### Network Access

Allow access from other devices on your network:
```bash
streamlit run app.py --server.address 0.0.0.0
```

### Development Mode

Enable auto-reload on file changes:
```bash
streamlit run app.py --server.runOnSave true
```

---

## Updating Dependencies

### Update all packages:
```bash
pip install -r requirements.txt --upgrade
```

### Update specific package:
```bash
pip install streamlit --upgrade
```

---

## Deactivating Virtual Environment

When you're done:
```bash
deactivate
```

---

## Uninstallation

To remove the application:

1. Deactivate virtual environment (if active)
2. Delete the project directory:
```bash
rm -rf /path/to/bowel_sound_app_COMPLETE
```

---

## Getting Help

### Check Logs

Streamlit logs are displayed in the terminal where you ran the app.

### Debug Mode

Run with additional debugging:
```bash
streamlit run app.py --logger.level=debug
```

### Common Resources

- Streamlit Documentation: https://docs.streamlit.io
- TensorFlow Documentation: https://www.tensorflow.org/api_docs
- Librosa Documentation: https://librosa.org/doc/latest/

---

## Next Steps

Once the application is running successfully:

1. Test with sample audio files
2. Verify predictions are reasonable
3. Explore different visualizations
4. Check the Help tab for usage instructions

---

## Support

For additional help:
- Review the main README.md
- Check the Help tab in the application
- Verify all model files are correct
- Ensure audio files meet requirements

---

**Happy Analyzing! 🔊**
