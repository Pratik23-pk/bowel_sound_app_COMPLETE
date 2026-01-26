# 🔊 Bowel Sound Detection AI

A Streamlit-based application for bowel sound detection using deep learning, designed as a **portable, cross-platform** tool that runs on macOS, Windows, and Linux.

The app now uses a **separate FastAPI inference server** for TensorFlow model execution, which makes it more stable and avoids segmentation faults in the UI process.

---

## ✨ Features

- Upload abdominal auscultation audio (WAV) and analyze bowel sounds.
- Deep learning CRNN/CDNN models for bowel sound vs. noise classification (based on Ficek et al., 2021).
- Waveform and spectrogram visualization.
- Detection statistics: frames, detections, bowel sounds per minute, confidence metrics.
- Export predictions as CSV, JSON, and PNG.
- Robust architecture: Streamlit UI + FastAPI inference microservice.

---

## 🧱 Project Structure (key files)

- `app.py` – Streamlit frontend (UI, upload, visualization, calling inference server).
- `inference_server.py` – FastAPI + Uvicorn inference server that loads the TensorFlow model and runs `model.predict`.
- `modules/`  
  - `audio_processing.py` – Audio loading, STFT, spectrograms, sequences.  
  - `prediction.py` – Prediction logic; sends sequences to the inference server.  
  - `model_builder.py` – Model loading, standardization params, thresholds, caching.  
  - `visualization.py` – Waveform, spectrogram, and prediction visualizations.  
  - `metrics.py` – Metrics and detection statistics.
- `models/` – Place your trained model + preprocessing artifacts here (not tracked in git).  
- `config.py` – Configuration for audio, model, visualization, thresholds, etc.
- `requirements.txt` – Python dependencies.

---

## ✅ Prerequisites

- Python **3.11** (tested).
- OS: macOS, Windows, or Linux.
- Recommended: 8 GB RAM.

---

## 📦 Installation

### 1. Clone the repository

```bash
cd /path/to/your/projects
git clone https://github.com/Pratik23-pk/bowel_sound_app_COMPLETE.git bowel_sound_app
cd bowel_sound_app
2. Create and activate virtual environment (using uv)
bash
uv venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows (CMD)
# .venv\Scripts\Activate.ps1  # Windows (PowerShell)
3. Install dependencies
bash
uv pip install -r requirements.txt
The key dependencies include:

streamlit – UI frontend.

tensorflow – Deep learning backend.

librosa, soundfile, scipy, numpy – Audio processing.

fastapi, uvicorn – Inference server.

scikit-learn, matplotlib, seaborn, pandas – Metrics and plots.

🧠 Model Setup
Place your trained model and preprocessing artifacts into models/:

Required files:

crnn_best.keras – Trained Keras model (SavedModel / .keras / .h5 compatible with your TF version).

standardization_mean.npy – Feature-wise mean used during training.

standardization_std.npy – Feature-wise std used during training.

optimal_threshold.npy – (Optional) optimal decision threshold.

The app expects these paths via ModelPaths in config.py.

🚀 Running the Application
The app now runs as two processes:

Inference server (FastAPI + TensorFlow).

Streamlit UI.

1. Start the inference server
In terminal 1:

bash
cd /path/to/bowel_sound_app
source .venv/bin/activate
python inference_server.py
This starts a FastAPI server on:

http://127.0.0.1:8502

The server loads the model once on startup and exposes a /predict endpoint that accepts preprocessed sequences and returns probabilities + binary predictions.

2. Start the Streamlit app
In terminal 2:

bash
cd /path/to/bowel_sound_app
source .venv/bin/activate
streamlit run app.py
You should see:

text
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Open http://localhost:8501 in your browser.

3. Use the app
Select the model (CRNN/CDNN) in the sidebar.

Upload a WAV file (≤ 10 MB).

Click “🚀 Run AI Analysis”.

View:

Waveform and spectrogram.

Detection statistics and timeline.

Export CSV / JSON / PNG.

Behind the scenes:

app.py calls AudioProcessor to compute spectrograms and frame sequences.

Sequences are sent via HTTP to inference_server.py (FastAPI).

The server runs model.predict and returns probabilities and predictions.

The UI displays results and visualizations.

🛠 Why a Separate Inference Server?
On some macOS/Apple Silicon setups, running TensorFlow inside the Streamlit process can cause segmentation faults due to native library interactions. [web:46][web:49][web:50][web:63]

To make the app robust and portable:

TensorFlow and the model live only in inference_server.py.

Streamlit only talks to the server via HTTP.

If TensorFlow ever crashes, it only affects the inference server process, not the UI.

Restarting the server is enough; the Streamlit UI stays responsive.

This architecture also makes it easier to:

Scale inference separately.

Deploy the backend and frontend on different machines/containers in the future.

🧪 Testing
You can test the pieces separately:

TensorFlow + model import:

bash
source .venv/bin/activate
python -c "import tensorflow as tf; print(tf.__version__)"
Inference server:

Visit:

http://127.0.0.1:8502/docs

to see the interactive FastAPI docs and test the /predict endpoint manually.

🔄 Updating Your Fork / GitHub Repo
After you modify code (e.g., app.py, inference_server.py, modules/):

bash
git status
git add app.py inference_server.py modules/*.py requirements.txt
git commit -m "Refactor: add FastAPI inference server and stabilize TF inference"
git push origin main  # or your branch name
📚 References
Ficek et al., 2021 – Deep learning for bowel sound classification.

Streamlit documentation: https://docs.streamlit.io

FastAPI documentation: https://fastapi.tiangolo.com

TensorFlow documentation: https://www.tensorflow.org/api_docs

