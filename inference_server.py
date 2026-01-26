"""
Inference Server for Bowel Sound Detection

Runs as a separate process to keep TensorFlow isolated from Streamlit.
"""

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from pathlib import Path

from modules.model_builder import get_model_manager

app = FastAPI(title="Bowel Sound Inference Server")

class InferenceRequest(BaseModel):
    model_type: str = "crnn"
    sequences: list  # nested list representation of np.ndarray
    threshold: float


class InferenceResponse(BaseModel):
    probabilities: list
    predictions: list
    threshold: float


# Load model + params once, on server startup
@app.on_event("startup")
def load_model_on_startup():
    manager = get_model_manager()
    # Default to CRNN; CDNN can also be requested dynamically
    components = manager.load_all_components("crnn")
    model = components["model"]
    # Build the model once with a dummy input to ensure graph is ready
    dummy = np.zeros((1, 9, 16, 1), dtype=np.float32)
    _ = model.predict(dummy)
    print("✓ Inference model initialized and warmed up")


@app.post("/predict", response_model=InferenceResponse)
def predict(req: InferenceRequest):
    """
    Receive preprocessed sequences, run model.predict, and return probabilities + binary predictions.
    """
    manager = get_model_manager()
    components = manager.load_all_components(req.model_type)
    model = components["model"]

    # Convert list to numpy array
    sequences = np.array(req.sequences, dtype=np.float32)

    # Run inference
    probs = model.predict(sequences, batch_size=64, verbose=0).ravel()
    preds = (probs >= req.threshold).astype(int)

    return InferenceResponse(
        probabilities=probs.tolist(),
        predictions=preds.tolist(),
        threshold=req.threshold,
    )


if __name__ == "__main__":
    # Run on localhost:8502 by default
    uvicorn.run(app, host="127.0.0.1", port=8502)
