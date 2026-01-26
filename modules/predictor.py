"""
Prediction Module
Handles bowel sound prediction from preprocessed audio sequences via external inference server
"""

import numpy as np
import requests
from typing import Dict, Tuple, List

import tensorflow as tf  # kept only for type hints; not used for heavy ops

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import PredictionConfig


class BowelSoundPredictor:
    """
    Handles prediction of bowel sounds from audio sequences via HTTP inference server.
    """

    def __init__(self, model: tf.keras.Model, threshold: float = 0.3,
                 model_type: str = "crnn",
                 server_url: str = "http://127.0.0.1:8502/predict"):
        """
        Initialize predictor.

        Args:
            model: Trained Keras model (kept for compatibility, not used directly)
            threshold: Classification threshold
            model_type: 'crnn' or 'cdnn'
            server_url: URL of the inference server endpoint
        """
        self.model = model
        self.threshold = threshold
        self.config = PredictionConfig()
        self.model_type = model_type
        self.server_url = server_url

    def predict_probabilities(self, sequences: np.ndarray) -> np.ndarray:
        """
        Get probability scores for each sequence via the inference server.
        """
        payload = {
            "model_type": self.model_type,
            "sequences": sequences.tolist(),
            "threshold": float(self.threshold),
        }

        resp = requests.post(self.server_url, json=payload)
        resp.raise_for_status()
        data = resp.json()

        probabilities = np.array(data["probabilities"], dtype=np.float32)
        return probabilities.ravel()

    def predict_binary(self, probabilities: np.ndarray, threshold: float = None) -> np.ndarray:
        """
        Convert probabilities to binary predictions.
        """
        if threshold is None:
            threshold = self.threshold

        return (probabilities >= threshold).astype(int)

    def predict(
        self,
        sequences: np.ndarray,
        threshold: float = None
    ) -> Dict[str, np.ndarray]:
        """
        Complete prediction pipeline.
        """
        probabilities = self.predict_probabilities(sequences)
        predictions = self.predict_binary(probabilities, threshold)

        return {
            "probabilities": probabilities,
            "predictions": predictions,
            "threshold": threshold if threshold is not None else self.threshold,
        }

    # The rest of your methods (get_detection_stats, find_detection_intervals, etc.)
    # remain unchanged; paste them below from your original file.

    def get_detection_stats(self, predictions: Dict) -> Dict:
        probs = predictions["probabilities"]
        preds = predictions["predictions"]

        n_total = len(preds)
        n_detected = np.sum(preds)
        n_noise = n_total - n_detected

        percentage = (n_detected / n_total) * 100 if n_total > 0 else 0
        bowel_sounds_per_minute = (n_detected / 2.0) * 60

        detected_probs = probs[preds == 1]

        stats = {
            "n_total_frames": n_total,
            "n_detected": n_detected,
            "n_noise": n_noise,
            "detection_percentage": percentage,
            "bowel_sounds_per_minute": bowel_sounds_per_minute,
            "mean_probability": float(np.mean(probs)),
            "std_probability": float(np.std(probs)),
            "max_probability": float(np.max(probs)),
            "min_probability": float(np.min(probs)),
            "mean_detected_probability": float(np.mean(detected_probs)) if len(detected_probs) > 0 else 0.0,
            "threshold_used": predictions["threshold"],
        }

        return stats

    def find_detection_intervals(self, predictions: np.ndarray) -> list:
        intervals = []
        in_detection = False
        start_idx = 0

        for i, pred in enumerate(predictions):
            if pred == 1 and not in_detection:
                start_idx = i
                in_detection = True
            elif pred == 0 and in_detection:
                intervals.append((start_idx, i - 1))
                in_detection = False

        if in_detection:
            intervals.append((start_idx, len(predictions) - 1))

        return intervals

    def get_interval_statistics(self, intervals: list, hop_samples: int, sr: int) -> Dict:
        if not intervals:
            return {
                "n_intervals": 0,
                "mean_duration": 0,
                "std_duration": 0,
                "max_duration": 0,
                "min_duration": 0,
                "total_duration": 0,
            }

        durations = []
        for start, end in intervals:
            n_frames = end - start + 1
            duration_sec = (n_frames * hop_samples) / sr
            durations.append(duration_sec)

        durations = np.array(durations)

        return {
            "n_intervals": len(intervals),
            "mean_duration": float(np.mean(durations)),
            "std_duration": float(np.std(durations)),
            "max_duration": float(np.max(durations)),
            "min_duration": float(np.min(durations)),
            "total_duration": float(np.sum(durations)),
            "durations": durations,
        }

    def set_threshold(self, threshold: float):
        self.threshold = float(np.clip(threshold, 0.0, 1.0))

    def get_threshold(self) -> float:
        return self.threshold
