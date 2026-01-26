"""
Prediction Module
Handles bowel sound prediction from preprocessed audio sequences
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Tuple
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import PredictionConfig


class BowelSoundPredictor:
    """
    Handles prediction of bowel sounds from audio sequences.
    """
    
    def __init__(self, model: tf.keras.Model, threshold: float = 0.3):
        """
        Initialize predictor.
        
        Args:
            model: Trained Keras model
            threshold: Classification threshold (default: 0.3)
        """
        self.model = model
        self.threshold = threshold
        self.config = PredictionConfig()
        
    def predict_probabilities(self, sequences: np.ndarray) -> np.ndarray:
        """
        Get probability scores for each sequence.
        
        Args:
            sequences: Input sequences (n_seq, seq_len, n_freq, 1)
            
        Returns:
            probabilities: Array of probabilities (n_seq,)
        """
        probabilities = self.model.predict(
            sequences,
            batch_size=self.config.BATCH_SIZE,
            verbose=self.config.VERBOSE
        )
        
        return probabilities.ravel()
    
    def predict_binary(self, probabilities: np.ndarray, threshold: float = None) -> np.ndarray:
        """
        Convert probabilities to binary predictions.
        
        Args:
            probabilities: Probability array
            threshold: Classification threshold (uses self.threshold if None)
            
        Returns:
            predictions: Binary array (0=noise, 1=bowel sound)
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
        
        Args:
            sequences: Input sequences
            threshold: Classification threshold
            
        Returns:
            Dictionary with probabilities and predictions
        """
        # Get probabilities
        probabilities = self.predict_probabilities(sequences)
        
        # Get binary predictions
        predictions = self.predict_binary(probabilities, threshold)
        
        return {
            'probabilities': probabilities,
            'predictions': predictions,
            'threshold': threshold if threshold is not None else self.threshold
        }
    
    def get_detection_stats(self, predictions: Dict) -> Dict:
        """
        Calculate detection statistics.
        
        Args:
            predictions: Prediction dictionary from self.predict()
            
        Returns:
            Dictionary with statistics
        """
        probs = predictions['probabilities']
        preds = predictions['predictions']
        
        n_total = len(preds)
        n_detected = np.sum(preds)
        n_noise = n_total - n_detected
        
        percentage = (n_detected / n_total) * 100 if n_total > 0 else 0
        
        # Estimate bowel sounds per minute (assuming 2-second audio)
        bowel_sounds_per_minute = (n_detected / 2.0) * 60
        
        # Confidence statistics
        detected_probs = probs[preds == 1]
        
        stats = {
            'n_total_frames': n_total,
            'n_detected': n_detected,
            'n_noise': n_noise,
            'detection_percentage': percentage,
            'bowel_sounds_per_minute': bowel_sounds_per_minute,
            'mean_probability': np.mean(probs),
            'std_probability': np.std(probs),
            'max_probability': np.max(probs),
            'min_probability': np.min(probs),
            'mean_detected_probability': np.mean(detected_probs) if len(detected_probs) > 0 else 0,
            'threshold_used': predictions['threshold']
        }
        
        return stats
    
    def find_detection_intervals(self, predictions: np.ndarray) -> list:
        """
        Find continuous intervals where bowel sounds are detected.
        
        Args:
            predictions: Binary prediction array
            
        Returns:
            List of (start_idx, end_idx) tuples
        """
        intervals = []
        in_detection = False
        start_idx = 0
        
        for i, pred in enumerate(predictions):
            if pred == 1 and not in_detection:
                # Start of detection
                start_idx = i
                in_detection = True
            elif pred == 0 and in_detection:
                # End of detection
                intervals.append((start_idx, i - 1))
                in_detection = False
        
        # Handle case where detection goes to end
        if in_detection:
            intervals.append((start_idx, len(predictions) - 1))
        
        return intervals
    
    def get_interval_statistics(self, intervals: list, hop_samples: int, sr: int) -> Dict:
        """
        Calculate statistics for detection intervals.
        
        Args:
            intervals: List of (start, end) tuples
            hop_samples: Hop length in samples
            sr: Sample rate
            
        Returns:
            Dictionary with interval statistics
        """
        if not intervals:
            return {
                'n_intervals': 0,
                'mean_duration': 0,
                'std_duration': 0,
                'max_duration': 0,
                'min_duration': 0,
                'total_duration': 0
            }
        
        # Convert frame indices to durations (seconds)
        durations = []
        for start, end in intervals:
            n_frames = end - start + 1
            duration_sec = (n_frames * hop_samples) / sr
            durations.append(duration_sec)
        
        durations = np.array(durations)
        
        return {
            'n_intervals': len(intervals),
            'mean_duration': np.mean(durations),
            'std_duration': np.std(durations),
            'max_duration': np.max(durations),
            'min_duration': np.min(durations),
            'total_duration': np.sum(durations),
            'durations': durations
        }
    
    def set_threshold(self, threshold: float):
        """Update classification threshold."""
        self.threshold = np.clip(threshold, 0.0, 1.0)
    
    def get_threshold(self) -> float:
        """Get current threshold."""
        return self.threshold


# Example usage
if __name__ == "__main__":
    print("BowelSoundPredictor module loaded")
    
    # Create dummy data for testing
    dummy_sequences = np.random.randn(100, 9, 50, 1).astype(np.float32)
    
    # Would need actual model to test prediction
    print(f"Dummy sequences shape: {dummy_sequences.shape}")
    print("Ready for prediction with real model!")