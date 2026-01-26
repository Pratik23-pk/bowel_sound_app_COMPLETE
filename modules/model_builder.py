"""
Model Loading Module
Handles loading trained models and standardization parameters
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Optional, Dict

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import ModelPaths, ModelConfig


def load_model_file(model_type: str = 'crnn') -> Optional[tf.keras.Model]:
    """
    Load a trained Keras model in a version-safe way.

    Args:
        model_type: 'crnn' or 'cdnn'

    Returns:
        Loaded model

    Raises:
        FileNotFoundError, RuntimeError
    """
    model_path = ModelPaths.get_model_path(model_type)

    if model_path is None:
        raise FileNotFoundError(
            f"Model file not found for {model_type.upper()}. "
            f"Please add the model to the models/ directory."
        )

    try:
        # Single safe path: load SavedModel / .keras / .h5 with compile=False
        # This avoids legacy optimizer/config issues that can cause crashes.
        model = tf.keras.models.load_model(str(model_path), compile=False)
        print(f"✓ Loaded {model_type.upper()} model from {model_path.name}")
        return model
    except Exception as e:
        # Fail cleanly instead of attempting fragile architecture reconstruction
        raise RuntimeError(
            f"Could not load {model_type.upper()} model from {model_path}.\n"
            f"Error: {e}"
        )


def load_standardization_params() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load standardization mean and std from training.

    Returns:
        mean: Standardization mean array
        std: Standardization std array
    """
    paths = ModelPaths.get_standardization_paths()

    try:
        mean = np.load(paths['mean'])
        std = np.load(paths['std'])
        print(f"✓ Loaded standardization parameters")
        print(f"  Mean shape: {mean.shape}")
        print(f"  Std shape: {std.shape}")
        return mean, std
    except FileNotFoundError:
        raise FileNotFoundError(
            "Standardization files not found. "
            "Please add standardization_mean.npy and standardization_std.npy "
            "to the models/ directory."
        )
    except Exception as e:
        raise RuntimeError(f"Error loading standardization params: {str(e)}")


def load_optimal_threshold() -> float:
    """
    Load optimal threshold from training.

    Returns:
        Optimal threshold value (or default if not found)
    """
    paths = ModelPaths.get_standardization_paths()
    threshold_path = paths['threshold']

    try:
        if threshold_path.exists():
            threshold = float(np.load(threshold_path))
            print(f"✓ Loaded optimal threshold: {threshold:.4f}")
            return threshold
        else:
            print(f"⚠️ Threshold file not found, using default: {ModelConfig.DEFAULT_THRESHOLD}")
            return ModelConfig.DEFAULT_THRESHOLD
    except Exception as e:
        print(f"⚠️ Error loading threshold, using default: {str(e)}")
        return ModelConfig.DEFAULT_THRESHOLD


def get_available_models() -> Dict[str, bool]:
    """
    Check which models are available.

    Returns:
        Dictionary with model availability
    """
    availability = {
        'crnn': ModelPaths.get_model_path('crnn') is not None,
        'cdnn': ModelPaths.get_model_path('cdnn') is not None
    }

    return availability


def get_model_info(model: tf.keras.Model) -> Dict:
    """
    Get information about a loaded model.

    Args:
        model: Loaded Keras model

    Returns:
        Dictionary with model information
    """
    info = {
        'name': model.name,
        'total_params': model.count_params(),
        'trainable_params': sum(
            [tf.keras.backend.count_params(w) for w in model.trainable_weights]
        ),
        'non_trainable_params': sum(
            [tf.keras.backend.count_params(w) for w in model.non_trainable_weights]
        ),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'layers': len(model.layers)
    }

    return info


def validate_model_compatibility(model: tf.keras.Model, expected_input_shape: Tuple) -> bool:
    """
    Validate that model has expected input shape.

    Args:
        model: Loaded model
        expected_input_shape: Expected input shape (excluding batch dimension)

    Returns:
        True if compatible, False otherwise
    """
    actual_shape = model.input_shape[1:]  # Remove batch dimension

    # Check if shapes match
    if len(actual_shape) != len(expected_input_shape):
        return False

    for actual, expected in zip(actual_shape, expected_input_shape):
        if expected is not None and actual != expected:
            return False

    return True


class ModelManager:
    """
    Manages model loading and caching.
    """

    def __init__(self):
        self.models: Dict[str, tf.keras.Model] = {}
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None

    def load_all_components(self, model_type: str = 'crnn') -> Dict:
        """
        Load model and all required components.

        Args:
            model_type: 'crnn' or 'cdnn'

        Returns:
            Dictionary with all loaded components
        """
        # Load model (cached)
        if model_type not in self.models:
            self.models[model_type] = load_model_file(model_type)

        # Load standardization params (shared across models)
        if self.mean is None or self.std is None:
            self.mean, self.std = load_standardization_params()

        # Load threshold
        if self.threshold is None:
            self.threshold = load_optimal_threshold()

        return {
            'model': self.models[model_type],
            'mean': self.mean,
            'std': self.std,
            'threshold': self.threshold
        }

    def get_model(self, model_type: str) -> tf.keras.Model:
        """Get cached model or load if not cached."""
        if model_type not in self.models:
            self.models[model_type] = load_model_file(model_type)
        return self.models[model_type]

    def clear_cache(self):
        """Clear cached models to free memory."""
        self.models.clear()
        self.mean = None
        self.std = None
        self.threshold = None


# Singleton instance
_model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """Get the singleton ModelManager instance."""
    return _model_manager


# Example usage
if __name__ == "__main__":
    print("Testing Model Builder...")

    # Check available models
    available = get_available_models()
    print(f"\nAvailable models:")
    print(f"  CRNN: {available['crnn']}")
    print(f"  CDNN: {available['cdnn']}")

    # Try loading components
    try:
        manager = get_model_manager()
        components = manager.load_all_components('crnn')
        print(f"\n✓ All components loaded successfully!")

        # Get model info
        info = get_model_info(components['model'])
        print(f"\nModel Info:")
        print(f"  Name: {info['name']}")
        print(f"  Total params: {info['total_params']:,}")
        print(f"  Input shape: {info['input_shape']}")
        print(f"  Output shape: {info['output_shape']}")
    except Exception as e:
        print(f"\n⚠️ Error: {str(e)}")
