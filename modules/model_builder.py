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
    Load a trained Keras model with version compatibility handling.
    
    Args:
        model_type: 'crnn' or 'cdnn'
        
    Returns:
        Loaded model or None if not found
    """
    model_path = ModelPaths.get_model_path(model_type)
    
    if model_path is None:
        raise FileNotFoundError(
            f"Model file not found for {model_type.upper()}. "
            f"Please add the model to the models/ directory."
        )
    
    try:
        # Method 1: Try loading with h5py backend (most compatible)
        import h5py
        model = tf.keras.models.load_model(str(model_path))
        print(f"✓ Loaded {model_type.upper()} model from {model_path.name}")
        return model
    except Exception as e1:
        print(f"Method 1 failed: {e1}")
        
        try:
            # Method 2: Load weights only and reconstruct
            print("Attempting to load weights only...")
            
            # We need to rebuild the model architecture from scratch
            # This is a workaround for Keras version incompatibility
            from tensorflow import keras
            from tensorflow.keras import layers
            
            # Build CRNN architecture (matches your model)
            inputs = layers.Input(shape=(9, 16, 1), name='input')
            
            # Conv layers
            x = layers.Conv2D(30, (3, 3), padding='same', activation='relu', name='conv1')(inputs)
            x = layers.MaxPooling2D((1, 2), name='pool1')(x)
            x = layers.Conv2D(60, (4, 2), padding='same', activation='relu', name='conv2')(x)
            x = layers.MaxPooling2D((1, 2), name='pool2')(x)
            
            # Reshape for RNN
            x = layers.Reshape((9, -1), name='reshape')(x)
            x = layers.Dropout(0.4, name='dropout1')(x)
            
            # Bidirectional GRU
            x = layers.Bidirectional(
                layers.GRU(80, activation='relu'), 
                name='bi_gru'
            )(x)
            x = layers.Dropout(0.4, name='dropout2')(x)
            
            # Output
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
            
            # Create model
            model = keras.Model(inputs=inputs, outputs=outputs, name='CRNN')
            
            # Try to load weights
            model.load_weights(str(model_path))
            
            # Compile
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            print(f"✓ Loaded {model_type.upper()} model (rebuilt architecture)")
            return model
            
        except Exception as e2:
            print(f"Method 2 failed: {e2}")
            
            # Method 3: Last resort - inform user to convert model
            raise RuntimeError(
                f"Could not load model due to Keras version incompatibility.\n"
                f"Your model was saved with Keras 3.x but you have Keras 2.x.\n\n"
                f"SOLUTION: Convert your model in your training notebook:\n"
                f"```python\n"
                f"import tensorflow as tf\n"
                f"model = tf.keras.models.load_model('crnn_best.keras')\n"
                f"model.save('crnn_best.h5', save_format='h5')\n"
                f"```\n"
                f"Then use the .h5 file instead."
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
    except FileNotFoundError as e:
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
        'trainable_params': sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
        'non_trainable_params': sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]),
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
        self.models = {}
        self.mean = None
        self.std = None
        self.threshold = None
        
    def load_all_components(self, model_type: str = 'crnn') -> Dict:
        """
        Load model and all required components.
        
        Args:
            model_type: 'crnn' or 'cdnn'
            
        Returns:
            Dictionary with all loaded components
        """
        # Load model
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