"""
Bowel Sound Detection Application - Modules Package
Contains all core functionality for audio processing, prediction, and visualization
"""

from .audio_processor import AudioProcessor
from .model_builder import load_model_file, get_available_models
from .predictor import BowelSoundPredictor
from .visualizer import AudioVisualizer, ResultsVisualizer
from .evaluator import MetricsCalculator

__all__ = [
    'AudioProcessor',
    'load_model_file',
    'get_available_models',
    'BowelSoundPredictor',
    'AudioVisualizer',
    'ResultsVisualizer',
    'MetricsCalculator'
]

__version__ = '1.0.0'
__author__ = 'Bowel Sound Detection Team'