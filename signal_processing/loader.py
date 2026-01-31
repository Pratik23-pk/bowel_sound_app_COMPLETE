"""
Audio Loader Module
Handles audio file loading and basic metadata extraction
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Dict


class AudioLoader:
    """
    Load and extract metadata from audio files
    """
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize audio loader
        
        Args:
            sample_rate: Target sample rate for loading
        """
        self.sample_rate = sample_rate
    
    def load_audio(self, filepath: str) -> Tuple[np.ndarray, Dict]:
        """
        Load audio file and extract metadata
        
        Args:
            filepath: Path to audio file
            
        Returns:
            audio: Audio waveform array
            metadata: Dictionary with audio information
        """
        # Load audio
        audio, sr = librosa.load(filepath, sr=self.sample_rate, mono=True)
        
        # Get file info
        info = sf.info(filepath)
        
        metadata = {
            'original_sr': info.samplerate,
            'original_duration': info.duration,
            'original_channels': info.channels,
            'loaded_sr': sr,
            'loaded_duration': len(audio) / sr,
            'loaded_samples': len(audio),
            'format': info.format,
            'subtype': info.subtype
        }
        
        return audio, metadata
    
    @staticmethod
    def normalize_length(audio: np.ndarray, target_length: int) -> np.ndarray:
        """
        Normalize audio to target length
        
        Args:
            audio: Input audio array
            target_length: Target number of samples
            
        Returns:
            Normalized audio array
        """
        if len(audio) < target_length:
            # Pad with zeros
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        elif len(audio) > target_length:
            # Trim
            audio = audio[:target_length]
        
        return audio
    
    @staticmethod
    def get_audio_stats(audio: np.ndarray) -> Dict:
        """
        Get basic statistics of audio signal
        
        Args:
            audio: Audio waveform
            
        Returns:
            Dictionary with statistics
        """
        return {
            'mean': float(np.mean(audio)),
            'std': float(np.std(audio)),
            'min': float(np.min(audio)),
            'max': float(np.max(audio)),
            'rms': float(np.sqrt(np.mean(audio ** 2))),
            'zero_crossings': int(np.sum(np.abs(np.diff(np.sign(audio)))) / 2)
        }