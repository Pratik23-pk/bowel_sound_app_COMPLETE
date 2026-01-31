"""
Audio Preprocessing Module
Signal filtering, enhancement, and preparation
"""

import numpy as np
import scipy.signal as signal
from typing import Tuple


class AudioPreprocessor:
    """
    Preprocess audio signals for analysis
    """
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize preprocessor
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
    
    def bandpass_filter(
        self, 
        audio: np.ndarray, 
        low_freq: float = 200, 
        high_freq: float = 800,
        order: int = 4
    ) -> np.ndarray:
        """
        Apply bandpass filter
        
        Args:
            audio: Input audio
            low_freq: Low cutoff frequency (Hz)
            high_freq: High cutoff frequency (Hz)
            order: Filter order
            
        Returns:
            Filtered audio
        """
        nyquist = self.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        b, a = signal.butter(order, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, audio)
        
        return filtered
    
    def extract_envelope(self, audio: np.ndarray, smooth_window_ms: float = 50) -> np.ndarray:
        """
        Extract signal envelope using Hilbert transform
        
        Args:
            audio: Input audio
            smooth_window_ms: Smoothing window size in milliseconds
            
        Returns:
            Signal envelope
        """
        # Hilbert transform
        analytic_signal = signal.hilbert(audio)
        envelope = np.abs(analytic_signal)
        
        # Smooth envelope
        window_size = int(smooth_window_ms / 1000 * self.sample_rate)
        if window_size % 2 == 0:
            window_size += 1
        
        envelope_smooth = signal.savgol_filter(envelope, window_size, 3)
        
        return envelope_smooth
    
    def remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """
        Remove DC offset from audio
        
        Args:
            audio: Input audio
            
        Returns:
            Audio with DC offset removed
        """
        return audio - np.mean(audio)
    
    def normalize_amplitude(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio amplitude to [-1, 1]
        
        Args:
            audio: Input audio
            
        Returns:
            Normalized audio
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio
    
    def denoise(self, audio: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
        """
        Simple noise reduction using spectral gating
        
        Args:
            audio: Input audio
            noise_factor: Noise threshold factor
            
        Returns:
            Denoised audio
        """
        # Compute STFT
        f, t, Zxx = signal.stft(audio, self.sample_rate, nperseg=1024)
        
        # Estimate noise from first and last frames
        noise_spectrum = np.mean(np.abs(Zxx[:, [0, -1]]), axis=1, keepdims=True)
        
        # Apply spectral gating
        threshold = noise_factor * noise_spectrum
        mask = np.abs(Zxx) > threshold
        Zxx_clean = Zxx * mask
        
        # Inverse STFT
        _, audio_clean = signal.istft(Zxx_clean, self.sample_rate, nperseg=1024)
        
        return audio_clean