"""
Feature Extraction Module
Extract acoustic features from audio signals
"""

import numpy as np
import librosa
import scipy.signal as signal
from scipy import stats
from typing import Dict, Tuple


class FeatureExtractor:
    """
    Extract various features from audio signals
    """
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize feature extractor
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
    
    def extract_energy_features(self, audio: np.ndarray) -> Dict:
        """
        Extract energy-based features
        
        Args:
            audio: Audio waveform
            
        Returns:
            Dictionary of energy features
        """
        # Total energy
        total_energy = np.sum(audio ** 2)
        
        # Energy in frequency bands (avoid 0 Hz)
        bands = {
            'low': (50, 200),      # Changed from (0, 200)
            'mid': (200, 800),
            'high': (800, 2000)
        }
        
        band_energies = {}
        nyquist = self.sample_rate / 2
        
        for band_name, (low, high) in bands.items():
            # Normalize frequencies
            low_norm = max(low / nyquist, 0.001)  # Ensure > 0
            high_norm = min(high / nyquist, 0.999)  # Ensure < 1
            
            # Skip if invalid range
            if low_norm >= high_norm:
                band_energies[f'{band_name}_band_energy'] = 0.0
                band_energies[f'{band_name}_band_pct'] = 0.0
                continue
            
            try:
                # Bandpass filter
                b, a = signal.butter(4, [low_norm, high_norm], btype='band')
                filtered = signal.filtfilt(b, a, audio)
                
                band_energy = np.sum(filtered ** 2)
                band_energies[f'{band_name}_band_energy'] = float(band_energy)
                band_energies[f'{band_name}_band_pct'] = float(band_energy / total_energy * 100) if total_energy > 0 else 0
            except Exception as e:
                # If filter fails, set energy to 0
                band_energies[f'{band_name}_band_energy'] = 0.0
                band_energies[f'{band_name}_band_pct'] = 0.0
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        
        return {
            'total_energy': float(total_energy),
            **band_energies,
            'spectral_centroid_mean_hz': float(np.mean(spectral_centroid)),
            'spectral_centroid_std_hz': float(np.std(spectral_centroid))
        }
    
    def extract_snr(self, audio: np.ndarray, noise_percentage: float = 0.1) -> float:
        """
        Calculate Signal-to-Noise Ratio
        
        Args:
            audio: Audio waveform
            noise_percentage: Percentage of audio to use for noise estimation
            
        Returns:
            SNR in dB
        """
        noise_samples = int(noise_percentage * len(audio))
        noise = np.concatenate([audio[:noise_samples], audio[-noise_samples:]])
        
        signal_part = audio[noise_samples:-noise_samples]
        
        signal_power = np.mean(signal_part ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = float('inf')
        
        return float(snr)
    
    def extract_temporal_features(self, peaks: np.ndarray) -> Dict:
        """
        Extract temporal features from detected peaks
        
        Args:
            peaks: Array of peak indices
            
        Returns:
            Dictionary of temporal features
        """
        if len(peaks) < 2:
            return {
                'n_events': len(peaks),
                'events_per_minute': 0,
                'mean_iei_ms': 0,
                'median_iei_ms': 0,
                'std_iei_ms': 0,
                'iei_cv': 0,
                'min_iei_ms': 0,
                'max_iei_ms': 0
            }
        
        # Inter-event intervals in milliseconds
        iei = np.diff(peaks) / self.sample_rate * 1000
        
        duration_s = peaks[-1] / self.sample_rate
        
        return {
            'n_events': len(peaks),
            'events_per_minute': float(len(peaks) / duration_s * 60) if duration_s > 0 else 0,
            'mean_iei_ms': float(np.mean(iei)),
            'median_iei_ms': float(np.median(iei)),
            'std_iei_ms': float(np.std(iei)),
            'iei_cv': float(np.std(iei) / np.mean(iei)) if np.mean(iei) > 0 else 0,
            'min_iei_ms': float(np.min(iei)),
            'max_iei_ms': float(np.max(iei))
        }
    
    def extract_variability_features(self, peaks: np.ndarray) -> Dict:
        """
        Extract variability features (similar to HRV)
        
        Args:
            peaks: Array of peak indices
            
        Returns:
            Dictionary of variability features
        """
        if len(peaks) < 2:
            return {
                'sdnn_ms': 0,
                'rmssd_ms': 0,
                'pnn50_pct': 0,
                'cv': 0,
                'sd1_ms': 0,
                'sd2_ms': 0
            }
        
        # Inter-event intervals in milliseconds
        iei = np.diff(peaks) / self.sample_rate * 1000
        
        # SDNN: Standard deviation
        sdnn = float(np.std(iei))
        
        # RMSSD: Root mean square of successive differences
        successive_diffs = np.diff(iei)
        rmssd = float(np.sqrt(np.mean(successive_diffs ** 2)))
        
        # pNN50: Percentage of successive differences > 50ms
        pnn50 = float(np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs) * 100)
        
        # CV: Coefficient of variation
        cv = float(sdnn / np.mean(iei)) if np.mean(iei) > 0 else 0
        
        # Poincaré metrics
        if len(iei) > 1:
            sd1 = float(np.std(successive_diffs) / np.sqrt(2))
            mean_centered = iei - np.mean(iei)
            sd2 = float(np.sqrt(2 * np.var(mean_centered) - sd1**2))
        else:
            sd1 = 0
            sd2 = 0
        
        return {
            'sdnn_ms': sdnn,
            'rmssd_ms': rmssd,
            'pnn50_pct': pnn50,
            'cv': cv,
            'sd1_ms': sd1,
            'sd2_ms': sd2
        }
    
    def extract_frequency_band_features(self, audio: np.ndarray, target_band: Tuple[float, float] = (200, 800)) -> Dict:
        """
        Extract features from specific frequency band
        
        Args:
            audio: Audio waveform
            target_band: Frequency band (low, high) in Hz
            
        Returns:
            Dictionary of band features
        """
        nyquist = self.sample_rate / 2
        low_norm = max(target_band[0] / nyquist, 0.001)  # Ensure > 0
        high_norm = min(target_band[1] / nyquist, 0.999)  # Ensure < 1
        
        # Validate frequency range
        if low_norm >= high_norm:
            return {
                'target_band_hz': list(target_band),
                'band_energy': 0.0,
                'band_energy_pct': 0.0
            }
        
        try:
            # Bandpass filter
            b, a = signal.butter(4, [low_norm, high_norm], btype='band')
            filtered = signal.filtfilt(b, a, audio)
            
            # Calculate energy
            band_energy = np.sum(filtered ** 2)
            total_energy = np.sum(audio ** 2)
            
            return {
                'target_band_hz': list(target_band),
                'band_energy': float(band_energy),
                'band_energy_pct': float(band_energy / total_energy * 100) if total_energy > 0 else 0
            }
        except Exception as e:
            # If filter fails, return zeros
            return {
                'target_band_hz': list(target_band),
                'band_energy': 0.0,
                'band_energy_pct': 0.0
            }
    
    def extract_irregularity_features(self, peaks: np.ndarray) -> Dict:
        """
        Extract irregularity features from event spacing
        
        Args:
            peaks: Array of peak indices
            
        Returns:
            Dictionary of irregularity features
        """
        if len(peaks) < 2:
            return {
                'intervals_mean_s': 0,
                'intervals_std_s': 0,
                'intervals_cv': 0,
                'n_outliers': 0
            }
        
        # Inter-event intervals in seconds
        iei_seconds = np.diff(peaks) / self.sample_rate
        
        # Detect outliers using z-score
        z_scores = np.abs(stats.zscore(iei_seconds))
        outliers = np.sum(z_scores > 2)
        
        return {
            'intervals_mean_s': float(np.mean(iei_seconds)),
            'intervals_std_s': float(np.std(iei_seconds)),
            'intervals_cv': float(np.std(iei_seconds) / np.mean(iei_seconds)) if np.mean(iei_seconds) > 0 else 0,
            'n_outliers': int(outliers)
        }
