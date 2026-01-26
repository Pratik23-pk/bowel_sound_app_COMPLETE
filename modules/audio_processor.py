"""
Audio Processing Module
Handles loading, preprocessing, and spectrogram generation for audio files
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import AudioConfig, ModelConfig


class AudioProcessor:
    """
    Handles all audio preprocessing operations for bowel sound detection.
    """
    
    def __init__(self):
        self.config = AudioConfig()
        self.model_config = ModelConfig()
        
    def load_audio(self, filepath: str) -> Tuple[np.ndarray, dict]:
        """
        Load audio file and return waveform with metadata.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            audio: Audio waveform array
            metadata: Dictionary with audio information
        """
        try:
            # Load audio
            audio, sr = librosa.load(filepath, sr=self.config.SR, mono=True)
            
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
            
        except Exception as e:
            raise ValueError(f"Error loading audio file: {str(e)}")
    
    def normalize_length(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to target length (2 seconds).
        Pads with zeros if too short, trims if too long.
        
        Args:
            audio: Input audio array
            
        Returns:
            Normalized audio array
        """
        target_length = self.config.TARGET_SAMPLES
        
        if len(audio) < target_length:
            # Pad with zeros
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        elif len(audio) > target_length:
            # Trim to target length
            audio = audio[:target_length]
        
        return audio
    
    def compute_spectrogram(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute STFT spectrogram filtered to match model's expected frequency bins.
        
        Args:
            audio: Audio waveform
            
        Returns:
            S_db_filtered: Spectrogram in dB (16, n_frames) - EXACTLY 16 bins
            freqs_filtered: Frequency bins (Hz)
        """
        # Compute STFT
        S = librosa.stft(
            audio,
            n_fft=self.config.FRAME_SAMPLES,
            hop_length=self.config.HOP_SAMPLES,
            window=self.config.WINDOW_TYPE,
            center=True
        )
        
        # Convert to magnitude and dB
        S_mag = np.abs(S)
        S_db = librosa.amplitude_to_db(S_mag, ref=np.max)
        
        # Get frequency bins
        freqs = librosa.fft_frequencies(
            sr=self.config.SR, 
            n_fft=self.config.FRAME_SAMPLES
        )
        
        # Filter to 0-1500 Hz
        freq_mask = (freqs >= self.config.MIN_FREQ) & (freqs <= self.config.MAX_FREQ)
        S_db_filtered = S_db[freq_mask, :]
        freqs_filtered = freqs[freq_mask]
        
        # CRITICAL: Resize to exactly 16 frequency bins to match model
        from scipy.ndimage import zoom
        
        current_bins = S_db_filtered.shape[0]
        target_bins = 16  # Model expects exactly 16 bins
        
        if current_bins != target_bins:
            # Resize frequency dimension to 16 bins
            zoom_factor = target_bins / current_bins
            S_db_filtered = zoom(S_db_filtered, (zoom_factor, 1), order=1)
            
            # Also resize frequency array
            freqs_filtered = np.linspace(freqs_filtered[0], freqs_filtered[-1], target_bins)
        
        return S_db_filtered, freqs_filtered
    
    def spectrogram_to_frames(self, S_db: np.ndarray) -> np.ndarray:
        """
        Convert spectrogram to frame representation.
        
        Args:
            S_db: Spectrogram (n_freq_bins, n_frames)
            
        Returns:
            frames: Frame array (n_frames, n_freq_bins)
        """
        return S_db.T
    
    def standardize_frames(
        self, 
        frames: np.ndarray, 
        mean: np.ndarray, 
        std: np.ndarray
    ) -> np.ndarray:
        """
        Standardize frames using provided mean and std.
        
        Args:
            frames: Frame array (n_frames, n_freq_bins)
            mean: Mean array from training
            std: Std array from training
            
        Returns:
            Standardized frames
        """
        return (frames - mean) / std
    
    def build_sequences(self, frames: np.ndarray) -> np.ndarray:
        """
        Build sequences of frames for temporal context.
        
        Args:
            frames: Frame array (n_frames, n_freq_bins)
            
        Returns:
            sequences: Sequence array (n_sequences, seq_len, n_freq_bins, 1)
        """
        half = self.model_config.HALF_SEQ
        n_frames, n_features = frames.shape
        
        sequences = []
        
        # Build sequences with center frame
        for i in range(half, n_frames - half):
            seq = frames[i - half : i + half + 1]  # (seq_len, n_features)
            sequences.append(seq)
        
        sequences = np.array(sequences)
        
        # Add channel dimension for Conv2D
        sequences = sequences[..., np.newaxis]  # (n_seq, seq_len, n_features, 1)
        
        return sequences
    
    def preprocess_for_prediction(
        self,
        filepath: str,
        mean: np.ndarray,
        std: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """
        Complete preprocessing pipeline for prediction.
        
        Args:
            filepath: Path to audio file
            mean: Standardization mean from training
            std: Standardization std from training
            
        Returns:
            sequences: Ready for model.predict() (n_seq, seq_len, n_freq, 1)
            info: Dictionary with preprocessing information
        """
        # Load audio
        audio, metadata = self.load_audio(filepath)
        
        # Normalize length
        audio = self.normalize_length(audio)
        
        # Compute spectrogram
        spectrogram, freqs = self.compute_spectrogram(audio)
        
        # Convert to frames
        frames = self.spectrogram_to_frames(spectrogram)
        
        # Standardize
        frames_std = self.standardize_frames(frames, mean, std)
        
        # Build sequences
        sequences = self.build_sequences(frames_std)
        
        # Preprocessing info
        info = {
            'metadata': metadata,
            'audio_samples': len(audio),
            'spectrogram_shape': spectrogram.shape,
            'n_freq_bins': spectrogram.shape[0],
            'n_frames': frames.shape[0],
            'n_sequences': sequences.shape[0],
            'sequence_shape': sequences.shape,
            'freq_range': (freqs[0], freqs[-1])
        }
        
        return sequences, info
    
    def get_time_axis(self, n_frames: int) -> np.ndarray:
        """
        Get time axis for plotting.
        
        Args:
            n_frames: Number of frames
            
        Returns:
            Time axis in seconds
        """
        return np.arange(n_frames) * (self.config.HOP_SAMPLES / self.config.SR)


# Example usage
if __name__ == "__main__":
    processor = AudioProcessor()
    print("AudioProcessor initialized successfully!")
    print(f"Sample rate: {processor.config.SR} Hz")
    print(f"Frame size: {processor.config.FRAME_SAMPLES} samples")
    print(f"Hop length: {processor.config.HOP_SAMPLES} samples")
    print(f"Sequence length: {processor.model_config.SEQ_LEN} frames")