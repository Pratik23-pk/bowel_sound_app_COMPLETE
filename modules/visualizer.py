"""
Visualization Module
Handles all plotting and visualization for audio and predictions
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from typing import Optional, Tuple
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import VisualizationConfig, AudioConfig


class AudioVisualizer:
    """
    Handles visualization of audio waveforms and spectrograms.
    """
    
    def __init__(self):
        self.config = VisualizationConfig()
        self.audio_config = AudioConfig()
        
    def plot_waveform(
        self, 
        audio: np.ndarray, 
        sr: int = None,
        title: str = "Audio Waveform"
    ) -> plt.Figure:
        """
        Plot audio waveform.
        
        Args:
            audio: Audio array
            sr: Sample rate (uses default if None)
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        if sr is None:
            sr = self.audio_config.SR
        
        fig, ax = plt.subplots(figsize=self.config.WAVEFORM_FIGSIZE, dpi=self.config.DPI)
        
        # Time axis
        time = np.arange(len(audio)) / sr
        
        # Plot waveform
        ax.plot(time, audio, color=self.config.PRIMARY_COLOR, linewidth=1, alpha=0.8)
        ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.set_ylabel('Amplitude', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=self.config.GRID_ALPHA)
        
        plt.tight_layout()
        return fig
    
    def plot_spectrogram(
        self,
        spectrogram: np.ndarray,
        sr: int = None,
        hop_length: int = None,
        title: str = "Spectrogram"
    ) -> plt.Figure:
        """
        Plot spectrogram.
        
        Args:
            spectrogram: Spectrogram array (n_freq, n_frames)
            sr: Sample rate
            hop_length: Hop length
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        if sr is None:
            sr = self.audio_config.SR
        if hop_length is None:
            hop_length = self.audio_config.HOP_SAMPLES
        
        fig, ax = plt.subplots(figsize=self.config.SPECTROGRAM_FIGSIZE, dpi=self.config.DPI)
        
        # Display spectrogram
        img = librosa.display.specshow(
            spectrogram,
            sr=sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis='hz',
            cmap='viridis',
            ax=ax
        )
        
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.set_ylabel('Frequency (Hz)', fontsize=11)
        
        # Add colorbar
        cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
        cbar.set_label('Amplitude (dB)', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_preprocessing_summary(
        self,
        audio: np.ndarray,
        spectrogram: np.ndarray,
        sr: int = None,
        hop_length: int = None
    ) -> plt.Figure:
        """
        Plot both waveform and spectrogram in one figure.
        
        Args:
            audio: Audio array
            spectrogram: Spectrogram array
            sr: Sample rate
            hop_length: Hop length
            
        Returns:
            matplotlib Figure
        """
        if sr is None:
            sr = self.audio_config.SR
        if hop_length is None:
            hop_length = self.audio_config.HOP_SAMPLES
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=self.config.DPI)
        
        # Waveform
        time = np.arange(len(audio)) / sr
        axes[0].plot(time, audio, color=self.config.PRIMARY_COLOR, linewidth=1, alpha=0.8)
        axes[0].set_xlabel('Time (seconds)', fontsize=11)
        axes[0].set_ylabel('Amplitude', fontsize=11)
        axes[0].set_title('Audio Waveform', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=self.config.GRID_ALPHA)
        
        # Spectrogram
        img = librosa.display.specshow(
            spectrogram,
            sr=sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis='hz',
            cmap='viridis',
            ax=axes[1]
        )
        axes[1].set_title('Spectrogram (0-1500 Hz)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Time (seconds)', fontsize=11)
        axes[1].set_ylabel('Frequency (Hz)', fontsize=11)
        
        # Colorbar
        cbar = fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
        cbar.set_label('Amplitude (dB)', fontsize=10)
        
        plt.tight_layout()
        return fig


class ResultsVisualizer:
    """
    Handles visualization of prediction results.
    """
    
    def __init__(self):
        self.config = VisualizationConfig()
        self.audio_config = AudioConfig()
    
    def plot_prediction_timeline(
        self,
        probabilities: np.ndarray,
        predictions: np.ndarray,
        threshold: float,
        hop_samples: int = None,
        sr: int = None
    ) -> plt.Figure:
        """
        Plot prediction timeline with probabilities and detections.
        
        Args:
            probabilities: Probability array
            predictions: Binary prediction array
            threshold: Classification threshold
            hop_samples: Hop length
            sr: Sample rate
            
        Returns:
            matplotlib Figure
        """
        if hop_samples is None:
            hop_samples = self.audio_config.HOP_SAMPLES
        if sr is None:
            sr = self.audio_config.SR
        
        # Time axis
        time = np.arange(len(probabilities)) * (hop_samples / sr)
        
        fig, ax = plt.subplots(figsize=self.config.PREDICTION_FIGSIZE, dpi=self.config.DPI)
        
        # Plot probability curve
        ax.plot(
            time, 
            probabilities, 
            color=self.config.PRIMARY_COLOR,
            linewidth=self.config.LINE_WIDTH,
            label='Probability',
            zorder=3
        )
        
        # Plot threshold line
        ax.axhline(
            y=threshold,
            color=self.config.THRESHOLD_COLOR,
            linestyle='--',
            linewidth=self.config.LINE_WIDTH,
            label=f'Threshold ({threshold:.3f})',
            zorder=2
        )
        
        # Highlight detected regions
        ax.fill_between(
            time,
            0,
            1,
            where=predictions > 0,
            alpha=0.3,
            color=self.config.DETECTION_COLOR,
            label='Detected Bowel Sounds',
            zorder=1
        )
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title('Bowel Sound Detection Timeline', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=self.config.GRID_ALPHA)
        
        plt.tight_layout()
        return fig
    
    def plot_probability_histogram(
        self,
        probabilities: np.ndarray,
        predictions: np.ndarray,
        threshold: float
    ) -> plt.Figure:
        """
        Plot histogram of prediction probabilities.
        
        Args:
            probabilities: Probability array
            predictions: Binary predictions
            threshold: Classification threshold
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 5), dpi=self.config.DPI)
        
        # Separate probabilities by prediction
        noise_probs = probabilities[predictions == 0]
        bowel_probs = probabilities[predictions == 1]
        
        # Plot histograms
        bins = np.linspace(0, 1, 30)
        ax.hist(
            noise_probs, 
            bins=bins, 
            alpha=0.6, 
            color='gray',
            label=f'Noise (n={len(noise_probs)})',
            edgecolor='black'
        )
        ax.hist(
            bowel_probs, 
            bins=bins, 
            alpha=0.6, 
            color=self.config.DETECTION_COLOR,
            label=f'Bowel Sounds (n={len(bowel_probs)})',
            edgecolor='black'
        )
        
        # Threshold line
        ax.axvline(
            x=threshold,
            color=self.config.THRESHOLD_COLOR,
            linestyle='--',
            linewidth=2,
            label=f'Threshold ({threshold:.3f})'
        )
        
        ax.set_xlabel('Probability', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Probability Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=self.config.GRID_ALPHA, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_metrics_summary(self, stats: dict) -> plt.Figure:
        """
        Plot summary metrics as a figure.
        
        Args:
            stats: Statistics dictionary
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.config.DPI)
        ax.axis('off')
        
        # Create text summary
        summary_text = f"""
        PREDICTION SUMMARY
        {'='*50}
        
        Detection Statistics:
        • Total Frames Analyzed:      {stats['n_total_frames']:,}
        • Bowel Sound Frames:          {stats['n_detected']:,}
        • Noise Frames:                {stats['n_noise']:,}
        • Detection Percentage:        {stats['detection_percentage']:.2f}%
        
        Temporal Analysis:
        • Bowel Sounds per Minute:     {stats['bowel_sounds_per_minute']:.1f}
        
        Confidence Metrics:
        • Mean Probability:            {stats['mean_probability']:.4f}
        • Std Probability:             {stats['std_probability']:.4f}
        • Max Probability:             {stats['max_probability']:.4f}
        • Min Probability:             {stats['min_probability']:.4f}
        • Mean Detected Probability:   {stats['mean_detected_probability']:.4f}
        
        Configuration:
        • Threshold Used:              {stats['threshold_used']:.4f}
        """
        
        ax.text(
            0.1, 0.9, 
            summary_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        )
        
        plt.tight_layout()
        return fig


# Example usage
if __name__ == "__main__":
    print("Visualizer modules loaded successfully!")
    
    # Create dummy data for testing
    audio = np.random.randn(88200)
    spectrogram = np.random.randn(100, 800)
    
    # Test visualizers
    audio_viz = AudioVisualizer()
    results_viz = ResultsVisualizer()
    
    print("AudioVisualizer and ResultsVisualizer ready!")