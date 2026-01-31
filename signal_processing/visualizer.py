"""
Signal Visualization Module
Create plots for signal processing analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from typing import Tuple


class SignalVisualizer:
    """
    Visualize signal processing results
    """
    
    @staticmethod
    def plot_waveform_with_peaks(
        audio: np.ndarray, 
        sample_rate: int, 
        peaks: np.ndarray, 
        envelope: np.ndarray
    ) -> plt.Figure:
        """
        Plot waveform with detected peaks
        
        Args:
            audio: Audio waveform
            sample_rate: Sample rate
            peaks: Peak indices
            envelope: Signal envelope
            
        Returns:
            Matplotlib figure
        """
        time = np.arange(len(audio)) / sample_rate
        peak_times = peaks / sample_rate
        
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Plot raw waveform
        ax.plot(time, audio, color='orange', alpha=0.6, linewidth=0.5, label='Raw')
        
        # Plot envelope
        ax.plot(time, envelope, color='red', linewidth=1.5, label='Envelope (filtered)')
        ax.plot(time, -envelope, color='red', linewidth=1.5, alpha=0.5)
        
        # Plot detected peaks
        ax.scatter(peak_times, envelope[peaks], color='green', s=100, marker='x', 
                  linewidths=2, label='Detected events', zorder=5)
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.set_title('Waveform with Detected Bowel Sound Events', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_spectrogram(audio: np.ndarray, sample_rate: int, max_freq: int = 2000) -> plt.Figure:
        """
        Plot spectrogram
        
        Args:
            audio: Audio waveform
            sample_rate: Sample rate
            max_freq: Maximum frequency to display
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Compute spectrogram
        f, t, Sxx = signal.spectrogram(audio, sample_rate, nperseg=1024, noverlap=512)
        
        # Focus on 0-max_freq Hz
        freq_mask = f <= max_freq
        
        # Plot
        im = ax.pcolormesh(t, f[freq_mask], 10 * np.log10(Sxx[freq_mask, :] + 1e-10), 
                          shading='gouraud', cmap='viridis')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Frequency (Hz)', fontsize=12)
        ax.set_title(f'Spectrogram (0-{max_freq} Hz)', fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)', fontsize=11)
        
        # Highlight bowel sound range
        ax.axhspan(200, 800, alpha=0.1, color='red', label='Bowel sound range (200-800 Hz)')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_iei_histogram(peaks: np.ndarray, sample_rate: int) -> plt.Figure:
        """
        Plot inter-event interval histogram
        
        Args:
            peaks: Peak indices
            sample_rate: Sample rate
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        
        if len(peaks) < 2:
            ax.text(0.5, 0.5, 'Insufficient events for histogram', 
                   ha='center', va='center', fontsize=14)
            ax.set_title('Inter-Event Interval Histogram', fontsize=14, fontweight='bold')
            return fig
        
        iei = np.diff(peaks) / sample_rate * 1000  # milliseconds
        
        ax.hist(iei, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        
        ax.set_xlabel('Inter-Event Interval (ms)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Inter-Event Interval Histogram', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add mean line
        mean_iei = np.mean(iei)
        ax.axvline(mean_iei, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_iei:.1f} ms')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_poincare(peaks: np.ndarray, sample_rate: int) -> plt.Figure:
        """
        Plot Poincaré plot
        
        Args:
            peaks: Peak indices
            sample_rate: Sample rate
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        if len(peaks) < 3:
            ax.text(0.5, 0.5, 'Insufficient events for Poincaré plot', 
                   ha='center', va='center', fontsize=14)
            ax.set_title('Poincaré Plot', fontsize=14, fontweight='bold')
            return fig
        
        iei = np.diff(peaks) / sample_rate * 1000  # milliseconds
        
        # Plot IEI(n) vs IEI(n+1)
        ax.scatter(iei[:-1], iei[1:], s=80, alpha=0.6, color='steelblue', edgecolors='black')
        
        # Add identity line
        max_iei = max(iei)
        min_iei = min(iei)
        ax.plot([min_iei, max_iei], [min_iei, max_iei], 'r--', linewidth=2, 
               alpha=0.5, label='Identity line')
        
        ax.set_xlabel('IEI_(n) (ms)', fontsize=12)
        ax.set_ylabel('IEI_(n+1) (ms)', fontsize=12)
        ax.set_title('Poincaré Plot', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        return fig
