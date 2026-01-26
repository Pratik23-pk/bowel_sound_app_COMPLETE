"""
Evaluation Module
Calculates performance metrics for predictions
"""

import numpy as np
from typing import Dict, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)


class MetricsCalculator:
    """
    Calculates evaluation metrics for predictions.
    """
    
    def __init__(self):
        pass
    
    def calculate_basic_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Calculate basic classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            
        Returns:
            Dictionary with metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'sensitivity': recall_score(y_true, y_pred, zero_division=0),  # Same as recall
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        # Calculate specificity from confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['specificity'] = specificity
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
        
        # Add ROC-AUC if probabilities provided
        if y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            except:
                metrics['roc_auc'] = None
        
        return metrics
    
    def calculate_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """
        Calculate confusion matrix and derived metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with confusion matrix info
        """
        cm = confusion_matrix(y_true, y_pred)
        
        result = {
            'confusion_matrix': cm,
            'matrix_shape': cm.shape
        }
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            result.update({
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'total_negative': int(tn + fp),
                'total_positive': int(fn + tp)
            })
        
        return result
    
    def calculate_detection_stats(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        duration_seconds: float = 2.0
    ) -> Dict:
        """
        Calculate detection statistics without true labels.
        
        Args:
            predictions: Binary predictions
            probabilities: Prediction probabilities
            duration_seconds: Duration of audio analyzed
            
        Returns:
            Dictionary with detection statistics
        """
        n_total = len(predictions)
        n_detected = np.sum(predictions == 1)
        n_noise = n_total - n_detected
        
        # Percentages
        detection_percentage = (n_detected / n_total) * 100 if n_total > 0 else 0
        noise_percentage = (n_noise / n_total) * 100 if n_total > 0 else 0
        
        # Rate per minute
        bowel_sounds_per_minute = (n_detected / duration_seconds) * 60
        
        # Probability statistics
        detected_probs = probabilities[predictions == 1]
        noise_probs = probabilities[predictions == 0]
        
        stats = {
            'n_total_frames': n_total,
            'n_detected': n_detected,
            'n_noise': n_noise,
            'detection_percentage': detection_percentage,
            'noise_percentage': noise_percentage,
            'bowel_sounds_per_minute': bowel_sounds_per_minute,
            
            # Overall probability stats
            'mean_probability': float(np.mean(probabilities)),
            'median_probability': float(np.median(probabilities)),
            'std_probability': float(np.std(probabilities)),
            'max_probability': float(np.max(probabilities)),
            'min_probability': float(np.min(probabilities)),
            
            # Detected frames probability stats
            'mean_detected_probability': float(np.mean(detected_probs)) if len(detected_probs) > 0 else 0,
            'median_detected_probability': float(np.median(detected_probs)) if len(detected_probs) > 0 else 0,
            
            # Noise frames probability stats
            'mean_noise_probability': float(np.mean(noise_probs)) if len(noise_probs) > 0 else 0,
            'median_noise_probability': float(np.median(noise_probs)) if len(noise_probs) > 0 else 0
        }
        
        return stats
    
    def format_metrics_for_display(self, metrics: Dict) -> str:
        """
        Format metrics dictionary for human-readable display.
        
        Args:
            metrics: Metrics dictionary
            
        Returns:
            Formatted string
        """
        lines = []
        
        if 'accuracy' in metrics:
            lines.append(f"Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        if 'precision' in metrics:
            lines.append(f"Precision:   {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        if 'sensitivity' in metrics or 'recall' in metrics:
            sens = metrics.get('sensitivity', metrics.get('recall', 0))
            lines.append(f"Sensitivity: {sens:.4f} ({sens*100:.2f}%)")
        if 'specificity' in metrics:
            lines.append(f"Specificity: {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
        if 'f1_score' in metrics:
            lines.append(f"F1-Score:    {metrics['f1_score']:.4f}")
        if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
            lines.append(f"ROC-AUC:     {metrics['roc_auc']:.4f}")
        
        return "\n".join(lines)
    
    def compare_with_paper_results(self, metrics: Dict, model_type: str = 'crnn') -> Dict:
        """
        Compare achieved metrics with paper results.
        
        Args:
            metrics: Calculated metrics
            model_type: 'crnn' or 'cdnn'
            
        Returns:
            Dictionary with comparison
        """
        # Paper results (from Ficek et al., 2021)
        paper_results = {
            'crnn': {
                'accuracy': 0.977,
                'precision': 0.827,
                'sensitivity': 0.773,
                'specificity': 0.990
            },
            'cdnn': {
                'accuracy': 0.974,
                'precision': 0.833,
                'sensitivity': 0.711,
                'specificity': 0.991
            }
        }
        
        if model_type.lower() not in paper_results:
            return {}
        
        paper = paper_results[model_type.lower()]
        comparison = {}
        
        for metric in ['accuracy', 'precision', 'sensitivity', 'specificity']:
            if metric in metrics and metric in paper:
                diff = metrics[metric] - paper[metric]
                diff_pct = (diff / paper[metric]) * 100 if paper[metric] > 0 else 0
                
                comparison[metric] = {
                    'yours': metrics[metric],
                    'paper': paper[metric],
                    'difference': diff,
                    'difference_pct': diff_pct
                }
        
        return comparison


# Example usage
if __name__ == "__main__":
    print("MetricsCalculator module loaded successfully!")
    
    # Create dummy data for testing
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_prob = np.random.rand(100)
    
    calc = MetricsCalculator()
    
    # Test basic metrics
    metrics = calc.calculate_basic_metrics(y_true, y_pred, y_prob)
    print("\nBasic Metrics:")
    print(calc.format_metrics_for_display(metrics))
    
    # Test detection stats (no true labels)
    predictions = np.random.randint(0, 2, 100)
    probabilities = np.random.rand(100)
    
    stats = calc.calculate_detection_stats(predictions, probabilities)
    print(f"\nDetection Stats:")
    print(f"Detected: {stats['n_detected']}/{stats['n_total_frames']}")
    print(f"Percentage: {stats['detection_percentage']:.2f}%")