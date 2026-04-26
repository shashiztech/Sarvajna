"""
Evaluation metrics for multimodal models.
"""

import torch
import numpy as np
from typing import List, Dict, Any
from collections import Counter
import math


class TextMetrics:
    """Evaluation metrics for text generation tasks."""
    
    @staticmethod
    def perplexity(loss: float) -> float:
        """
        Calculate perplexity from cross-entropy loss.
        
        Args:
            loss: Cross-entropy loss
            
        Returns:
            Perplexity score
        """
        return math.exp(loss)
    
    @staticmethod
    def exact_match(predictions: List[str], references: List[str]) -> float:
        """
        Calculate exact match score.
        
        Args:
            predictions: List of predicted strings
            references: List of reference strings
            
        Returns:
            Exact match score (0-1)
        """
        assert len(predictions) == len(references)
        matches = sum(pred.strip() == ref.strip() for pred, ref in zip(predictions, references))
        return matches / len(predictions)
    
    @staticmethod
    def token_f1(predictions: List[str], references: List[str]) -> float:
        """
        Calculate token-level F1 score.
        
        Args:
            predictions: List of predicted strings
            references: List of reference strings
            
        Returns:
            Average F1 score
        """
        f1_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            
            if len(pred_tokens) == 0 or len(ref_tokens) == 0:
                f1_scores.append(0.0)
                continue
            
            # Calculate precision and recall
            common = Counter(pred_tokens) & Counter(ref_tokens)
            num_common = sum(common.values())
            
            if num_common == 0:
                f1_scores.append(0.0)
            else:
                precision = num_common / len(pred_tokens)
                recall = num_common / len(ref_tokens)
                f1 = 2 * (precision * recall) / (precision + recall)
                f1_scores.append(f1)
        
        return sum(f1_scores) / len(f1_scores)
    
    @staticmethod
    def bleu_score(predictions: List[str], references: List[str], n: int = 4) -> float:
        """
        Calculate BLEU score (simplified implementation).
        
        Args:
            predictions: List of predicted strings
            references: List of reference strings
            n: Maximum n-gram size
            
        Returns:
            BLEU score
        """
        from collections import defaultdict
        
        bleu_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            
            if len(pred_tokens) == 0:
                bleu_scores.append(0.0)
                continue
            
            # Calculate n-gram precisions
            precisions = []
            for i in range(1, n + 1):
                pred_ngrams = Counter([tuple(pred_tokens[j:j+i]) for j in range(len(pred_tokens) - i + 1)])
                ref_ngrams = Counter([tuple(ref_tokens[j:j+i]) for j in range(len(ref_tokens) - i + 1)])
                
                matches = sum((pred_ngrams & ref_ngrams).values())
                total = sum(pred_ngrams.values())
                
                if total > 0:
                    precisions.append(matches / total)
                else:
                    precisions.append(0.0)
            
            # Geometric mean of precisions
            if all(p > 0 for p in precisions):
                log_precisions = [math.log(p) for p in precisions]
                geo_mean = math.exp(sum(log_precisions) / len(log_precisions))
                
                # Brevity penalty
                bp = 1.0 if len(pred_tokens) >= len(ref_tokens) else math.exp(1 - len(ref_tokens) / len(pred_tokens))
                
                bleu_scores.append(bp * geo_mean)
            else:
                bleu_scores.append(0.0)
        
        return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0


class VisionMetrics:
    """Evaluation metrics for vision tasks."""
    
    @staticmethod
    def mean_squared_error(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate MSE."""
        return torch.mean((predictions - targets) ** 2).item()
    
    @staticmethod
    def peak_signal_noise_ratio(predictions: torch.Tensor, targets: torch.Tensor, max_val: float = 1.0) -> float:
        """Calculate PSNR."""
        mse = torch.mean((predictions - targets) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse)).item()


def compute_metrics(
    task: str,
    predictions: Any,
    references: Any,
    **kwargs,
) -> Dict[str, float]:
    """
    Compute metrics for a given task.
    
    Args:
        task: Task type (text, image, video, audio)
        predictions: Model predictions
        references: Ground truth references
        **kwargs: Additional arguments
        
    Returns:
        Dictionary of metrics
    """
    if task == "text":
        metrics = {}
        
        if "loss" in kwargs:
            metrics["perplexity"] = TextMetrics.perplexity(kwargs["loss"])
        
        if isinstance(predictions, list) and isinstance(references, list):
            metrics["exact_match"] = TextMetrics.exact_match(predictions, references)
            metrics["token_f1"] = TextMetrics.token_f1(predictions, references)
            metrics["bleu"] = TextMetrics.bleu_score(predictions, references)
        
        return metrics
    
    elif task == "image":
        metrics = {}
        
        if torch.is_tensor(predictions) and torch.is_tensor(references):
            metrics["mse"] = VisionMetrics.mean_squared_error(predictions, references)
            metrics["psnr"] = VisionMetrics.peak_signal_noise_ratio(predictions, references)
        
        return metrics
    
    else:
        raise ValueError(f"Unknown task: {task}")
