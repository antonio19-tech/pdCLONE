import torch

from src.metrics.base_metric import BaseMetric


class Specificity(BaseMetric):
    """Binary Specificity score computed from accumulated batch predictions."""

    def __init__(self) -> None:
        self.reset()
        
    def update(self, outputs: torch.Tensor, labels: torch.Tensor) -> None:
        predictions = outputs.argmax(dim=1)

        self._true_negatives += ((predictions == 0) & (labels == 0)).sum().item()
        self._false_positives += ((predictions == 1) & (labels == 0)).sum().item()
    
    def compute(self):
        if self._true_negatives + self._false_positives == 0:
            return 0.0
        return self._true_negatives / (self._true_negatives + self._false_positives)

    def reset(self) -> None:
        self._true_negatives = 0
        self._false_positives = 0
    
    @property
    def name(self) -> str:
        return "specificity"