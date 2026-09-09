import torch

from src.metrics.base_metric import BaseMetric


class F1Score(BaseMetric):
    """Binary F1 score computed from accumulated batch predictions."""

    def __init__(self) -> None:
        self.reset()

    def update(self, outputs: torch.Tensor, labels: torch.Tensor) -> None:
        predictions = outputs.argmax(dim=1)

        self._true_positives += ((predictions == 1) & (labels == 1)).sum().item()
        self._false_positives += ((predictions == 1) & (labels == 0)).sum().item()
        self._false_negatives += ((predictions == 0) & (labels == 1)).sum().item()

    def compute(self) -> float:
        denominator = (2 * self._true_positives) + self._false_positives + self._false_negatives
        if denominator == 0:
            return 0.0
        return (2 * self._true_positives) / denominator

    def reset(self) -> None:
        self._true_positives = 0
        self._false_positives = 0
        self._false_negatives = 0

    @property
    def name(self) -> str:
        return "f1"
