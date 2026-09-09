import torch

from src.metrics.base_metric import BaseMetric


class Recall(BaseMetric):
    """Binary Recall score computed from accumulated batch predictions."""

    def __init__(self) -> None:
        self.reset()

    def update(self, outputs: torch.Tensor, labels: torch.Tensor) -> None:
        predictions = outputs.argmax(dim=1)

        self._true_positives += ((predictions == 1) & (labels == 1)).sum().item()
        self._false_negatives += ((predictions == 0) & (labels == 1)).sum().item()

    def compute(self):
        if self._true_positives + self._false_negatives == 0:
            return 0.0
        return self._true_positives / (self._true_positives + self._false_negatives)

    def reset(self) -> None:
        self._true_positives = 0
        self._false_negatives = 0

    @property
    def name(self) -> str:
        return "recall"
