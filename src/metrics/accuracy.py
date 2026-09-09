import torch
from src.metrics.base_metric import BaseMetric


class Accuracy(BaseMetric):

    def __init__(self) -> None:
        self._correct = 0
        self._total   = 0

    def update(self, outputs: torch.Tensor, labels: torch.Tensor) -> None:
        self._correct += (outputs.argmax(dim=1) == labels).sum().item()
        self._total   += labels.size(0)

    def compute(self) -> float:
        if self._total == 0:
            return 0.0
        return self._correct / self._total

    def reset(self) -> None:
        self._correct = 0
        self._total   = 0

    @property
    def name(self) -> str:
        return "accuracy"