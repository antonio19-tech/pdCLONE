import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.metrics.base_metric import BaseMetric


class AUC(BaseMetric):
    """Binary AUC score computed from accumulated batch predictions."""

    def __init__(self) -> None:
        self.reset()

    def update(self, outputs: torch.Tensor, labels: torch.Tensor) -> None:
        positive_scores = torch.softmax(outputs.detach(), dim=1)[:, 1]
        self._predictions.append(positive_scores.cpu())
        self._labels.append(labels.detach().cpu())

    def compute(self) -> float:
        if not self._predictions:
            return 0.0
        predictions = torch.cat(self._predictions)
        labels = torch.cat(self._labels)
        return self._compute_auc(predictions, labels)

    def reset(self) -> None:
        self._predictions = []
        self._labels = []

    @property
    def name(self) -> str:
        return "auc"

    @staticmethod
    def _compute_auc(
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        predictions_np = predictions.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        if np.unique(labels_np).size < 2:
            return float("nan")

        return float(
            roc_auc_score(
                labels_np,
                predictions_np,
            )
        )
