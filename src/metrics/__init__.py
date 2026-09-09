from src.metrics.accuracy import Accuracy
from src.metrics.auc import AUC
from src.metrics.base_metric import BaseMetric
from src.metrics.f1_score import F1Score
from src.metrics.precision import Precision
from src.metrics.recall import Recall
from src.metrics.specificity import Specificity

METRIC_REGISTRY: dict[str, type] = {
    "accuracy": Accuracy,
    "f1": F1Score,
    "auc": AUC,
    "precision": Precision,
    "recall": Recall,
    "specificity": Specificity,
}

def build_metric(name: str) -> "BaseMetric":
    if name not in METRIC_REGISTRY:
        raise ValueError(
            f"Unknown metric '{name}'. "
            f"Available: {list(METRIC_REGISTRY.keys())}"
        )
    return METRIC_REGISTRY[name]()
