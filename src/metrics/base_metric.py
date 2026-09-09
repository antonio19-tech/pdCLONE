from abc import ABC, abstractmethod
import torch


class BaseMetric(ABC):
    """Interface that every metric must implement."""

    @abstractmethod
    def update(self, outputs: torch.Tensor, labels: torch.Tensor) -> None:
        """Accumulate predictions and labels for one batch."""
        ...

    @abstractmethod
    def compute(self) -> float:
        """Return the final scalar metric value."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state between epochs/folds."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name used in logs and summaries."""
        ...