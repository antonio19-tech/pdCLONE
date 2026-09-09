from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseResultWriter(ABC):
    """Persists experiment results without coupling the runner to a file format."""

    @abstractmethod
    def save_fold_result(self, output_dir: Path, fold_result: dict[str, Any]) -> None:
        """Persist the results collected for a single fold."""
        ...

    @abstractmethod
    def save_experiment_summary(self, output_dir: Path, summary: dict[str, Any]) -> None:
        """Persist the final summary for one experiment."""
        ...
