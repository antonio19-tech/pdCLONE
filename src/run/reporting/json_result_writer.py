import json
from pathlib import Path
from typing import Any

from src.run.reporting.base_result_writer import BaseResultWriter


class JsonResultWriter(BaseResultWriter):
    """Writes fold metrics and experiment summaries as JSON files."""

    def save_fold_result(self, output_dir: Path, fold_result: dict[str, Any]) -> None:
        fold = fold_result["fold"]
        path = output_dir / f"fold_{fold}_metrics.json"
        path.write_text(json.dumps(fold_result, indent=2), encoding="utf-8")

    def save_experiment_summary(self, output_dir: Path, summary: dict[str, Any]) -> None:
        path = output_dir / "summary.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
