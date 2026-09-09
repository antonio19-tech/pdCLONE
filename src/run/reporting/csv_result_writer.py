import csv
from pathlib import Path
from typing import Any

from src.run.reporting.base_result_writer import BaseResultWriter


class CsvResultWriter(BaseResultWriter):
    """Writes fold histories and experiment summaries as CSV files."""

    def save_fold_result(self, output_dir: Path, fold_result: dict[str, Any]) -> None:
        fold = fold_result["fold"]
        history_path = output_dir / f"fold_{fold}_history.csv"
        best_metrics_path = output_dir / f"fold_{fold}_best_metrics.csv"

        self._write_rows(history_path, fold_result["history"])
        self._write_rows(
            best_metrics_path,
            [{key: value for key, value in fold_result.items() if key != "history"}],
        )

    def save_experiment_summary(self, output_dir: Path, summary: dict[str, Any]) -> None:
        fold_summary_path = output_dir / "fold_summary.csv"
        averages_path = output_dir / "summary.csv"

        fold_rows = [
            {key: value for key, value in fold_result.items() if key != "history"}
            for fold_result in summary["folds"]
        ]
        average_row = {
            "experiment": summary["experiment"],
            **summary["averages"],
        }

        self._write_rows(fold_summary_path, fold_rows)
        self._write_rows(averages_path, [average_row])

    @staticmethod
    def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return

        fieldnames = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
