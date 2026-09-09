from typing import Any

from src.run.reporting.base_result_writer import BaseResultWriter
from src.run.reporting.csv_result_writer import CsvResultWriter
from src.run.reporting.json_result_writer import JsonResultWriter

WRITER_REGISTRY: dict[str, type[BaseResultWriter]] = {
    "csv": CsvResultWriter,
    "json": JsonResultWriter,
}


def build_result_writer(training_cfg: dict[str, Any]) -> BaseResultWriter | None:
    if not training_cfg.get("save_metrics", True):
        return None

    writer_name = training_cfg.get("results_writer", "json")
    if writer_name not in WRITER_REGISTRY:
        raise ValueError(
            f"Unknown results writer '{writer_name}'. "
            f"Available: {list(WRITER_REGISTRY.keys())}"
        )
    return WRITER_REGISTRY[writer_name]()
