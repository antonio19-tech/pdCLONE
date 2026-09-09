from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from src.preprocess.timeseries.timeseries_dataset import TimeSeriesDataset

@dataclass(frozen=True)
class TimeSeriesDataConfig:
    """Configuration for building the validation dataset."""

    data_root: Path
    metadata_path: Path

    patient_ids: Sequence[str] | None = None
    tasks: Sequence[int] | None = None

    feature_names: Sequence[str] = ("y", "x")
    engineered_features: Sequence[str] = ()

    cache_files: bool = True

    label_column: str = "Disease"
    positive_value: str = "PD"


def build_timeseries_dataset(
    cfg: TimeSeriesDataConfig,
) -> TimeSeriesDataset:
    """Build the full patient-level validation dataset."""

    return TimeSeriesDataset(
        data_root=cfg.data_root,
        metadata_path=cfg.metadata_path,

        patient_ids=(
            [str(p).zfill(5) for p in cfg.patient_ids]
            if cfg.patient_ids
            else None
        ),

        tasks=list(cfg.tasks) if cfg.tasks else None,

        feature_names=tuple(cfg.feature_names),
        engineered_features=tuple(cfg.engineered_features),

        cache_files=cfg.cache_files,

        label_column=cfg.label_column,
        positive_value=cfg.positive_value,
    )
