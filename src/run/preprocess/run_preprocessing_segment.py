from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.preprocess.segment.segment_dataset import SegmentDataset


@dataclass(frozen=True)
class SegmentDataConfig:
    """Configuration for building the segment dataset."""

    data_root: Path
    metadata_path: Path

    patient_ids: Sequence[str] | None = None
    tasks: Sequence[int] | None = None

    feature_names: Sequence[str] = ("y", "x")
    engineered_features: Sequence[str] = ()

    window_size: int = 128
    stride: int | None = None

    include_last: bool = True
    cache_files: bool = True

    label_column: str = "Disease"
    positive_value: str = "PD"


def build_segment_dataset(cfg: SegmentDataConfig) -> SegmentDataset:
    """Build the full segment-level dataset."""
    return SegmentDataset(
        data_root=cfg.data_root,
        metadata_path=cfg.metadata_path,
        patient_ids=[str(p).zfill(5) for p in cfg.patient_ids] if cfg.patient_ids else None,
        tasks=list(cfg.tasks) if cfg.tasks else None,
        feature_names=tuple(cfg.feature_names),
        engineered_features=tuple(cfg.engineered_features),
        window_size=cfg.window_size,
        stride=cfg.stride,
        include_last=cfg.include_last,
        cache_files=cfg.cache_files,
        label_column=cfg.label_column,
        positive_value=cfg.positive_value,
    )

