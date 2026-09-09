import numpy as np
import torch
from torch.utils.data import Dataset

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.data.pahaw_io import (
    build_recording_index,
    discover_svc_files,
    load_metadata,
    load_svc_file,
    select_features,
)
from src.preprocess.segment.segmenter import (
    pad_segment,
    segment_bounds,
)

__all__ = [
    "SegmentDataset",
    "SegmentIndex",
    "build_segment_index",
    "discover_svc_files",
    "load_metadata",
]


@dataclass(frozen=True, slots=True)
class SegmentIndex:
    """Metadata for a single segment extracted from a PaHaW task."""

    patient_id: str
    task_id: int
    file_path: Path
    label: int
    start: int
    end: int
    n_samples: int

def build_segment_index(
    data_root: str | Path,
    metadata_path: str | Path,
    patient_ids: Sequence[str] | None = None,
    tasks: Sequence[int] | None = None,
    window_size: int = 128,
    stride: int | None = None,
    include_last: bool = True,
    label_column: str = "Disease",
    positive_value: str = "PD",
) -> list[SegmentIndex]:
    """Create an index of all segments without loading them into memory."""
    recordings = build_recording_index(
        data_root=data_root,
        metadata_path=metadata_path,
        patient_ids=patient_ids,
        tasks=tasks,
        label_column=label_column,
        positive_value=positive_value,
        skip_empty=True,
    )

    segments: list[SegmentIndex] = []
    for recording in recordings:
        for bounds in segment_bounds(
            n_samples=recording.n_samples,
            window_size=window_size,
            stride=stride,
            include_last=include_last,
        ):
            segments.append(
                SegmentIndex(
                    patient_id=recording.patient_id,
                    task_id=recording.task_id,
                    file_path=recording.file_path,
                    label=recording.label,
                    start=bounds.start,
                    end=bounds.end,
                    n_samples=recording.n_samples,
                )
            )

    return segments


class SegmentDataset(Dataset):
    """Dataset of fixed-size segments extracted from PaHaW handwriting tasks.

    Each sample represents one segment and inherits the label of the patient
    from which it was extracted.
    """

    def __init__(
        self,
        data_root: str | Path,
        metadata_path: str | Path,
        patient_ids: Sequence[str] | None = None,
        tasks: Sequence[int] | None = None,
        feature_names: Sequence[str] = ("y",),
        engineered_features: Sequence[str] = (),
        window_size: int = 128,
        stride: int | None = None,
        include_last: bool = True,
        label_column: str = "Disease",
        positive_value: str = "PD",
        cache_files: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.metadata_path = Path(metadata_path)
        self.metadata = load_metadata(self.metadata_path)
        self.patient_ids = [str(pid).zfill(5) for pid in patient_ids] if patient_ids is not None else None
        self.tasks = list(tasks) if tasks is not None else None
        self.feature_names = tuple(feature_names)
        self.engineered_features = tuple(engineered_features)
        self.window_size = int(window_size)
        self.stride = self.window_size if stride is None else int(stride)
        self.include_last = bool(include_last)
        self.label_column = label_column
        self.positive_value = positive_value
        self.cache_files = bool(cache_files)


        self._segments = build_segment_index(
            data_root=self.data_root,
            metadata_path=self.metadata_path,
            patient_ids=self.patient_ids,
            tasks=self.tasks,
            window_size=self.window_size,
            stride=self.stride,
            include_last=self.include_last,
            label_column=self.label_column,
            positive_value=self.positive_value,
        )
        self._raw_cache: dict[Path, np.ndarray] = {}
        self._feature_cache: dict[Path, np.ndarray] = {}

    def subset_by_patients(self, patient_ids: Sequence[str]) -> "SegmentDataset":
        """Return a lightweight view with the same configuration but filtered patients."""
        patient_set = {str(pid).zfill(5) for pid in patient_ids}
        subset = object.__new__(SegmentDataset)
        subset.data_root = self.data_root
        subset.metadata_path = self.metadata_path
        subset.metadata = self.metadata
        subset.patient_ids = sorted(patient_set)
        subset.tasks = self.tasks
        subset.feature_names = self.feature_names
        subset.engineered_features = self.engineered_features
        subset.window_size = self.window_size
        subset.stride = self.stride
        subset.include_last = self.include_last
        subset.label_column = self.label_column
        subset.positive_value = self.positive_value
        subset.cache_files = self.cache_files
        subset._segments = [seg for seg in self._segments if seg.patient_id in patient_set]
        subset._raw_cache = {}
        subset._feature_cache = {}
        return subset

    def patient_labels(self) -> dict[str, int]:
        """Return one label per patient contained in the dataset."""
        labels: dict[str, int] = {}
        for segment in self._segments:
            labels.setdefault(segment.patient_id, segment.label)
        return labels

    def unique_patient_ids(self) -> list[str]:
        return sorted(self.patient_labels().keys())

    def task_ids(self) -> list[int]:
        return sorted({segment.task_id for segment in self._segments})

    def _load_raw(self, file_path: Path) -> np.ndarray:
        if self.cache_files and file_path in self._raw_cache:
            return self._raw_cache[file_path]

        raw = load_svc_file(str(file_path))
        if self.cache_files:
            self._raw_cache[file_path] = raw
        return raw

    def _load_features(self, file_path: Path) -> np.ndarray:
        if self.cache_files and file_path in self._feature_cache:
            return self._feature_cache[file_path]

        raw = self._load_raw(file_path)
        feature_columns = list(self.feature_names) + list(self.engineered_features)
        features = select_features(raw, feature_columns)

        if self.cache_files:
            self._feature_cache[file_path] = features
        return features

    def __len__(self) -> int:
        return len(self._segments)

    def __getitem__(self, idx: int) -> dict[str, object]:
        segment_meta = self._segments[idx]
        features = self._load_features(segment_meta.file_path)
        segment = features[segment_meta.start : segment_meta.end]
        segment, mask = pad_segment(segment, self.window_size)

        sample = {
            "segment": torch.from_numpy(np.asarray(segment, dtype=np.float32)).float(),
            "mask": torch.from_numpy(mask).float(),
            "label": torch.tensor(segment_meta.label, dtype=torch.long),
            "patient_id": segment_meta.patient_id,
            "task_id": torch.tensor(segment_meta.task_id, dtype=torch.long),
            "start": torch.tensor(segment_meta.start, dtype=torch.long),
            "end": torch.tensor(segment_meta.end, dtype=torch.long),
        }
        return sample

    def materialize(self) -> dict[str, object]:
        """Load the whole dataset into NumPy arrays."""

        X: list[np.ndarray] = []
        y: list[int] = []
        patient_ids: list[str] = []
        task_ids: list[int] = []
        starts: list[int] = []
        ends: list[int] = []

        for segment_meta in self._segments:
            features = self._load_features(segment_meta.file_path)
            segment = features[segment_meta.start : segment_meta.end]
            segment = pad_segment(segment, self.window_size)

            X.append(np.asarray(segment, dtype=np.float32))
            y.append(segment_meta.label)
            patient_ids.append(segment_meta.patient_id)
            task_ids.append(segment_meta.task_id)
            starts.append(segment_meta.start)
            ends.append(segment_meta.end)

        X_array = np.asarray(X, dtype=np.float32)
        y_array = np.asarray(y, dtype=np.int64)

        return {
            "X": X_array,
            "y": y_array,
            "patient_ids": patient_ids,
            "task_ids": np.asarray(task_ids, dtype=np.int64),
            "starts": np.asarray(starts, dtype=np.int64),
            "ends": np.asarray(ends, dtype=np.int64),
        }
