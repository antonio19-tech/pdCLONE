from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from src.data.pahaw_io import (
    build_recording_index,
    load_metadata,
    load_svc_file,
    select_features,
)

@dataclass(frozen=True, slots=True)
class ValidationIndex:
    patient_id: str
    task_id: int
    file_path: Path
    label: int

class TimeSeriesDataset(Dataset):
    """
    Dataset of full patient sequences for validation/testing.

    Each sample corresponds to one full .svc recording:
    - no segmentation
    - no padding
    - variable-length sequence returned as-is
    """

    def __init__(
        self,
        data_root: str | Path,
        metadata_path: str | Path,
        patient_ids: Sequence[str] | None = None,
        tasks: Sequence[int] | None = None,
        feature_names: Sequence[str] = ("y",),
        engineered_features: Sequence[str] = (),
        label_column: str = "Disease",
        positive_value: str = "PD",
        cache_files: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.metadata_path = Path(metadata_path)
        self.metadata = load_metadata(self.metadata_path)

        self.patient_ids = (
            [str(pid).zfill(5) for pid in patient_ids]
            if patient_ids is not None
            else None
        )
        self.tasks = set(tasks) if tasks is not None else None

        self.feature_names = tuple(feature_names)
        self.engineered_features = tuple(engineered_features)
        self.label_column = label_column
        self.positive_value = positive_value
        self.cache_files = bool(cache_files)

        self._raw_cache: dict[Path, np.ndarray] = {}
        self._feature_cache: dict[Path, np.ndarray] = {}

        self._samples: list[ValidationIndex] = self._build_index()

    def _build_index(self) -> list[ValidationIndex]:
        recordings = build_recording_index(
            data_root=self.data_root,
            metadata_path=self.metadata_path,
            patient_ids=self.patient_ids,
            tasks=list(self.tasks) if self.tasks is not None else None,
            label_column=self.label_column,
            positive_value=self.positive_value,
            skip_empty=True,
        )

        return [
            ValidationIndex(
                patient_id=recording.patient_id,
                task_id=recording.task_id,
                file_path=recording.file_path,
                label=recording.label,
            )
            for recording in recordings
        ]

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
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample_meta = self._samples[idx]
        sequence = self._load_features(sample_meta.file_path)

        return {
            "sequence": torch.from_numpy(np.asarray(sequence, dtype=np.float32)).float(),
            "label": torch.tensor(sample_meta.label, dtype=torch.long),
            "patient_id": sample_meta.patient_id,
            "task_id": torch.tensor(sample_meta.task_id, dtype=torch.long),
            "file_path": str(sample_meta.file_path),
        }

    def unique_patient_ids(self) -> list[str]:
        return sorted({sample.patient_id for sample in self._samples})

    def patient_labels(self) -> dict[str, int]:
        labels: dict[str, int] = {}
        for sample in self._samples:
            labels.setdefault(sample.patient_id, sample.label)
        return labels

    def subset_by_patients(self, patient_ids: Sequence[str]) -> "TimeSeriesDataset":
        patient_set = {str(pid).zfill(5) for pid in patient_ids}

        subset = object.__new__(TimeSeriesDataset)
        subset.data_root = self.data_root
        subset.metadata_path = self.metadata_path
        subset.metadata = self.metadata
        subset.patient_ids = sorted(patient_set)
        subset.tasks = self.tasks
        subset.feature_names = self.feature_names
        subset.engineered_features = self.engineered_features
        subset.label_column = self.label_column
        subset.positive_value = self.positive_value
        subset.cache_files = self.cache_files
        subset._raw_cache = {}
        subset._feature_cache = {}
        subset._samples = [s for s in self._samples if s.patient_id in patient_set]
        return subset

    @staticmethod
    def pad_timeseries(batch):

        sequences = [
            sample["sequence"]
            for sample in batch
        ]

        labels = torch.stack([
            sample["label"]
            for sample in batch
        ])

        patient_ids = [
            sample["patient_id"]
            for sample in batch
        ]

        lengths = torch.tensor(
            [seq.shape[0] for seq in sequences],
            dtype=torch.long
        )

        padded = pad_sequence(
            sequences,
            batch_first=True,
            padding_value=0.0,
        )

        mask = (
            torch.arange(padded.size(1))[None, :]
            < lengths[:, None]
        )

        return {
            "sequence": padded,
            "mask": mask,
            "lengths": lengths,
            "label": labels,
            "patient_id": patient_ids,
        }
