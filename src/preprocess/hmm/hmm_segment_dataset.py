from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.pahaw_io import (
    build_recording_index,
    load_svc_file,
    select_features,
)

from src.preprocess.hmm.hmm_segmenter import (
    HMMTimeSeriesSegmenter,
    SegmentChunk,
)


class HMMSegmentDataset(Dataset):
    """
    Dataset based on HMM segmentation.

    Pipeline:
        raw sequence
            ↓
        HMM segmentation
            ↓
        variable-length segments
            ↓
        padding + mask
            ↓
        samples
    """

    def __init__(
        self,
        hmm_segmenter: HMMTimeSeriesSegmenter,
        data_root: str | Path,
        metadata_path: str | Path,
        patient_ids: Sequence[str] | None = None,
        tasks: Sequence[int] | None = None,
        feature_names: Sequence[str] = ("y",),
        engineered_features: Sequence[str] = (),
        pad_to: int | None = None,
        pad_value: float = 0.0,
        label_column: str = "Disease",
        positive_value: str = "PD",
        cache_files: bool = True,
    ) -> None:

        self.hmm_segmenter = hmm_segmenter

        self.data_root = Path(data_root)
        self.metadata_path = Path(metadata_path)

        self.feature_names = tuple(feature_names)
        self.engineered_features = tuple(engineered_features)

        self.pad_to = pad_to
        self.pad_value = pad_value

        self.cache_files = bool(cache_files)

        self._raw_cache: dict[Path, np.ndarray] = {}
        self._feature_cache: dict[Path, np.ndarray] = {}

        self._samples = []

        recordings = build_recording_index(
            data_root=self.data_root,
            metadata_path=self.metadata_path,
            patient_ids=patient_ids,
            tasks=tasks,
            label_column=label_column,
            positive_value=positive_value,
            skip_empty=True,
        )

        for recording in recordings:

            features = self._load_features(
                recording.file_path
            )

            chunks = self.hmm_segmenter.segment_sequence(
                sequence=features,
                pad_to=self.pad_to,
                pad_value=self.pad_value,
            )

            for chunk in chunks:

                self._samples.append(
                    {
                        "patient_id": recording.patient_id,
                        "task_id": recording.task_id,
                        "label": recording.label,
                        "chunk": chunk,
                    }
                )

    def _load_raw(
        self,
        file_path: Path,
    ) -> np.ndarray:

        if self.cache_files and file_path in self._raw_cache:
            return self._raw_cache[file_path]

        raw = load_svc_file(file_path)

        if self.cache_files:
            self._raw_cache[file_path] = raw

        return raw

    def _load_features(
        self,
        file_path: Path,
    ) -> np.ndarray:

        if self.cache_files and file_path in self._feature_cache:
            return self._feature_cache[file_path]

        raw = self._load_raw(file_path)

        feature_columns = (
            list(self.feature_names)
            + list(self.engineered_features)
        )

        features = select_features(
            raw,
            feature_columns,
        )

        if self.cache_files:
            self._feature_cache[file_path] = features

        return features

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(
        self,
        idx: int,
    ) -> dict[str, object]:

        sample = self._samples[idx]

        chunk: SegmentChunk = sample["chunk"]

        return {
            "segment": torch.from_numpy(
                chunk.segment
            ).float(),

            "mask": torch.from_numpy(
                chunk.mask
            ).float(),

            "state": torch.tensor(
                chunk.state,
                dtype=torch.long,
            ),

            "label": torch.tensor(
                sample["label"],
                dtype=torch.long,
            ),

            "patient_id": sample["patient_id"],

            "task_id": torch.tensor(
                sample["task_id"],
                dtype=torch.long,
            ),

            "start": torch.tensor(
                chunk.start,
                dtype=torch.long,
            ),

            "end": torch.tensor(
                chunk.end,
                dtype=torch.long,
            ),

            "length": torch.tensor(
                chunk.end - chunk.start,
                dtype=torch.long,
            ),
        }

    def subset_by_patients(
        self,
        patient_ids: Sequence[str],
    ) -> "HMMSegmentDataset":

        patient_set = {
            str(pid).zfill(5)
            for pid in patient_ids
        }

        subset = object.__new__(HMMSegmentDataset)

        subset.hmm_segmenter = self.hmm_segmenter

        subset.data_root = self.data_root
        subset.metadata_path = self.metadata_path

        subset.feature_names = self.feature_names
        subset.engineered_features = self.engineered_features

        subset.pad_to = self.pad_to
        subset.pad_value = self.pad_value

        subset.cache_files = self.cache_files

        subset._raw_cache = {}
        subset._feature_cache = {}

        subset._samples = [
            sample
            for sample in self._samples
            if sample["patient_id"] in patient_set
        ]

        return subset

    def patient_labels(
        self,
    ) -> dict[str, int]:

        labels = {}

        for sample in self._samples:

            pid = sample["patient_id"]

            if pid not in labels:
                labels[pid] = sample["label"]

        return labels

    def unique_patient_ids(
        self,
    ) -> list[str]:

        return sorted(
            self.patient_labels().keys()
        )

    def state_distribution(
        self,
    ) -> dict[int, int]:

        counts = {}

        for sample in self._samples:

            state = int(
                sample["chunk"].state
            )

            counts[state] = (
                counts.get(state, 0) + 1
            )

        return counts
