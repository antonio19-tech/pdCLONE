from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.utils.data import Dataset


class PaHaWSequenceDataset(Dataset):
    """
    Reads raw PaHaW trajectories as variable-length multivariate sequences.

    Each item is a tensor with shape [timesteps, features], plus the binary label
    and the original sequence length. A collate function is provided to pad a
    batch and build a boolean mask for valid timesteps.
    """

    def __init__(
        self,
        sequences: list[torch.Tensor],
        labels: list[int],
        feature_names: list[str],
        normalization_stats: dict[str, list[float]] | None = None,
    ) -> None:
        self._sequences = sequences
        self._labels = labels
        self._feature_names = feature_names
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

        if normalization_stats is not None:
            self._mean = torch.tensor(normalization_stats["mean"], dtype=torch.float32)
            self._std = torch.tensor(normalization_stats["std"], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        sequence = self._sequences[idx].clone()
        if self._mean is not None and self._std is not None:
            sequence = (sequence - self._mean) / self._std

        label = self._labels[idx]
        length = sequence.size(0)
        return sequence, label, length

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    @staticmethod
    def collate_fn(
        batch: list[tuple[torch.Tensor, int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sequences, labels, lengths = zip(*batch, strict=True)

        max_length = max(lengths)
        batch_size = len(sequences)
        num_features = sequences[0].size(-1)

        padded_sequences = torch.zeros(
            batch_size,
            max_length,
            num_features,
            dtype=sequences[0].dtype,
        )
        mask = torch.zeros(batch_size, max_length, dtype=torch.bool)

        for index, sequence in enumerate(sequences):
            length = sequence.size(0)
            padded_sequences[index, :length] = sequence
            mask[index, :length] = True

        return (
            padded_sequences,
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(lengths, dtype=torch.long),
            mask,
        )

    @classmethod
    def from_patient_ids(
        cls,
        patient_ids: list[str],
        pahaw_dataset,
        feature_names: list[str],
        disease_column: str = "Disease",
        normalization_stats: dict[str, list[float]] | None = None,
        task_indices: Iterable[int] | None = None,
        downsample_factor: int = 1,
        max_sequence_length: int | None = None,
    ) -> "PaHaWSequenceDataset":
        sequences: list[torch.Tensor] = []
        labels: list[int] = []

        for patient_id in patient_ids:
            disease = pahaw_dataset.df.loc[int(patient_id), disease_column]
            label = 1 if disease == "PD" else 0

            patient_samples = pahaw_dataset.mapping[patient_id]
            selected_task_indices = cls._resolve_task_indices(
                num_tasks=len(patient_samples),
                task_indices=task_indices,
            )

            for task_idx in selected_task_indices:
                sample = patient_samples[task_idx]
                sequences.append(
                    cls._build_sequence(
                        sample=sample,
                        feature_names=feature_names,
                        downsample_factor=downsample_factor,
                        max_sequence_length=max_sequence_length,
                    )
                )
                labels.append(label)

        return cls(
            sequences=sequences,
            labels=labels,
            feature_names=feature_names,
            normalization_stats=normalization_stats,
        )

    @staticmethod
    def _resolve_task_indices(
        num_tasks: int,
        task_indices: Iterable[int] | None,
    ) -> list[int]:
        if task_indices is None:
            return list(range(num_tasks))

        resolved = sorted(set(int(task_idx) for task_idx in task_indices))
        invalid = [task_idx for task_idx in resolved if task_idx < 0 or task_idx >= num_tasks]
        if invalid:
            raise ValueError(
                f"Invalid task indices {invalid}. Expected values in [0, {num_tasks - 1}]."
            )
        return resolved

    @staticmethod
    def _build_sequence(
        sample: dict,
        feature_names: list[str],
        downsample_factor: int = 1,
        max_sequence_length: int | None = None,
    ) -> torch.Tensor:
        missing_features = [name for name in feature_names if name not in sample]
        if missing_features:
            raise ValueError(f"Sample is missing features: {missing_features}")

        if downsample_factor < 1:
            raise ValueError("downsample_factor must be >= 1.")
        if max_sequence_length is not None and max_sequence_length < 1:
            raise ValueError("max_sequence_length must be >= 1 when provided.")

        columns = [sample[name].to(dtype=torch.float32) for name in feature_names]
        lengths = {column.size(0) for column in columns}
        if len(lengths) != 1:
            raise ValueError(
                "All selected PaHaW features must share the same number of timesteps."
            )

        sequence = torch.stack(columns, dim=-1)

        if downsample_factor > 1:
            sequence = sequence[::downsample_factor]

        if max_sequence_length is not None and sequence.size(0) > max_sequence_length:
            indices = torch.linspace(
                0,
                sequence.size(0) - 1,
                steps=max_sequence_length,
                dtype=torch.float32,
            ).round().to(dtype=torch.long)
            sequence = sequence.index_select(0, indices)

        return sequence
