from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.preprocess.hmm.hmm_segment_dataset import HMMSegmentDataset


@dataclass(frozen=True, slots=True)
class PatientSequenceSample:
    patient_id: str
    label: int
    segments: Tensor          # [S, T, F]
    segment_masks: Tensor     # [S, T]
    segment_states: Tensor     # [S]
    task_ids: Tensor          # [S]
    starts: Tensor            # [S]
    ends: Tensor              # [S]
    lengths: Tensor           # [S]


class HMMPatientDataset(Dataset):
    """
    Groups HMM segments by patient.

    Output per item:
        - segments:      [S, T, F]
        - segment_masks: [S, T]
        - segment_states:[S]
        - task_ids:      [S]
        - starts:        [S]
        - ends:          [S]
        - lengths:       [S]
        - label:         scalar
        - patient_id:    str

    Notes:
        - Segments are ordered by (task_id, start).
        - This is the right format for a sequence model like Mamba.
    """

    def __init__(
        self,
        segment_dataset: HMMSegmentDataset,
        sort_by_task_then_start: bool = True,
    ) -> None:
        self.segment_dataset = segment_dataset
        self.sort_by_task_then_start = sort_by_task_then_start

        self._patient_ids = sorted(
            {self._get_patient_id(i) for i in range(len(self.segment_dataset))}
        )
        self._patient_to_indices = self._build_patient_index()

    def _get_patient_id(self, idx: int) -> str:
        sample = self.segment_dataset[idx]
        return str(sample["patient_id"])

    def _build_patient_index(self) -> dict[str, list[int]]:
        mapping: dict[str, list[int]] = {}
        for idx in range(len(self.segment_dataset)):
            sample = self.segment_dataset[idx]
            pid = str(sample["patient_id"])
            mapping.setdefault(pid, []).append(idx)

        if self.sort_by_task_then_start:
            for pid, indices in mapping.items():
                indices.sort(
                    key=lambda i: (
                        int(self.segment_dataset[i]["task_id"]),
                        int(self.segment_dataset[i]["start"]),
                    )
                )
        else:
            for pid, indices in mapping.items():
                indices.sort(key=lambda i: int(self.segment_dataset[i]["start"]))

        return mapping

    def __len__(self) -> int:
        return len(self._patient_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pid = self._patient_ids[idx]
        indices = self._patient_to_indices[pid]

        samples = [self.segment_dataset[i] for i in indices]

        segments = torch.stack([s["segment"] for s in samples], dim=0)          # [S, T, F]
        segment_masks = torch.stack([s["mask"] for s in samples], dim=0)        # [S, T]
        segment_states = torch.tensor([int(s["state"]) for s in samples], dtype=torch.long)
        task_ids = torch.tensor([int(s["task_id"]) for s in samples], dtype=torch.long)
        starts = torch.tensor([int(s["start"]) for s in samples], dtype=torch.long)
        ends = torch.tensor([int(s["end"]) for s in samples], dtype=torch.long)
        lengths = torch.tensor([int(s["length"]) for s in samples], dtype=torch.long)

        label = int(samples[0]["label"])

        return {
            "patient_id": pid,
            "label": torch.tensor(label, dtype=torch.long),
            "segments": segments,
            "segment_masks": segment_masks,
            "segment_states": segment_states,
            "task_ids": task_ids,
            "starts": starts,
            "ends": ends,
            "lengths": lengths,
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Pads the number of segments S across patients.

        Returns:
            segments:       [B, Smax, T, F]
            segment_masks:  [B, Smax, T]
            sequence_mask:  [B, Smax]
            labels:         [B]
        """
        patient_ids = [item["patient_id"] for item in batch]
        labels = torch.stack([item["label"] for item in batch], dim=0)

        segments_list = [item["segments"] for item in batch]          # each [S, T, F]
        masks_list = [item["segment_masks"] for item in batch]        # each [S, T]
        states_list = [item["segment_states"] for item in batch]
        task_ids_list = [item["task_ids"] for item in batch]
        starts_list = [item["starts"] for item in batch]
        ends_list = [item["ends"] for item in batch]
        lengths_list = [item["lengths"] for item in batch]

        seq_lengths = torch.tensor([x.size(0) for x in segments_list], dtype=torch.long)
        max_segments = int(seq_lengths.max().item()) if len(seq_lengths) > 0 else 0

        padded_segments = pad_sequence(segments_list, batch_first=True, padding_value=0.0)
        padded_segment_masks = pad_sequence(masks_list, batch_first=True, padding_value=0.0)
        padded_states = pad_sequence(states_list, batch_first=True, padding_value=-1)
        padded_task_ids = pad_sequence(task_ids_list, batch_first=True, padding_value=-1)
        padded_starts = pad_sequence(starts_list, batch_first=True, padding_value=-1)
        padded_ends = pad_sequence(ends_list, batch_first=True, padding_value=-1)
        padded_lengths = pad_sequence(lengths_list, batch_first=True, padding_value=0)

        sequence_mask = (
            torch.arange(max_segments)[None, :] < seq_lengths[:, None]
        )

        return {
            "patient_ids": patient_ids,
            "labels": labels,
            "segments": padded_segments,             # [B, Smax, T, F]
            "segment_masks": padded_segment_masks,   # [B, Smax, T]
            "sequence_mask": sequence_mask,          # [B, Smax]
            "segment_states": padded_states,
            "task_ids": padded_task_ids,
            "starts": padded_starts,
            "ends": padded_ends,
            "lengths": padded_lengths,
            "seq_lengths": seq_lengths,
        }
