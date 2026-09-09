import pandas as pd
import torch

from src.data.pahaw_dataset import PaHaWDataset
from src.data.pahaw_sequence_dataset import PaHaWSequenceDataset


def test_sequence_dataset_builds_and_pads_sequences():
    mapping = {
        "00001": [
            {
                "X coordinate": torch.tensor([1.0, 2.0, 3.0]),
                "Y coordinate": torch.tensor([4.0, 5.0, 6.0]),
            }
        ],
        "00002": [
            {
                "X coordinate": torch.tensor([7.0, 8.0]),
                "Y coordinate": torch.tensor([9.0, 10.0]),
            }
        ],
    }
    df = pd.DataFrame(
        [
            {"ID": 1, "Disease": "PD"},
            {"ID": 2, "Disease": "H"},
        ]
    )
    pahaw_dataset = PaHaWDataset(mapping=mapping, df=df)

    dataset = PaHaWSequenceDataset.from_patient_ids(
        patient_ids=["00001", "00002"],
        pahaw_dataset=pahaw_dataset,
        feature_names=["X coordinate", "Y coordinate"],
        normalization_stats={"mean": [1.0, 4.0], "std": [1.0, 2.0]},
    )

    sequence, label, length = dataset[0]
    assert sequence.shape == (3, 2)
    assert label == 1
    assert length == 3
    assert torch.allclose(sequence[0], torch.tensor([0.0, 0.0]))

    padded, labels, lengths, mask = PaHaWSequenceDataset.collate_fn(
        [dataset[0], dataset[1]]
    )
    assert padded.shape == (2, 3, 2)
    assert labels.tolist() == [1, 0]
    assert lengths.tolist() == [3, 2]
    assert mask.tolist() == [
        [True, True, True],
        [True, True, False],
    ]


def test_sequence_dataset_can_downsample_and_cap_length():
    mapping = {
        "00001": [
            {
                "X coordinate": torch.arange(10, dtype=torch.float32),
                "Y coordinate": torch.arange(10, dtype=torch.float32) + 100,
            }
        ]
    }
    df = pd.DataFrame([{"ID": 1, "Disease": "PD"}])
    pahaw_dataset = PaHaWDataset(mapping=mapping, df=df)

    dataset = PaHaWSequenceDataset.from_patient_ids(
        patient_ids=["00001"],
        pahaw_dataset=pahaw_dataset,
        feature_names=["X coordinate", "Y coordinate"],
        downsample_factor=2,
        max_sequence_length=3,
    )

    sequence, _, length = dataset[0]
    assert sequence.shape == (3, 2)
    assert length == 3
    assert sequence[:, 0].tolist() == [0.0, 4.0, 8.0]
