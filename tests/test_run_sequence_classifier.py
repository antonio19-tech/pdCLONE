from pathlib import Path

import pandas as pd
import torch

from src.data.pahaw_dataset import PaHaWDataset
from src.model.mamba_sequence_classifier import MambaSequenceClassifier
from src.run.model.run_sequence_classifier import RunSequenceClassifier


def _build_toy_dataset() -> PaHaWDataset:
    mapping = {
        "00001": [
            {
                "X coordinate": torch.tensor([1.0, 2.0, 3.0, 4.0]),
                "Y coordinate": torch.tensor([1.0, 1.5, 2.0, 2.5]),
                "pressure": torch.tensor([0.2, 0.2, 0.3, 0.4]),
            }
        ],
        "00002": [
            {
                "X coordinate": torch.tensor([4.0, 3.0, 2.0, 1.0]),
                "Y coordinate": torch.tensor([2.5, 2.0, 1.5, 1.0]),
                "pressure": torch.tensor([0.5, 0.4, 0.3, 0.2]),
            }
        ],
        "00003": [
            {
                "X coordinate": torch.tensor([1.0, 1.0, 1.0, 1.0]),
                "Y coordinate": torch.tensor([0.0, 0.1, 0.2, 0.3]),
                "pressure": torch.tensor([0.1, 0.1, 0.1, 0.1]),
            }
        ],
        "00004": [
            {
                "X coordinate": torch.tensor([3.0, 3.0, 3.0, 3.0]),
                "Y coordinate": torch.tensor([0.3, 0.2, 0.1, 0.0]),
                "pressure": torch.tensor([0.6, 0.6, 0.6, 0.6]),
            }
        ],
    }
    df = pd.DataFrame(
        [
            {"ID": 1, "Disease": "PD"},
            {"ID": 2, "Disease": "H"},
            {"ID": 3, "Disease": "PD"},
            {"ID": 4, "Disease": "H"},
        ]
    )
    return PaHaWDataset(mapping=mapping, df=df)


def test_run_sequence_classifier_saves_best_metric_checkpoints(tmp_path: Path):
    dataset = _build_toy_dataset()
    runner = RunSequenceClassifier(
        model_class=MambaSequenceClassifier,
        dataset=dataset,
        valid_patients=["00001", "00002", "00003", "00004"],
        experiment_cfgs=[
            {
                "name": "toy_sequence",
                "output_dir": str(tmp_path / "toy_sequence"),
                "disease_column": "Disease",
                "feature_names": ["X coordinate", "Y coordinate", "pressure"],
            }
        ],
        model_cfg={
            "num_classes": 2,
            "d_model": 8,
            "n_layers": 1,
            "d_state": 4,
            "expand": 1,
            "d_conv": 2,
            "dropout": 0.1,
        },
        training_cfg={
            "device": "cpu",
            "metrics": ["accuracy", "f1"],
            "save_metrics": True,
            "results_writer": "csv",
            "batch_size": 2,
            "num_workers": 0,
            "stats_batch_size": 2,
            "normalize_sequences": True,
            "log_every_n_batches": 0,
            "k_folds": 2,
            "epochs": 1,
            "learning_rate": 1e-3,
        },
        seed=11,
    )

    runner.run()

    output_dir = tmp_path / "toy_sequence"
    assert (output_dir / "fold_1_best_accuracy_checkpoint.pth").exists()
    assert (output_dir / "fold_1_best_f1_checkpoint.pth").exists()
    assert (output_dir / "fold_2_best_accuracy_checkpoint.pth").exists()
    assert (output_dir / "fold_2_best_f1_checkpoint.pth").exists()
    assert (output_dir / "fold_summary.csv").exists()
    assert (output_dir / "summary.csv").exists()


def test_run_sequence_classifier_resolves_split_specific_task_numbers():
    experiment_cfg = {
        "feature_names": ["X coordinate", "Y coordinate", "pressure"],
        "train_task_numbers": [1, 3],
        "val_task_numbers": [2],
    }

    assert RunSequenceClassifier._resolve_task_indices(experiment_cfg, split_name="train") == [0, 2]
    assert RunSequenceClassifier._resolve_task_indices(experiment_cfg, split_name="val") == [1]
