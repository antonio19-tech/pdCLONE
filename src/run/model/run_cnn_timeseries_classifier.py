from pathlib import Path
from typing import Any, Sequence

import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold

from src.metrics import build_metric
from src.metrics.base_metric import BaseMetric
from src.model.vanilla_cnn1D import VanillaCNN1D
from src.run.preprocess.run_preprocessing_timeseries import (
    TimeSeriesDataConfig,
    build_timeseries_dataset,
)
from src.preprocess.timeseries.timeseries_dataset import TimeSeriesDataset
from src.common.utils import set_determinism
from src.run.reporting import build_result_writer
from src.run.reporting.base_result_writer import BaseResultWriter



class _NormalizedDataset(Dataset):
    """Apply fold-level normalization without mutating the underlying dataset."""

    def __init__(
        self,
        dataset: Dataset,
        normalization_stats: dict[str, Any],
        key: str,
        eps: float,
    ) -> None:
        self.dataset = dataset
        self.key = key
        self.eps = eps
        self.mean = torch.tensor(normalization_stats["mean"], dtype=torch.float32)
        self.std = torch.tensor(normalization_stats["std"], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample = dict(self.dataset[idx])
        x = sample[self.key].float()

        mean = self.mean
        std = self.std.clamp_min(self.eps)

        sample[self.key] = (x - mean) / std
        return sample


class RunTimeSeriesCNN:
    """
    Patient-wise cross-validation using a CNN on fixed-size segments.

    - Training unit: segment
    - Evaluation unit: patient (aggregated predictions)
    """

    def __init__(
        self,
        data_root: str | Path,
        metadata_path: str | Path,
        patient_ids: Sequence[str] | None = None,
        tasks: Sequence[int] | None = None,
        feature_names: Sequence[str] | None = None,
        engineered_features: Sequence[str] = (),
        batch_size: int = 32,
        epochs: int = 20,
        learning_rate: float = 1e-3,
        regularization: float = 1e-3,
        n_splits: int = 5,
        normalize: bool = True,
        normalization_eps: float = 1e-6,
        seed: int = 42,
        device: str = "cpu",
        output_dir: str | Path | None = None,
        training_cfg: dict[str, Any] | None = None,
        # CNN hyperparameters
        embedding_dim: int = 64,
        dropout: float = 0.3,
    ):
        self.data_root = Path(data_root)
        self.metadata_path = Path(metadata_path)
        self.feature_names = tuple(feature_names) if feature_names is not None else ("y", "x")
        self.engineered_features = tuple(engineered_features)

        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.regularization = regularization

        self._training_cfg = training_cfg or {}

        self.n_splits = n_splits
        self.normalize = normalize
        self.normalization_eps = normalization_eps
        self.seed = seed

        self.device = torch.device(device)
        self.output_dir = Path(output_dir) if output_dir else None

        self.patient_ids = [str(p).zfill(5) for p in patient_ids] if patient_ids else None
        self.tasks = list(tasks) if tasks else None

        self.normalize = bool(
            self._training_cfg.get("normalize", self.normalize)
        )
        self.normalization_eps = float(
            self._training_cfg.get("normalization_eps", self.normalization_eps)
        )
        self.metric_names: tuple[str, ...] = tuple(
            self._training_cfg.get("metrics", ("accuracy", "f1"))
        )
        if not self.metric_names:
            raise ValueError("At least one metric must be configured.")

        if self.num_features == 0:
            raise ValueError("At least one base or engineered feature is required.")

        self._metrics: dict[str, BaseMetric] = {
            name: build_metric(name) for name in self.metric_names
        }
        self._result_writer: BaseResultWriter | None = build_result_writer(self._training_cfg)

        self.embedding_dim = embedding_dim
        self.dropout = dropout

    @property
    def num_features(self) -> int:
        return len(self.feature_names) + len(self.engineered_features)

    def run(self):
        """Execute full patient-wise cross-validation experiment."""

        set_determinism(self.seed)

        dataset_cfg = TimeSeriesDataConfig(
            data_root=self.data_root,
            metadata_path=self.metadata_path,
            patient_ids=self.patient_ids,
            tasks=self.tasks,
            feature_names=self.feature_names,
            engineered_features=self.engineered_features,
            cache_files=True,
        )

        validation_cfg = TimeSeriesDataConfig(
            data_root=self.data_root,
            metadata_path=self.metadata_path,
            patient_ids=self.patient_ids,
            tasks=self.tasks,
            feature_names=self.feature_names,
            engineered_features=self.engineered_features,
            cache_files=True,
        )

        training_dataset = build_timeseries_dataset(dataset_cfg)
        validation_dataset = build_timeseries_dataset(validation_cfg)

        patient_labels = training_dataset.patient_labels()
        patient_ids = np.array(training_dataset.unique_patient_ids())
        labels = np.array([patient_labels[pid] for pid in patient_ids])

        all_fold_payloads: list[dict[str, Any]] = []

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        for fold, (train_idx, val_idx) in enumerate(
            self._split_patient_wise(patient_ids, labels),
            1,
        ):
            train_patients = patient_ids[train_idx].tolist()
            val_patients = patient_ids[val_idx].tolist()

            train_ds = training_dataset.subset_by_patients(train_patients)
            val_ds = validation_dataset.subset_by_patients(val_patients)

            normalization_payload = self._compute_normalization_stats(train_ds)
            if self.output_dir is not None:
                self._save_normalization_stats(fold, normalization_payload)

            if self.normalize:
                train_ds = _NormalizedDataset(
                    train_ds,
                    normalization_payload,
                    key = "sequence",
                    eps=self.normalization_eps,
                )
                val_ds = _NormalizedDataset(
                    val_ds,
                    normalization_payload,
                    key = "sequence",
                    eps=self.normalization_eps,
                )

            train_loader = DataLoader(
                train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=TimeSeriesDataset.pad_timeseries,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=1,
                shuffle=False,
                collate_fn=TimeSeriesDataset.pad_timeseries,
            )

            model = VanillaCNN1D(
                in_channels=self.num_features,
                num_classes=2,
                embedding_dim=self.embedding_dim,
                dropout_p=self.dropout,
            ).to(self.device)

            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.regularization)
            criterion = nn.CrossEntropyLoss()

            history : list[dict[str, Any]] = []

            for epoch in range(1, self.epochs + 1):
                model.train()
                running_loss = 0.0

                train_metrics = {
                    name: build_metric(name)
                    for name in self.metric_names
                }

                for batch in train_loader:

                    time_series = batch["sequence"].to(self.device)

                    mask = batch["mask"].to(self.device)
                    lbls = batch["label"].to(self.device)

                    optimizer.zero_grad()
                    logits = model(time_series, mask=mask)

                    for metric in train_metrics.values():
                        metric.update(logits.detach(), lbls)

                    loss = criterion(logits, lbls)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * time_series.size(0)

                train_scores = {
                    name: float(metric.compute())
                    for name, metric in train_metrics.items()
                }
                train_loss = running_loss / len(train_ds)
                val_loss, val_metrics = self._evaluate_patient(model, val_loader, criterion)

                epoch_payload = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    **{f"train_{name}": value for name, value in train_scores.items()},
                    "val_loss": val_loss,
                    **{f"val_{name}": value for name, value in val_metrics.items()},
                }
                history.append(epoch_payload)

                print(
                    f"Fold {fold:02d} | Epoch {epoch:02d} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | "
                    f"train_accuracy={train_scores['accuracy']:.4f} | "
                    f"val_accuracy={val_metrics['accuracy']:.4f}"
                )

            # Metriche ultima epoca
            last_payload = history[-1]

            fold_payload = {
                "fold": fold,
                "last_epoch": last_payload["epoch"],
                "val_loss": last_payload["val_loss"],
                **{
                    key: value
                    for key, value in last_payload.items()
                    if key.startswith("train_") or key.startswith("val_")
                },
                "history": history,
            }
            all_fold_payloads.append(fold_payload)

            if self.output_dir and self._result_writer is not None:
                self._result_writer.save_fold_result(self.output_dir, fold_payload)

        summary = self._build_summary(all_fold_payloads)

        if self.output_dir and self._result_writer is not None:
            self._result_writer.save_experiment_summary(self.output_dir, summary)

        return summary

    def _split_patient_wise(self, patient_ids: np.ndarray, labels: np.ndarray):
        """Patient-wise stratified split."""

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.seed,
        )
        return skf.split(patient_ids, labels)

    def _compute_normalization_stats(self, dataset: Dataset) -> dict[str, Any]:
        time_series = []
        for idx in range(len(dataset)):
            sample = dataset[idx]
            time_series.append(sample["sequence"].numpy())

        if not time_series:
            raise ValueError("Cannot compute normalization stats on an empty dataset.")

        stacked = np.concatenate(time_series, axis=0)

        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)

        return {
            "mean": mean.astype(float).tolist(),
            "std": std.astype(float).tolist(),
            "epsilon": self.normalization_eps,
        }

    def _save_normalization_stats(
        self,
        fold: int,
        normalization_payload: dict[str, Any],
    ) -> None:
        if self.output_dir is None:
            return

        normalization_path = self.output_dir / f"fold_{fold}_normalization.json"
        with normalization_path.open("w", encoding="utf-8") as f:
            json.dump(normalization_payload, f, indent=4)

    def _evaluate_patient(self, model, loader, criterion) -> tuple[float, dict[str, float]]:
        """
        Evaluate model at patient level:
        - Predict on full validation sequences
        - Aggregate logits per patient
        - Compute metrics using BaseMetric only
        """
        model.eval()

        all_logits = []
        all_labels = []

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in loader:
                sequence = batch["sequence"].to(self.device)  # (1, T, F) if batch_size=1
                mask = batch["mask"].to(self.device)
                lbls = batch["label"]

                logits = model(sequence, mask=mask)
                loss = criterion(logits, lbls.to(self.device))

                total_loss += loss.item() * sequence.size(0)
                total_samples += sequence.size(0)

                all_logits.append(logits.detach().cpu())
                all_labels.append(lbls.detach().cpu())


        if not all_logits:
            raise ValueError("No validation samples were processed.")

        logits_tensor = torch.cat(all_logits, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)

        results: dict[str, float] = {}
        for name, metric in self._metrics.items():
            metric.reset()
            metric.update(logits_tensor, labels_tensor)

            value = metric.compute()
            if isinstance(value, torch.Tensor):
                value = value.item()

            results[name] = float(value)

        return total_loss / total_samples, results

    def _build_summary(self, all_fold_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Build summary payload compatible with result writers."""
        averages = {}

        # ----- ultima epoca  -----
        averages["val_loss"] = float(
            np.mean([fold["val_loss"] for fold in all_fold_results])
        )

        for name in self.metric_names:
            averages[f"val_{name}"] = float(
                np.mean([fold[f"val_{name}"] for fold in all_fold_results])
            )

        # ----- last epoch -----
        averages["last_epoch"] = float(
            np.mean([fold["last_epoch"] for fold in all_fold_results])
        )

        return {
            "experiment": "TimeseriesCnn",
            "normalize": self.normalize,
            "features": {
                "base": list(self.feature_names),
                "engineered": list(self.engineered_features),
            },
            "folds": all_fold_results,
            "averages": averages,
        }
