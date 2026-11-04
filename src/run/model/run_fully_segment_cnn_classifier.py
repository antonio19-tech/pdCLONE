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
from src.run.preprocess.run_preprocessing_segment import (
    SegmentDataConfig,
    build_segment_dataset
)
from src.common.utils import set_determinism
from src.preprocess.segment.segmenter import aggregate_patient_logits
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


class RunFullySegmentCNN:
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
        window_size: int = 128,
        stride: int | None = None,
        batch_size: int = 32,
        epochs: int = 20,
        learning_rate: float = 1e-3,
        regularization: float = 1e-3,
        n_splits: int = 5,
        include_last: bool = True,
        normalize: bool = True,
        normalization_eps: float = 1e-6,
        seed: int = 42,
        device: str = "cpu",
        aggregation: str = "mean",
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
        self.window_size = window_size
        self.stride = stride

        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.regularization = regularization

        self._training_cfg = training_cfg or {}

        self.n_splits = n_splits
        self.include_last = include_last
        self.normalize = normalize
        self.normalization_eps = normalization_eps
        self.seed = seed

        self.device = torch.device(device)
        self.aggregation = aggregation
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
        # Storage per Grad-CAM
        self.trained_models_: dict[int, nn.Module] = {}
        self.normalization_payloads_: dict[int, dict] = {}
        self.val_datasets_: dict[int, Dataset] = {}

    @property
    def num_features(self) -> int:
        return len(self.feature_names) + len(self.engineered_features)

    def run(self):
        """Execute full patient-wise cross-validation experiment."""

        set_determinism(self.seed)

        dataset_cfg = SegmentDataConfig(
            data_root=self.data_root,
            metadata_path=self.metadata_path,
            patient_ids=self.patient_ids,
            tasks=self.tasks,
            feature_names=self.feature_names,
            engineered_features=self.engineered_features,
            window_size=self.window_size,
            stride=self.stride,
            include_last=self.include_last,
            cache_files=True,
        )

        validation_cfg = SegmentDataConfig(
            data_root=self.data_root,
            metadata_path=self.metadata_path,
            patient_ids=self.patient_ids,
            tasks=self.tasks,
            feature_names=self.feature_names,
            engineered_features=self.engineered_features,
            window_size=self.window_size,
            stride=self.stride,
            include_last=self.include_last,
            cache_files=True,
        )

        training_dataset = build_segment_dataset(dataset_cfg)
        validation_dataset = build_segment_dataset(validation_cfg)

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

            normalization_payload = self._compute_normalization_stats(training_dataset.subset_by_patients(train_patients))
            if self.output_dir is not None:
                self._save_normalization_stats(fold, normalization_payload)

            self.normalization_payloads_[fold] = normalization_payload

            fold_payload = self._fit(
                training_dataset = training_dataset,
                validation_dataset = validation_dataset,
                train_patients = train_patients,
                val_patients = val_patients,
                normalization_payload = normalization_payload,
                fold_label = f"Fold {fold:02d}",
                fold = fold,
            )
            all_fold_payloads.append(fold_payload)

            if self.output_dir and self._result_writer is not None:
                self._result_writer.save_fold_result(self.output_dir, fold_payload)

        summary = self._build_summary(all_fold_payloads)

        if self.output_dir and self._result_writer is not None:
            self._result_writer.save_experiment_summary(self.output_dir, summary)

        return summary

    def _fit(
        self,
        training_dataset: Dataset,
        validation_dataset: Dataset,
        train_patients: list[str],
        val_patients: list[str],
        normalization_payload: dict[str, Any],
        fold_label: str,
        fold: int,
    ) -> dict[str, Any]:

        train_ds = training_dataset.subset_by_patients(
            train_patients
        )

        val_ds = validation_dataset.subset_by_patients(
            val_patients
        )

        if self.normalize:
            train_ds = _NormalizedDataset(
                train_ds,
                normalization_payload,
                key="segment",
                eps=self.normalization_eps,
            )

            val_ds = _NormalizedDataset(
                val_ds,
                normalization_payload,
                key="segment",
                eps=self.normalization_eps,
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
        )

        model = VanillaCNN1D(
            in_channels=self.num_features,
            num_classes=2,
            embedding_dim=self.embedding_dim,
            dropout_p=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.regularization,
        )

        criterion = nn.CrossEntropyLoss()

        history: list[dict[str, Any]] = []

        # =====================================================
        # TRAINING
        # =====================================================

        for epoch in range(1, self.epochs + 1):

            model.train()

            running_loss = 0.0

            train_metrics = {
                name: build_metric(name)
                for name in self.metric_names
            }

            # -------------------------------------------------
            # TRAIN
            # -------------------------------------------------

            for batch in train_loader:

                segment = batch["segment"].to(self.device)
                lbls = batch["label"].to(self.device)
                mask = batch["mask"].to(self.device)

                optimizer.zero_grad()

                logits = model(
                    segment,
                    mask=mask,
                )

                for metric in train_metrics.values():
                    metric.update(
                        logits.detach(),
                        lbls,
                    )

                loss = criterion(
                    logits,
                    lbls,
                )

                loss.backward()
                optimizer.step()

                running_loss += (
                    loss.item() * segment.size(0)
                )

            train_loss = (
                running_loss / len(train_ds)
            )

            train_scores = {
                name: float(metric.compute())
                for name, metric in train_metrics.items()
            }

            # -------------------------------------------------
            # VALIDATION
            # -------------------------------------------------

            val_loss, val_metrics = self._evaluate_patient(
                model,
                val_loader,
                criterion,
            )

            # -------------------------------------------------
            # HISTORY
            # -------------------------------------------------

            epoch_payload = {
                "epoch": epoch,
                "train_loss": train_loss,

                **{
                    f"train_{name}": value
                    for name, value in train_scores.items()
                },

                "val_loss": val_loss,

                **{
                    f"val_{name}": value
                    for name, value in val_metrics.items()
                },
            }

            history.append(epoch_payload)

            self._print_epoch(
                fold_label=fold_label,
                epoch=epoch,
                train_loss=train_loss,
                train_scores=train_scores,
                validation_name="val",
                validation_loss=val_loss,
                validation_scores=val_metrics,
            )

        # =====================================================
        # LAST PAYLOAD
        # =====================================================

        last_payload = history[-1]

        fold_payload = {
            "fold": fold,
            "last_epoch": last_payload["epoch"],
            "train_loss": last_payload["train_loss"],
            "val_loss": last_payload["val_loss"],
            **{
                key: value
                for key, value in last_payload.items()
                if key.startswith("train_") or key.startswith("val_")
            },

            "history": history,
        }

        self.trained_models_[fold] = model
        self.val_datasets_[fold] = val_ds

        return fold_payload

    @staticmethod
    def _print_epoch(
        fold_label: str,
        epoch: int,
        train_loss: float,
        train_scores: dict[str, float],
        validation_name: str | None = None,
        validation_loss: float | None = None,
        validation_scores: dict[str, float] | None = None,
    ) -> None:

        pieces = [
            fold_label,
            f"Epoch {epoch:02d}",
            f"train_loss={train_loss:.4f}",
        ]

        for name, value in train_scores.items():
            pieces.append(
                f"train_{name}={value:.4f}"
            )

        if (
            validation_name is not None
            and validation_loss is not None
        ):
            pieces.append(
                f"{validation_name}_loss="
                f"{validation_loss:.4f}"
            )

        if (
            validation_name is not None
            and validation_scores is not None
        ):
            for name, value in validation_scores.items():
                pieces.append(
                    f"{validation_name}_{name}={value:.4f}"
                )

        print(" | ".join(pieces))

    def _split_patient_wise(self, patient_ids: np.ndarray, labels: np.ndarray):
        """Patient-wise stratified split."""

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.seed,
        )
        return skf.split(patient_ids, labels)

    def _compute_normalization_stats(
        self,
        dataset: Dataset,
    ) -> dict[str, Any]:

        valid_values = []

        for idx in range(len(dataset)):

            sample = dataset[idx]

            x = sample["segment"].float()
            mask = sample["mask"].bool()

            # Mantieni solo i campioni reali
            x_valid = x[mask]

            if x_valid.numel() > 0:
                valid_values.append(
                    x_valid
                )

        if not valid_values:
            raise ValueError(
                "Cannot compute normalization stats "
                "on an empty dataset."
            )

        stacked = torch.cat(
            valid_values,
            dim=0
        )

        mean = stacked.mean(
            dim=0
        )

        std = stacked.std(
            dim=0,
            unbiased=False
        )

        return {
            "mean": mean.tolist(),
            "std": std.tolist(),
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

    def _evaluate_patient(
        self,
        model,
        loader,
        criterion,
    ) -> tuple[float, dict[str, float]]:

        """
        Evaluate at patient level using segmented validation.

        Pipeline:
            segment -> CNN -> logits
            logits grouped by patient
            aggregation(mean) -> patient logits
        """

        model.eval()

        all_logits = []
        all_labels = []
        all_pids = []

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():

            for batch in loader:

                segment = batch["segment"].to(self.device)
                mask = batch["mask"].to(self.device)
                lbls = batch["label"].to(self.device)
                pids = batch["patient_id"]

                logits = model(
                    segment,
                    mask=mask,
                )
                loss = criterion(
                    logits,
                    lbls,
                )

                total_loss += (loss.item() * segment.size(0))
                total_samples += segment.size(0)

                all_logits.append(logits.detach().cpu().numpy())
                all_labels.append(lbls.detach().cpu().numpy())
                all_pids.extend(pids)

        if not all_logits:
            raise ValueError(
                "No validation samples were processed."
            )

        # =====================================
        # AGGREGATE SEGMENT LOGITS
        # =====================================

        patient_logits = aggregate_patient_logits(
            logits=np.concatenate(all_logits, axis=0),
            patient_ids=all_pids,
            method=self.aggregation,
        )

        segment_labels = np.concatenate(
            all_labels,
            axis=0,
        )

        # =====================================
        # BUILD PATIENT LABELS
        # =====================================

        pid_to_label = {}

        for pid, label in zip(all_pids,segment_labels):
            pid_to_label.setdefault(pid,int(label),)

        patient_ids_sorted = list(patient_logits.keys())

        y_true = torch.tensor(
            [pid_to_label[pid] for pid in patient_ids_sorted],
            dtype=torch.long,
            device=self.device,
        )

        y_pred_logits = torch.tensor(
            np.array([patient_logits[pid]for pid in patient_ids_sorted]),
            dtype=torch.float32,
            device=self.device,
        )

        # =====================================
        # METRICS
        # =====================================

        results = {}
        for name, metric in self._metrics.items():

            metric.reset()
            metric.update(
                y_pred_logits,
                y_true,
            )
            value = metric.compute()

            if isinstance(value, torch.Tensor):
                value = value.item()

            results[name] = float(value)

        return (total_loss / total_samples, results,)

    def _build_summary(self, all_fold_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Build summary payload compatible with result writers."""
        averages = {}

        # ----- ultima epoca -----
        averages["train_loss"] = float(
            np.mean([fold["train_loss"] for fold in all_fold_results])
        )

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
            "experiment": "FullySegmentCnn",
            "normalize": self.normalize,
            "features": {
                "base": list(self.feature_names),
                "engineered": list(self.engineered_features),
            },
            "folds": all_fold_results,
            "averages": averages,
        }
