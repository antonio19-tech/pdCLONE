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
from src.run.preprocess.run_preprocessing_timeseries import (
    TimeSeriesDataConfig,
    build_timeseries_dataset,
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


class RunSegmentCNN:
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
        epochs: int = 50,
        learning_rate: float = 1e-5,
        regularization: float = 1e-6,
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

        self.early_stopping_patience = self._training_cfg.get(
            "early_stopping_patience", 15
        )

        self.min_delta = self._training_cfg.get(
            "min_delta", 1e-4
        )

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
            self._training_cfg.get("metrics", ("accuracy", "f1", "auc", "precision", "recall"))
        )
        if not self.metric_names:
            raise ValueError("At least one metric must be configured.")

        self.monitor_metric = self._training_cfg.get("monitor_metric")

        if self.monitor_metric is None:
            self.monitor_metric = "auc"

        valid_monitor_metrics = set(self.metric_names) | {"val_loss"}

        if self.monitor_metric not in valid_monitor_metrics:
            raise ValueError(
                f"monitor_metric '{self.monitor_metric}' "
                f"must be one of {valid_monitor_metrics}."
            )

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
        """Execute standard patient-wise k-fold cross-validation."""

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

        validation_cfg = TimeSeriesDataConfig(
            data_root=self.data_root,
            metadata_path=self.metadata_path,
            patient_ids=self.patient_ids,
            tasks=self.tasks,
            feature_names=self.feature_names,
            engineered_features=self.engineered_features,
            cache_files=True,
        )

        training_dataset = build_segment_dataset(dataset_cfg)
        validation_dataset = build_timeseries_dataset(validation_cfg)

        patient_labels = training_dataset.patient_labels()
        patient_ids = np.array(training_dataset.unique_patient_ids())
        labels = np.array([patient_labels[pid] for pid in patient_ids], dtype=int)

        all_fold_payloads: list[dict[str, Any]] = []

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        for fold, (train_idx, val_idx) in enumerate(
            self._split_patient_wise(patient_ids, labels),
            1,
        ):
            train_patients = patient_ids[train_idx].tolist()
            val_patients = patient_ids[val_idx].tolist()

            print("\n" + "=" * 80)
            print(f"FOLD {fold:02d}/{self.n_splits}")
            print("=" * 80)
            print(f"Train patients: {len(train_patients)}")
            print(f"Validation patients: {len(val_patients)}")

            # Save fold normalization stats using only the training patients.
            train_ds_for_norm = training_dataset.subset_by_patients(train_patients)
            normalization_payload = self._compute_normalization_stats(train_ds_for_norm)
            if self.output_dir is not None:
                self._save_normalization_stats(fold, normalization_payload)

            fold_payload = self._fit_early_stopping(training_dataset=training_dataset,
                                                    validation_dataset=validation_dataset,
                                                    train_patients=train_patients, val_patients=val_patients,
                                                    fold_label=f"Fold {fold:02d}", fold=fold)
            all_fold_payloads.append(fold_payload)

            print(
                f"Fold {fold:02d} done | "
                f"best_epoch={fold_payload['best_epoch']} | "
                f"val_loss={fold_payload['val_loss']:.4f} | "
                f"val_{self.metric_names[0]}={fold_payload.get(f'val_{self.metric_names[0]}', float('nan')):.4f}"
            )

            if self.output_dir is not None and self._result_writer is not None:
                self._result_writer.save_fold_result(self.output_dir, fold_payload)

        summary = self._build_summary(all_fold_payloads)

        if self.output_dir is not None and self._result_writer is not None:
            self._result_writer.save_experiment_summary(self.output_dir, summary)

        return summary

    def _split_patient_wise(
        self,
        patient_ids: np.ndarray,
        labels: np.ndarray,
        n_splits: int | None = None,
        seed: int | None = None,
    ):
        """Patient-wise stratified split."""

        skf = StratifiedKFold(
            n_splits=n_splits or self.n_splits,
            shuffle=True,
            random_state=self.seed if seed is None else seed,
        )
        return skf.split(patient_ids, labels)

    def _fit_early_stopping(
        self,
        training_dataset: Dataset,
        validation_dataset: Dataset,
        train_patients: list[str],
        val_patients: list[str],
        fold_label: str,
        fold: int,
    ) -> dict[str, Any]:
        train_ds = training_dataset.subset_by_patients(train_patients)
        val_ds = validation_dataset.subset_by_patients(val_patients)
        normalization_payload = self._compute_normalization_stats(train_ds)

        train_loader = self._build_segment_loader(
            train_ds,
            normalization_payload,
            shuffle=True,
        )
        val_loader = self._build_sequence_loader(
            val_ds,
            normalization_payload,
        )
        model, optimizer, criterion = self._build_training_components()

        best_monitor_value = np.inf if self.monitor_metric == "val_loss" else -np.inf
        best_accuracy = -np.inf
        best_val_loss = np.inf
        best_epoch = 0
        patience_counter = 0
        history: list[dict[str, Any]] = []

        for epoch in range(1, self.epochs + 1):
            train_loss, train_scores = self._train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
            )
            val_loss, val_metrics = self._evaluate_patient(
                model,
                val_loader,
                criterion,
            )
            val_stats = {
                "val_loss": val_loss,
                **{f"val_{name}": value for name, value in val_metrics.items()},
            }

            epoch_payload = {
                "epoch": epoch,
                "train_loss": train_loss,
                **{
                    f"train_{name}": value
                    for name, value in train_scores.items()
                },
                **val_stats,
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
            # SELEZIONE BEST EPOCH
            # =====================================================

            if self.monitor_metric == "val_loss":
                current_monitor = val_loss
                improved = current_monitor < (
                    best_monitor_value - self.min_delta
                )
            else:
                current_monitor = val_metrics[self.monitor_metric]
                current_accuracy = val_metrics["accuracy"]

                # AUC maggiore
                if current_monitor > (
                    best_monitor_value + self.min_delta
                ):
                    improved = True

                # AUC uguale -> accuracy maggiore
                elif abs(current_monitor - best_monitor_value) <= self.min_delta:

                    if current_accuracy > best_accuracy:
                        improved = True

                    # AUC e Accuracy uguali -> loss minore
                    elif abs(current_accuracy - best_accuracy) <= self.min_delta:

                        improved = val_loss < (best_val_loss - self.min_delta)

                    else:
                        improved = False
                else:
                    improved = False

            # Aggiornamento best epoch
            if improved:
                best_monitor_value = current_monitor

                if self.monitor_metric != "val_loss":
                    best_accuracy = val_metrics["accuracy"]

                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= self.early_stopping_patience:
                print(
                    f"{fold_label} | Early stopping at epoch {epoch:02d} "
                    f"(best epoch: {best_epoch:02d}, "
                    f"best {self.monitor_metric}: {best_monitor_value:.4f})"
                )
                break

        best_payload = history[best_epoch - 1]
        last_payload = history[-1]

        fold_payload = {
            "fold": fold,

            # ----- best epoch -----
            "best_epoch": best_epoch,
            "val_loss": best_payload["val_loss"],
            **{
                key: value
                for key, value in best_payload.items()
                if key.startswith("train_") or key.startswith("val_")
            },

            # ----- last epoch -----
            "last_epoch": last_payload["epoch"],
            **{
                f"last_{key}": value
                for key, value in last_payload.items()
                if key.startswith("train_") or key.startswith("val_")
            },

            "history": history,
        }
        return fold_payload

    def _build_training_components(self):
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
        return model, optimizer, criterion

    def _build_segment_loader(
        self,
        dataset: Dataset,
        normalization_payload: dict[str, Any],
        shuffle: bool,
    ) -> DataLoader:
        if self.normalize:
            dataset = _NormalizedDataset(
                dataset,
                normalization_payload,
                key="segment",
                eps=self.normalization_eps,
            )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
        )

    def _build_sequence_loader(
        self,
        dataset: Dataset,
        normalization_payload: dict[str, Any],
    ) -> DataLoader:
        if self.normalize:
            dataset = _NormalizedDataset(
                dataset,
                normalization_payload,
                key="sequence",
                eps=self.normalization_eps,
            )

        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )

    def _train_epoch(
        self,
        model,
        loader: DataLoader,
        optimizer,
        criterion,
    ) -> tuple[float, dict[str, float]]:
        model.train()
        running_loss = 0.0
        train_metrics = {
            name: build_metric(name)
            for name in self.metric_names
        }

        for batch in loader:
            segment = batch["segment"].to(self.device)
            lbls = batch["label"].to(self.device)
            mask = batch["mask"].to(self.device)

            optimizer.zero_grad()
            logits = model(segment, mask=mask)

            for metric in train_metrics.values():
                metric.update(logits.detach(), lbls)

            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * segment.size(0)

        train_scores = {
            name: float(metric.compute())
            for name, metric in train_metrics.items()
        }
        return running_loss / len(loader.dataset), train_scores

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

        # -----------------------------------------
        # Train metrics
        # -----------------------------------------
        for name, value in train_scores.items():
            pieces.append(f"train_{name}={value:.4f}")

        # -----------------------------------------
        # Validation metrics
        # -----------------------------------------
        if validation_name is not None and validation_loss is not None:
            pieces.append(
                f"{validation_name}_loss={validation_loss:.4f}"
            )
        if validation_name is not None and validation_scores is not None:
            for name, value in validation_scores.items():
                pieces.append(
                    f"{validation_name}_{name}={value:.4f}"
                )

        print(" | ".join(pieces))

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
        all_pids = []

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in loader:
                sequence = batch["sequence"].to(self.device)  # (1, T, F) if batch_size=1

                lbls = batch["label"]
                pids = batch["patient_id"]

                logits = model(sequence)
                loss = criterion(logits, lbls.to(self.device))

                total_loss += loss.item() * sequence.size(0)
                total_samples += sequence.size(0)

                all_logits.append(logits.detach().cpu().numpy())
                all_labels.append(lbls.detach().cpu().numpy())
                all_pids.extend(pids)

        if not all_logits:
            raise ValueError("No validation samples were processed.")

        patient_logits = aggregate_patient_logits(
            logits=np.concatenate(all_logits, axis=0),
            patient_ids=all_pids,
            method=self.aggregation,
        )

        segment_labels = np.concatenate(all_labels, axis=0)

        pid_to_label: dict[str, int] = {}
        for pid, label in zip(all_pids, segment_labels):
            if pid not in pid_to_label:
                pid_to_label[pid] = int(label)
            else:
                assert pid_to_label[pid] == int(label), f"Inconsistent labels for patient {pid}"

        patient_ids_sorted = list(patient_logits.keys())
        y_true = [pid_to_label[pid] for pid in patient_ids_sorted]

        y_true_tensor = torch.tensor(y_true, dtype=torch.long, device=self.device)
        y_pred_logits = torch.tensor(
            np.array([patient_logits[pid] for pid in patient_ids_sorted]),
            dtype=torch.float32,
            device=self.device,
        )

        results: dict[str, float] = {}
        for name, metric in self._metrics.items():
            metric.reset()
            metric.update(y_pred_logits, y_true_tensor)

            value = metric.compute()
            if isinstance(value, torch.Tensor):
                value = value.item()

            results[name] = float(value)

        return total_loss / total_samples, results

    def _build_summary(
        self,
        all_fold_results: list[dict[str, Any]],
    ) -> dict[str, Any]:

        averages = {}

        averages["train_loss"] = float(
            np.mean(
                [fold["train_loss"] for fold in all_fold_results]
            )
        )

        averages["val_loss"] = float(
            np.mean(
                [fold["val_loss"] for fold in all_fold_results]
            )
        )

        for name in self.metric_names:
            averages[f"train_{name}"] = float(
                np.mean(
                    [
                        fold[f"train_{name}"]
                        for fold in all_fold_results
                    ]
                )
            )

            averages[f"val_{name}"] = float(
                np.mean(
                    [
                        fold[f"val_{name}"]
                        for fold in all_fold_results
                    ]
                )
            )

        # ----- last epoch -----
        averages["last_epoch"] = float(
            np.mean([fold["last_epoch"] for fold in all_fold_results])
        )

        averages["last_train_loss"] = float(
            np.mean([fold["last_train_loss"] for fold in all_fold_results])
        )

        averages["last_val_loss"] = float(
            np.mean([fold["last_val_loss"] for fold in all_fold_results])
        )

        for name in self.metric_names:
            averages[f"last_train_{name}"] = float(
                np.mean([fold[f"last_train_{name}"] for fold in all_fold_results])
            )

            averages[f"last_val_{name}"] = float(
                np.mean([fold[f"last_val_{name}"] for fold in all_fold_results])
            )

        return {
            "experiment": "Vanilla_cnn1D",

            "cross_validation": {
                "splits": self.n_splits,
                "early_stopping": self.early_stopping_patience != float("inf"),
            },

            "normalize": self.normalize,

            "features": {
                "base": list(self.feature_names),
                "engineered": list(self.engineered_features),
            },

            "folds": all_fold_results,

            "averages": averages,
        }
