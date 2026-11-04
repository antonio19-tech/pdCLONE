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
from src.model.cnn_mamba_classifier import CNNMambaClassifier

from src.run.preprocess.run_preprocessing_segment import (
    SegmentDataConfig,
    build_segment_dataset
)

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
        mask_key: str | None,
        eps: float,
    ) -> None:
        self.dataset = dataset
        self.key = key
        self.mask_key = mask_key
        self.eps = eps

        self.mean = torch.tensor(
            normalization_stats["mean"],
            dtype=torch.float32,
        )

        self.std = torch.tensor(
            normalization_stats["std"],
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, object]:

        sample = dict(self.dataset[idx])

        x = sample[self.key].float()

        mean = self.mean
        std = self.std.clamp_min(self.eps)

        normalized = (x - mean) / std

        if self.mask_key is not None:

            mask = sample[self.mask_key].bool()

            normalized = (
                normalized *
                mask.unsqueeze(-1)
            )

        sample[self.key] = normalized

        return sample

class SegmentPatientDataset(Dataset):
    """
    Raggruppa i segmenti a finestra fissa appartenenti allo stesso paziente.

    Output per paziente:
        segments:      [S, T, F]
        segment_masks: [S, T]
        label:         int
        patient_id:    str
    """

    def __init__(self, segment_dataset: Dataset) -> None:
        self.segment_dataset = segment_dataset

        patient_to_indices: dict[str, list[int]] = {}

        for idx in range(len(segment_dataset)):
            sample = segment_dataset[idx]
            patient_id = str(sample["patient_id"])

            patient_to_indices.setdefault(
                patient_id,
                []
            ).append(idx)

        self.patient_ids = list(patient_to_indices.keys())
        self.patient_to_indices = patient_to_indices

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> dict[str, object]:

        patient_id = self.patient_ids[idx]
        indices = self.patient_to_indices[patient_id]

        samples = [
            self.segment_dataset[i]
            for i in indices
        ]

        segments = torch.stack(
            [
                sample["segment"].float()
                for sample in samples
            ],
            dim=0,
        )  # [S,T,F]

        segment_masks = torch.stack(
            [
                sample["mask"].bool()
                for sample in samples
            ],
            dim=0,
        )  # [S,T]

        labels = {
            int(sample["label"])
            for sample in samples
        }

        if len(labels) != 1:
            raise ValueError(
                f"Inconsistent labels for patient {patient_id}: {labels}"
            )

        return {
            "patient_id": patient_id,
            "segments": segments,
            "segment_masks": segment_masks,
            "label": next(iter(labels)),
        }

def collate_segment_patients(batch):
    """
    Crea batch di pazienti con numero variabile di segmenti.
    """

    batch_size = len(batch)

    max_segments = max(
        sample["segments"].shape[0]
        for sample in batch
    )

    segment_length = max(
        sample["segments"].shape[1]
        for sample in batch
    )

    n_features = batch[0]["segments"].shape[2]

    segments = torch.zeros(
        batch_size,
        max_segments,
        segment_length,
        n_features,
        dtype=torch.float32,
    )

    segment_masks = torch.zeros(
        batch_size,
        max_segments,
        segment_length,
        dtype=torch.bool,
    )

    sequence_mask = torch.zeros(
        batch_size,
        max_segments,
        dtype=torch.bool,
    )

    labels = torch.zeros(
        batch_size,
        dtype=torch.long,
    )

    patient_ids = []

    for i, sample in enumerate(batch):

        s = sample["segments"].shape[0]

        segments[
            i,
            :s,
            :sample["segments"].shape[1],
            :
        ] = sample["segments"]

        segment_masks[
            i,
            :s,
            :sample["segment_masks"].shape[1]
        ] = sample["segment_masks"]

        sequence_mask[i, :s] = True

        labels[i] = sample["label"]
        patient_ids.append(sample["patient_id"])

    return {
        "segments": segments,
        "segment_masks": segment_masks,
        "sequence_mask": sequence_mask,
        "labels": labels,
        "patient_ids": patient_ids,
    }

class RunFullySegmentMamba:
    """
     Patient-wise cross-validation using CNN segment embeddings + Mamba.

     - Training representation: fixed-size segments
     - Validation representation: fixed-size segments
     - Evaluation unit: patient
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
        output_dir: str | Path | None = None,
        training_cfg: dict[str, Any] | None = None,

        # CNN hyperparameters
        segment_embedding_dim: int = 64,

        # Mamba Hyperparameters
        mamba_d_model: int = 128,
        mamba_layers: int = 2,
        mamba_state: int = 16,
        mamba_expand: int = 2,
        mamba_conv: int = 4,
        dropout: float = 0.2,
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
        self.seed = seed

        self.device = torch.device(device)
        self.output_dir = Path(output_dir) if output_dir else None

        self.patient_ids = [str(p).zfill(5) for p in patient_ids] if patient_ids else None
        self.tasks = list(tasks) if tasks else None

        self.normalize = bool(
            self._training_cfg.get("normalize", normalize)
        )
        self.normalization_eps = float(
            self._training_cfg.get("normalization_eps", normalization_eps)
        )
        self.metric_names: tuple[str, ...] = tuple(
            self._training_cfg.get("metrics", ("accuracy", "f1", "auc", "precision", "recall"))
        )
        if not self.metric_names:
            raise ValueError("At least one metric must be configured.")

        if self.num_features == 0:
            raise ValueError("At least one base or engineered feature is required.")

        self._metrics: dict[str, BaseMetric] = {
            name: build_metric(name) for name in self.metric_names
        }
        self._result_writer: BaseResultWriter | None = build_result_writer(self._training_cfg)

        self.segment_embedding_dim = segment_embedding_dim

        self.mamba_d_model = mamba_d_model
        self.mamba_layers = mamba_layers
        self.mamba_state = mamba_state
        self.mamba_expand = mamba_expand
        self.mamba_conv = mamba_conv
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

            fold_payload = self._fit(training_dataset=training_dataset,
                                     validation_dataset=validation_dataset,
                                     train_patients=train_patients, val_patients=val_patients,
                                     fold_label=f"Fold {fold:02d}", fold=fold)
            all_fold_payloads.append(fold_payload)

            print(
                f"Fold {fold:02d} done | "
                f"last_epoch={fold_payload['last_epoch']} | "
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

    def _fit(
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

        self._save_normalization_stats(fold, normalization_payload)

        train_loader = self._build_segment_patient_loader(
            train_ds,
            normalization_payload,
            shuffle=True,
        )

        val_loader = self._build_segment_patient_loader(
            val_ds,
            normalization_payload,
            shuffle=False,
        )
        model, optimizer, criterion = self._build_training_components()

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
        return fold_payload

    def _build_training_components(self):

        model = CNNMambaClassifier(
            segment_input_dim=self.num_features,
            segment_embedding_dim=self.segment_embedding_dim,
            num_classes=2,
            mamba_d_model=self.mamba_d_model,
            mamba_layers=self.mamba_layers,
            mamba_state=self.mamba_state,
            mamba_expand=self.mamba_expand,
            mamba_conv=self.mamba_conv,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.regularization,
        )

        criterion = nn.CrossEntropyLoss()

        return model, optimizer, criterion

    def _build_segment_patient_loader(
        self,
        dataset: Dataset,
        normalization_payload: dict[str, Any],
        shuffle: bool,
    ) -> DataLoader:

        dataset = SegmentPatientDataset(dataset)

        if self.normalize:
            dataset = _NormalizedDataset(
                dataset,
                normalization_payload,
                key="segments",
                mask_key="segment_masks",
                eps=self.normalization_eps,
            )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate_segment_patients,
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

            segments = batch["segments"].to(self.device)
            segment_masks = batch["segment_masks"].to(self.device)
            sequence_mask = batch["sequence_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            optimizer.zero_grad()

            logits = model(
                segments=segments,
                segment_masks=segment_masks,
                sequence_mask=sequence_mask,
            )

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()
            optimizer.step()

            for metric in train_metrics.values():
                metric.update(
                    logits.detach(),
                    labels,
                )

            running_loss += (
                loss.item() *
                labels.size(0)
            )

        train_scores = {
            name: float(metric.compute())
            for name, metric in train_metrics.items()
        }

        return (
            running_loss / len(loader.dataset),
            train_scores,
        )
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

    def _evaluate_patient(
        self,
        model,
        loader,
        criterion,
    ) -> tuple[float, dict[str, float]]:

        model.eval()

        total_loss = 0.0
        total_samples = 0

        all_logits = []
        all_labels = []

        with torch.no_grad():

            for batch in loader:
                segments = batch["segments"].to(self.device)
                segment_masks = batch["segment_masks"].to(self.device)
                sequence_mask = batch["sequence_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = model(
                    segments=segments,
                    segment_masks=segment_masks,
                    sequence_mask=sequence_mask,
                )

                loss = criterion(
                    logits,
                    labels,
                )

                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)

                all_logits.append(logits.detach())
                all_labels.append(labels.detach())

        y_pred = torch.cat(all_logits, dim=0)
        y_true = torch.cat(all_labels, dim=0)

        results = {}

        for name, metric in self._metrics.items():

            metric.reset()
            metric.update(y_pred, y_true)

            value = metric.compute()

            if isinstance(value, torch.Tensor):
                value = value.item()

            results[name] = float(value)

        return (
            total_loss / max(total_samples, 1),
            results,
        )

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

        return {
            "experiment": "FullySegmentMamba",

            "cross_validation": {
                "splits": self.n_splits,
                "early_stopping": False,
            },

            "normalize": self.normalize,

            "features": {
                "base": list(self.feature_names),
                "engineered": list(self.engineered_features),
            },

            "folds": all_fold_results,

            "averages": averages,
        }
