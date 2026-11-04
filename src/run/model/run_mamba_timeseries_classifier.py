from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from src.metrics import build_metric
from src.metrics.base_metric import BaseMetric

from src.model.cnn_mamba_classifier import (
    CNNMambaClassifier,
)

from src.run.preprocess.run_preprocessing_timeseries import (
    TimeSeriesDataConfig,
    build_timeseries_dataset,
)

from src.preprocess.timeseries.timeseries_dataset import (
    TimeSeriesDataset,
)

from src.common.utils import set_determinism
from src.run.reporting import build_result_writer
from src.run.reporting.base_result_writer import BaseResultWriter


class _NormalizedDataset(Dataset):
    """
    Apply fold-level normalization without mutating
    the underlying dataset.
    """

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

        sample[self.key] = (x - mean) / std

        return sample


class RunTimeSeriesCNNMamba:
    """
    Patient-wise cross-validation using:

        complete time series
            ↓
        CNN encoder (VanillaCNN1D: conv + adaptive pooling + FC)
            ↓
        64-dim embedding of the whole series
            ↓
        Mamba on a sequence of ONE element
            ↓
        classification

    Same architecture (CNNMambaClassifier) used by the
    segment-based steps 4, 5, 8 and 9: here the complete
    series is treated as a single segment, both in
    training and in validation.

    Training unit:
        complete patient time series

    Evaluation unit:
        patient
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

        # ---------------------------------------------------------
        # MAMBA HYPERPARAMETERS
        # ---------------------------------------------------------
        mamba_d_model: int = 128,
        mamba_layers: int = 2,
        mamba_state: int = 16,
        mamba_expand: int = 2,
        mamba_conv: int = 4,
        dropout: float = 0.2,
        segment_embedding_dim: int = 64,
    ) -> None:

        self.data_root = Path(data_root)
        self.metadata_path = Path(metadata_path)

        self.feature_names = (
            tuple(feature_names)
            if feature_names is not None
            else ("y", "x")
        )

        self.engineered_features = tuple(engineered_features)

        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.regularization = regularization

        self.n_splits = n_splits

        self.normalize = normalize
        self.normalization_eps = normalization_eps

        self.seed = seed

        self.device = torch.device(device)

        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else None
        )

        self.patient_ids = (
            [str(p).zfill(5) for p in patient_ids]
            if patient_ids is not None
            else None
        )

        self.tasks = (
            list(tasks)
            if tasks is not None
            else None
        )

        # ---------------------------------------------------------
        # TRAINING CONFIG
        # ---------------------------------------------------------

        self._training_cfg = training_cfg or {}

        self.early_stopping_patience = self._training_cfg.get(
            "early_stopping_patience",
            15,
        )

        self.min_delta = self._training_cfg.get(
            "min_delta",
            1e-3,
        )

        self.normalize = bool(
            self._training_cfg.get(
                "normalize",
                self.normalize,
            )
        )

        self.normalization_eps = float(
            self._training_cfg.get(
                "normalization_eps",
                self.normalization_eps,
            )
        )

        self.metric_names: tuple[str, ...] = tuple(
            self._training_cfg.get(
                "metrics",
                (
                    "accuracy",
                    "f1",
                ),
            )
        )

        if not self.metric_names:
            raise ValueError(
                "At least one metric must be configured."
            )

        self.monitor_metric = self._training_cfg.get(
            "monitor_metric"
        )

        if self.monitor_metric is None:
            self.monitor_metric = (
                "accuracy"
                if "accuracy" in self.metric_names
                else self.metric_names[0]
            )

        valid_monitor_metrics = (
            set(self.metric_names)
            | {"val_loss"}
        )

        if self.monitor_metric not in valid_monitor_metrics:
            raise ValueError(
                f"monitor_metric '{self.monitor_metric}' "
                f"must be one of {valid_monitor_metrics}."
            )

        # ---------------------------------------------------------
        # MAMBA CONFIG
        # ---------------------------------------------------------

        self.mamba_d_model = mamba_d_model
        self.mamba_layers = mamba_layers
        self.mamba_state = mamba_state
        self.mamba_expand = mamba_expand
        self.mamba_conv = mamba_conv
        self.dropout = dropout
        self.segment_embedding_dim = segment_embedding_dim

        # ---------------------------------------------------------
        # METRICS / RESULT WRITER
        # ---------------------------------------------------------

        self._metrics: dict[str, BaseMetric] = {
            name: build_metric(name)
            for name in self.metric_names
        }

        self._result_writer: BaseResultWriter | None = (
            build_result_writer(
                self._training_cfg
            )
        )

    @property
    def num_features(self) -> int:
        return (
            len(self.feature_names)
            + len(self.engineered_features)
        )

    # ============================================================
    # RUN
    # ============================================================

    def run(self) -> dict[str, Any]:
        """
        Execute full patient-wise cross-validation experiment.
        """

        set_determinism(self.seed)

        # ---------------------------------------------------------
        # DATASET CONFIGURATION
        # ---------------------------------------------------------

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

        training_dataset = build_timeseries_dataset(
            dataset_cfg
        )

        validation_dataset = build_timeseries_dataset(
            validation_cfg
        )

        # ---------------------------------------------------------
        # PATIENT LABELS
        # ---------------------------------------------------------

        patient_labels = (
            training_dataset.patient_labels()
        )

        patient_ids = np.array(
            training_dataset.unique_patient_ids()
        )

        labels = np.array(
            [
                patient_labels[pid]
                for pid in patient_ids
            ]
        )

        all_fold_payloads: list[
            dict[str, Any]
        ] = []

        if self.output_dir is not None:
            self.output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        # =========================================================
        # PATIENT-WISE K-FOLD
        # =========================================================

        for fold, (
            train_idx,
            val_idx,
        ) in enumerate(
            self._split_patient_wise(
                patient_ids,
                labels,
            ),
            1,
        ):

            train_patients = (
                patient_ids[
                    train_idx
                ].tolist()
            )

            val_patients = (
                patient_ids[
                    val_idx
                ].tolist()
            )

            print("\n" + "=" * 80)
            print(
                f"FOLD {fold:02d}/{self.n_splits}"
            )
            print("=" * 80)

            print(
                f"Train patients: "
                f"{len(train_patients)}"
            )

            print(
                f"Validation patients: "
                f"{len(val_patients)}"
            )

            # -----------------------------------------------------
            # SUBSETS
            # -----------------------------------------------------

            train_ds = (
                training_dataset.subset_by_patients(
                    train_patients
                )
            )

            val_ds = (
                validation_dataset.subset_by_patients(
                    val_patients
                )
            )

            # -----------------------------------------------------
            # NORMALIZATION
            # -----------------------------------------------------

            normalization_payload = (
                self._compute_normalization_stats(
                    train_ds
                )
            )

            if self.output_dir is not None:
                self._save_normalization_stats(
                    fold,
                    normalization_payload,
                )

            if self.normalize:

                train_ds = _NormalizedDataset(
                    train_ds,
                    normalization_payload,
                    key="sequence",
                    eps=self.normalization_eps,
                )

                val_ds = _NormalizedDataset(
                    val_ds,
                    normalization_payload,
                    key="sequence",
                    eps=self.normalization_eps,
                )

            # -----------------------------------------------------
            # DATALOADERS
            # -----------------------------------------------------

            train_loader = DataLoader(
                train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=(
                    TimeSeriesDataset.pad_timeseries
                ),
            )

            val_loader = DataLoader(
                val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=(
                    TimeSeriesDataset.pad_timeseries
                ),
            )

            # -----------------------------------------------------
            # MODEL
            # -----------------------------------------------------

            model = self._build_model()

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.regularization,
            )

            criterion = nn.CrossEntropyLoss()

            # -----------------------------------------------------
            # EARLY STOPPING
            # -----------------------------------------------------

            best_monitor_value = (
                np.inf
                if self.monitor_metric == "val_loss"
                else -np.inf
            )

            best_accuracy = -np.inf
            best_val_loss = np.inf

            best_epoch = 0
            patience_counter = 0

            history: list[
                dict[str, Any]
            ] = []

            # =====================================================
            # EPOCH LOOP
            # =====================================================

            for epoch in range(
                1,
                self.epochs + 1,
            ):

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

                    time_series = batch[
                        "sequence"
                    ].to(self.device)

                    mask = batch[
                        "mask"
                    ].to(self.device)

                    labels_batch = batch[
                        "label"
                    ].to(self.device)

                    optimizer.zero_grad()

                    # CNN + Mamba: series as a single segment
                    logits = self._forward_single_segment(
                        model,
                        time_series,
                        mask,
                    )

                    loss = criterion(
                        logits,
                        labels_batch,
                    )

                    loss.backward()

                    optimizer.step()

                    for metric in (
                        train_metrics.values()
                    ):
                        metric.update(
                            logits.detach(),
                            labels_batch,
                        )

                    running_loss += (
                        loss.item()
                        * time_series.size(0)
                    )

                # -------------------------------------------------
                # TRAIN METRICS
                # -------------------------------------------------

                train_scores = {
                    name: float(
                        metric.compute()
                    )
                    for name, metric
                    in train_metrics.items()
                }

                train_loss = (
                    running_loss
                    / len(train_ds)
                )

                # -------------------------------------------------
                # VALIDATION
                # -------------------------------------------------

                val_loss, val_metrics = (
                    self._evaluate_patient(
                        model,
                        val_loader,
                        criterion,
                    )
                )

                # -------------------------------------------------
                # HISTORY
                # -------------------------------------------------

                epoch_payload = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    **{
                        f"train_{name}": value
                        for name, value
                        in train_scores.items()
                    },
                    "val_loss": val_loss,
                    **{
                        f"val_{name}": value
                        for name, value
                        in val_metrics.items()
                    },
                }

                history.append(
                    epoch_payload
                )

                print(
                    f"Fold {fold:02d} | "
                    f"Epoch {epoch:02d} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | "
                    f"train_accuracy="
                    f"{train_scores['accuracy']:.4f} | "
                    f"val_accuracy="
                    f"{val_metrics['accuracy']:.4f}"
                )

                # =================================================
                # BEST EPOCH SELECTION
                # =================================================

                if self.monitor_metric == "val_loss":

                    current_monitor = val_loss

                    improved = (
                        current_monitor
                        < (
                            best_monitor_value
                            - self.min_delta
                        )
                    )

                else:

                    current_monitor = (
                        val_metrics[
                            self.monitor_metric
                        ]
                    )

                    current_accuracy = (
                        val_metrics["accuracy"]
                    )

                    # -------------------------------------------------
                    # 1. Monitor metric improves
                    # -------------------------------------------------

                    if current_monitor > (
                        best_monitor_value
                        + self.min_delta
                    ):

                        improved = True

                    # -------------------------------------------------
                    # 2. Monitor metric tie
                    #    -> better accuracy
                    # -------------------------------------------------

                    elif abs(
                        current_monitor
                        - best_monitor_value
                    ) <= self.min_delta:

                        if (
                            current_accuracy
                            > best_accuracy
                        ):

                            improved = True

                        # -------------------------------------------------
                        # 3. Accuracy tie
                        #    -> lower validation loss
                        # -------------------------------------------------

                        elif abs(
                            current_accuracy
                            - best_accuracy
                        ) <= self.min_delta:

                            improved = (
                                val_loss
                                < (
                                    best_val_loss
                                    - self.min_delta
                                )
                            )

                        else:

                            improved = False

                    else:

                        improved = False

                # =================================================
                # UPDATE BEST
                # =================================================

                if improved:

                    best_monitor_value = (
                        current_monitor
                    )

                    if (
                        self.monitor_metric
                        != "val_loss"
                    ):
                        best_accuracy = (
                            val_metrics[
                                "accuracy"
                            ]
                        )

                    best_val_loss = val_loss

                    best_epoch = epoch

                    patience_counter = 0

                else:

                    patience_counter += 1

                # =================================================
                # EARLY STOPPING
                # =================================================

                if (
                    patience_counter
                    >= self.early_stopping_patience
                ):

                    print(
                        f"Early stopping triggered "
                        f"at epoch {epoch:02d} "
                        f"(best epoch: "
                        f"{best_epoch:02d}, "
                        f"best "
                        f"{self.monitor_metric}: "
                        f"{best_monitor_value:.4f})"
                    )

                    break

            # =====================================================
            # BEST + LAST FOLD RESULT
            # =====================================================

            if best_epoch == 0:
                raise RuntimeError(
                    f"No valid best epoch found "
                    f"for fold {fold}."
                )

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

            all_fold_payloads.append(fold_payload)

            if (
                self.output_dir is not None
                and self._result_writer is not None
            ):
                self._result_writer.save_fold_result(
                    self.output_dir,
                    fold_payload,
                )

            print(
                f"Fold {fold:02d} | "
                f"best_epoch={best_epoch} "
                f"val_{self.monitor_metric}="
                f"{best_payload.get(f'val_{self.monitor_metric}', best_payload['val_loss']):.4f} | "
                f"last_epoch={last_payload['epoch']} "
                f"val_{self.monitor_metric}="
                f"{last_payload.get(f'val_{self.monitor_metric}', last_payload['val_loss']):.4f}"
            )


        # =========================================================
        # EXPERIMENT SUMMARY
        # =========================================================

        summary = self._build_summary(
            all_fold_payloads
        )

        if (
            self.output_dir is not None
            and self._result_writer is not None
        ):
            self._result_writer.save_experiment_summary(
                self.output_dir,
                summary,
            )

        return summary

    # ============================================================
    # MODEL
    # ============================================================

    def _build_model(self) -> nn.Module:

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

        return model

    # ============================================================
    # FORWARD: COMPLETE SERIES AS A SINGLE SEGMENT
    # ============================================================

    def _forward_single_segment(
        self,
        model: nn.Module,
        sequence: torch.Tensor,  # [B, T, F]
        mask: torch.Tensor,  # [B, T]
    ) -> torch.Tensor:
        """
        Feed each complete series to CNNMambaClassifier as a
        sequence of one segment.

        segment_masks keeps the real length of every series,
        so the CNN encoder never sees the padding added by
        pad_timeseries; sequence_mask marks the single element
        as valid.
        """

        # [B, T, F] -> [B, 1, T, F]
        segments = sequence.unsqueeze(1)

        # [B, T] -> [B, 1, T]
        segment_masks = mask.bool().unsqueeze(1)

        # one valid element per patient
        sequence_mask = torch.ones(
            sequence.size(0),
            1,
            dtype=torch.bool,
            device=sequence.device,
        )

        return model(
            segments=segments,
            segment_masks=segment_masks,
            sequence_mask=sequence_mask,
        )

    # ============================================================
    # PATIENT-WISE SPLIT
    # ============================================================

    def _split_patient_wise(
        self,
        patient_ids: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Patient-wise stratified split.
        """

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.seed,
        )

        return skf.split(
            patient_ids,
            labels,
        )

    # ============================================================
    # NORMALIZATION
    # ============================================================

    def _compute_normalization_stats(
        self,
        dataset: Dataset,
    ) -> dict[str, Any]:

        time_series = []

        for idx in range(
            len(dataset)
        ):

            sample = dataset[idx]

            time_series.append(
                sample["sequence"].numpy()
            )

        if not time_series:

            raise ValueError(
                "Cannot compute normalization "
                "stats on an empty dataset."
            )

        stacked = np.concatenate(
            time_series,
            axis=0,
        )

        mean = stacked.mean(
            axis=0
        )

        std = stacked.std(
            axis=0
        )

        return {
            "mean": mean.astype(
                float
            ).tolist(),

            "std": std.astype(
                float
            ).tolist(),

            "epsilon": self.normalization_eps,
        }

    # ============================================================
    # SAVE NORMALIZATION
    # ============================================================

    def _save_normalization_stats(
        self,
        fold: int,
        normalization_payload: dict[str, Any],
    ) -> None:

        if self.output_dir is None:
            return

        normalization_path = (
            self.output_dir
            / f"fold_{fold}_normalization.json"
        )

        with normalization_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                normalization_payload,
                f,
                indent=4,
            )

    # ============================================================
    # VALIDATION
    # ============================================================

    def _evaluate_patient(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> tuple[float, dict[str, float]]:

        model.eval()

        all_logits = []
        all_labels = []

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():

            for batch in loader:

                sequence = batch[
                    "sequence"
                ].to(self.device)

                mask = batch[
                    "mask"
                ].to(self.device)

                labels = batch[
                    "label"
                ].to(self.device)

                # CNN + Mamba: series as a single segment
                logits = self._forward_single_segment(
                    model,
                    sequence,
                    mask,
                )

                loss = criterion(
                    logits,
                    labels,
                )

                total_loss += (
                    loss.item()
                    * sequence.size(0)
                )

                total_samples += (
                    sequence.size(0)
                )

                all_logits.append(
                    logits.detach().cpu()
                )

                all_labels.append(
                    labels.detach().cpu()
                )

        if not all_logits:

            raise ValueError(
                "No validation samples "
                "were processed."
            )

        logits_tensor = torch.cat(
            all_logits,
            dim=0,
        )

        labels_tensor = torch.cat(
            all_labels,
            dim=0,
        )

        results: dict[str, float] = {}

        for name, metric in (
            self._metrics.items()
        ):

            metric.reset()

            metric.update(
                logits_tensor,
                labels_tensor,
            )

            value = metric.compute()

            if isinstance(
                value,
                torch.Tensor,
            ):
                value = value.item()

            results[name] = float(
                value
            )

        return (
            total_loss
            / max(total_samples, 1),
            results,
        )

    # ============================================================
    # SUMMARY
    # ============================================================

    def _build_summary(
        self,
        all_fold_results: list[
            dict[str, Any]
        ],
    ) -> dict[str, Any]:

        averages: dict[str, float] = {}

        # ----- best epoch -----
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

        averages["last_val_loss"] = float(
            np.mean([fold["last_val_loss"] for fold in all_fold_results])
        )

        for name in self.metric_names:
            averages[f"last_val_{name}"] = float(
                np.mean([fold[f"last_val_{name}"] for fold in all_fold_results])
            )

        return {
            "experiment": "CNN_Mamba_TimeSeries",

            "cross_validation": {
                "splits": self.n_splits,
                "patient_wise": True,
                "early_stopping": self.early_stopping_patience != float("inf"),
            },

            "normalize": self.normalize,

            "features": {
                "base": list(
                    self.feature_names
                ),
                "engineered": list(
                    self.engineered_features
                ),
            },

            "model": {
                "cnn": "VanillaCNN1D encoder (conv + adaptive pooling + FC)",
                "segment_embedding_dim": self.segment_embedding_dim,
                "sequence_length": 1,
                "mamba": {
                    "d_model": self.mamba_d_model,
                    "layers": self.mamba_layers,
                    "state": self.mamba_state,
                    "expand": self.mamba_expand,
                    "conv": self.mamba_conv,
                    "dropout": self.dropout,
                },
            },

            "folds": all_fold_results,

            "averages": averages,
        }
