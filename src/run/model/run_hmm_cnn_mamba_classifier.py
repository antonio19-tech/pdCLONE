from pathlib import Path
from typing import Any, Sequence

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold

from src.common.utils import set_determinism
from src.data.pahaw_io import build_recording_index, load_svc_file, select_features
from src.metrics import build_metric
from src.metrics.base_metric import BaseMetric

from src.preprocess.hmm.hmm_segment_dataset import HMMSegmentDataset
from src.preprocess.hmm.hmm_patient_dataset import HMMPatientDataset
from src.preprocess.hmm.hmm_segmenter import HMMTimeSeriesSegmenter

from src.preprocess.segment.segmenter import aggregate_patient_logits

from src.run.preprocess.run_preprocessing_timeseries import (
    TimeSeriesDataConfig,
    build_timeseries_dataset,
)

from src.run.reporting import build_result_writer
from src.run.reporting.base_result_writer import BaseResultWriter

from src.model.cnn_mamba_classifier import CNNMambaClassifier


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

        std = self.std.clamp_min(self.eps)
        normalized = (x - self.mean) / std

        if self.mask_key is not None:

            mask = sample[self.mask_key].bool()

            # Il padding resta esattamente zero.
            normalized = normalized * mask.unsqueeze(-1)

        sample[self.key] = normalized

        return sample


def _load_sequences_for_patients(
    data_root: Path,
    metadata_path: Path,
    patient_ids: Sequence[str] | None,
    tasks: Sequence[int] | None,
    feature_names: Sequence[str],
    engineered_features: Sequence[str],
    label_column: str,
    positive_value: str,
) -> list[np.ndarray]:
    recordings = build_recording_index(
        data_root=data_root,
        metadata_path=metadata_path,
        patient_ids=patient_ids,
        tasks=tasks,
        label_column=label_column,
        positive_value=positive_value,
        skip_empty=True,
    )

    feature_columns = list(feature_names) + list(engineered_features)
    sequences: list[np.ndarray] = []

    for recording in recordings:
        raw = load_svc_file(str(recording.file_path))
        features = select_features(raw, feature_columns)
        sequences.append(features)

    return sequences


def _collect_patient_labels(
    data_root: Path,
    metadata_path: Path,
    patient_ids: Sequence[str] | None,
    tasks: Sequence[int] | None,
    label_column: str,
    positive_value: str,
) -> tuple[np.ndarray, np.ndarray]:
    recordings = build_recording_index(
        data_root=data_root,
        metadata_path=metadata_path,
        patient_ids=patient_ids,
        tasks=tasks,
        label_column=label_column,
        positive_value=positive_value,
        skip_empty=True,
    )

    patient_labels: dict[str, int] = {}
    for recording in recordings:
        patient_labels.setdefault(recording.patient_id, recording.label)

    pids = np.array(sorted(patient_labels.keys()))
    labels = np.array([patient_labels[pid] for pid in pids])

    return pids, labels


class RunHmmCnnMambaClassifier:
    """
    Patient-wise cross-validation using:
        raw sequences -> HMM segmentation -> CNN encoder -> Mamba

    - Training representation: HMM segments
    - Validation representation: full time series
    - Evaluation unit: patient

    L'HMM viene fittato esclusivamente sui pazienti di training di ciascun fold.
    In validazione ogni recording completo viene passato al modello come sequenza
    con un solo segmento; i logit dei recording dello stesso paziente vengono
    poi aggregati.

    Le metriche di training sono calcolate con una passata di inferenza in
    model.eval() al termine di ogni epoca, come in RunFullyHmmCnnMambaClassifier.
    """

    def __init__(
        self,
        data_root: str | Path,
        metadata_path: str | Path,
        patient_ids: Sequence[str] | None = None,
        tasks: Sequence[int] | None = None,
        feature_names: Sequence[str] | None = None,
        engineered_features: Sequence[str] = (),

        # HMM hyperparameters
        n_states: int = 5,
        covariance_type: str = "diag",
        n_iter: int = 50,
        tol: float = 1e-2,
        random_state: int = 42,
        algorithm: str = "viterbi",
        verbose: bool = False,
        pad_to: int = 128,
        pad_value: float = 0.0,

        batch_size: int = 8,
        epochs: int = 20,
        learning_rate: float = 1e-3,
        regularization: float = 1e-3,
        n_splits: int = 5,
        normalize: bool = True,
        normalization_eps: float = 1e-6,
        seed: int = 42,
        device: str = "cpu",
        aggregation: str = "mean",
        output_dir: str | Path | None = None,
        training_cfg: dict[str, Any] | None = None,

        # CNN hyperparameters
        segment_embedding_dim: int = 64,

        # Mamba hyperparameters
        mamba_d_model: int = 128,
        mamba_layers: int = 2,
        mamba_state: int = 16,
        mamba_expand: int = 2,
        mamba_conv: int = 4,
        dropout: float = 0.2,

        # Feature usate SOLO dall'HMM per trovare i confini dei segmenti.
        # Se entrambi sono None, l'HMM usa le stesse feature della CNN
        # (comportamento originale). La CNN riceve sempre feature_names +
        # engineered_features.
        hmm_feature_names: Sequence[str] | None = None,
        hmm_engineered_features: Sequence[str] | None = None,
    ):
        self.data_root = Path(data_root)
        self.metadata_path = Path(metadata_path)

        self.feature_names = tuple(feature_names) if feature_names is not None else ("y", "x")
        self.engineered_features = tuple(engineered_features)

        self._separate_hmm_features = (
            hmm_feature_names is not None or hmm_engineered_features is not None
        )
        if self._separate_hmm_features:
            self.hmm_feature_names = tuple(hmm_feature_names or ())
            self.hmm_engineered_features = tuple(hmm_engineered_features or ())
            if not (self.hmm_feature_names or self.hmm_engineered_features):
                raise ValueError("Le feature dell'HMM non possono essere vuote.")
        else:
            self.hmm_feature_names = self.feature_names
            self.hmm_engineered_features = self.engineered_features

        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.algorithm = algorithm
        self.verbose = verbose

        self.pad_to = pad_to
        self.pad_value = pad_value

        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.regularization = regularization

        self._training_cfg = training_cfg or {}

        self.n_splits = int(n_splits)

        self.normalize = bool(self._training_cfg.get("normalize", normalize))
        self.normalization_eps = float(
            self._training_cfg.get("normalization_eps", normalization_eps)
        )
        self.seed = seed

        self.device = torch.device(device)
        self.aggregation = aggregation
        self.output_dir = Path(output_dir) if output_dir else None

        self.patient_ids = [str(p).zfill(5) for p in patient_ids] if patient_ids else None
        self.tasks = list(tasks) if tasks else None

        self.label_column = self._training_cfg.get("label_column", "Disease")
        self.positive_value = self._training_cfg.get("positive_value", "PD")

        self.metric_names: tuple[str, ...] = tuple(
            self._training_cfg.get("metrics", ("accuracy", "f1", "auc", "precision", "recall"))
        )
        if not self.metric_names:
            raise ValueError("At least one metric must be configured.")

        if self.pad_to is None:
            raise ValueError("pad_to must be set.")
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
        """Execute patient-wise k-fold cross-validation."""

        set_determinism(self.seed)

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        patient_ids, labels = _collect_patient_labels(
            data_root=self.data_root,
            metadata_path=self.metadata_path,
            patient_ids=self.patient_ids,
            tasks=self.tasks,
            label_column=self.label_column,
            positive_value=self.positive_value,
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

        validation_dataset = build_timeseries_dataset(validation_cfg)

        all_fold_payloads: list[dict[str, Any]] = []

        print("\n==============================")
        print("START PATIENT-WISE K-FOLD CROSS VALIDATION")
        print("==============================")
        print(f"Total patients: {len(patient_ids)}")
        print(f"Folds: {self.n_splits}")
        print("==============================\n")

        for fold, (train_idx, val_idx) in enumerate(
            self._split_patient_wise(patient_ids, labels),
            1,
        ):

            train_ids = patient_ids[train_idx].tolist()
            val_ids = patient_ids[val_idx].tolist()

            print("\n" + "=" * 80)
            print(f"FOLD {fold:02d}/{self.n_splits}")
            print("=" * 80)
            print(f"Train patients: {len(train_ids)}")
            print(f"Validation patients: {len(val_ids)}")

            # =========================================================
            # HMM FIT (solo training set)
            # =========================================================

            hmm = HMMTimeSeriesSegmenter(
                n_states=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                tol=self.tol,
                random_state=self.random_state,
                algorithm=self.algorithm,
                verbose=self.verbose,
            )

            hmm.fit(
                _load_sequences_for_patients(
                    data_root=self.data_root,
                    metadata_path=self.metadata_path,
                    patient_ids=train_ids,
                    tasks=self.tasks,
                    feature_names=self.hmm_feature_names,
                    engineered_features=self.hmm_engineered_features,
                    label_column=self.label_column,
                    positive_value=self.positive_value,
                )
            )

            # =========================================================
            # DATASETS
            # =========================================================

            train_seg = HMMPatientDataset(
                self._build_hmm_segment_dataset(train_ids, hmm)
            )

            val_ds = validation_dataset.subset_by_patients(val_ids)

            normalization_payload = self._compute_normalization_stats(train_seg)

            if self.output_dir is not None:
                self._save_normalization_stats(fold, normalization_payload)

            print(f"[Fold {fold:02d}] Training Mamba model...")

            fold_payload = self._fit(
                training_dataset=train_seg,
                validation_dataset=val_ds,
                normalization_payload=normalization_payload,
                fold_label=f"Fold {fold:02d}",
                fold=fold,
            )

            all_fold_payloads.append(fold_payload)

            print(
                f"Fold {fold:02d} done | "
                f"last_epoch={fold_payload['last_epoch']} | "
                f"val_loss={fold_payload['val_loss']:.4f} | "
                f"val_accuracy={fold_payload['val_accuracy']:.4f}"
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

    def _build_hmm_segment_dataset(
        self,
        patient_ids: Sequence[str],
        hmm_segmenter: HMMTimeSeriesSegmenter,
    ) -> HMMSegmentDataset:
        return HMMSegmentDataset(
            hmm_segmenter=hmm_segmenter,
            data_root=self.data_root,
            metadata_path=self.metadata_path,
            patient_ids=patient_ids,
            tasks=self.tasks,
            feature_names=self.feature_names,
            engineered_features=self.engineered_features,
            hmm_feature_names=(
                self.hmm_feature_names if self._separate_hmm_features else None
            ),
            hmm_engineered_features=(
                self.hmm_engineered_features if self._separate_hmm_features else None
            ),
            pad_to=self.pad_to,
            pad_value=self.pad_value,
            label_column=self.label_column,
            positive_value=self.positive_value,
            cache_files=True,
        )

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

    def _build_patient_loader(
        self,
        dataset: Dataset,
        normalization_payload: dict[str, Any],
        shuffle: bool,
    ) -> DataLoader:

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
            collate_fn=HMMPatientDataset.collate_fn,
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
                mask_key=None,
                eps=self.normalization_eps,
            )

        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )

    def _fit(
        self,
        training_dataset: Dataset,
        validation_dataset: Dataset,
        normalization_payload: dict[str, Any],
        fold_label: str,
        fold: int,
    ) -> dict[str, Any]:

        train_loader = self._build_patient_loader(
            training_dataset,
            normalization_payload,
            shuffle=True,
        )

        val_loader = self._build_sequence_loader(
            validation_dataset,
            normalization_payload,
        )

        model, optimizer, criterion = self._build_training_components()

        history: list[dict[str, Any]] = []

        for epoch in range(1, self.epochs + 1):

            self._train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
            )

            train_loss, train_scores = self._evaluate_segments(
                model=model,
                loader=train_loader,
                criterion=criterion,
            )

            val_loss, val_metrics = self._evaluate_patient(
                model=model,
                loader=val_loader,
                criterion=criterion,
            )

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

        # Metriche ultima epoca
        last_payload = history[-1]

        return {
            "fold": fold,

            # ----- last epoch -----
            "last_epoch": last_payload["epoch"],
            "val_loss": last_payload["val_loss"],

            **{
                key: value
                for key, value in last_payload.items()
                if key.startswith("train_") or key.startswith("val_")
            },

            "history": history,
        }

    def _train_one_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:

        model.train()

        total_loss = 0.0
        total_samples = 0

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

            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

        return total_loss / max(total_samples, 1)

    def _evaluate_segments(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> tuple[float, dict[str, float]]:
        """
        Inferenza su un loader di pazienti segmentati (HMMPatientDataset).
        """

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

                loss = criterion(logits, labels)

                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)

                all_logits.append(logits.detach().cpu())
                all_labels.append(labels.detach().cpu())

        if not all_logits:
            raise ValueError(
                "No samples were processed."
            )

        y_pred = torch.cat(all_logits, dim=0)
        y_true = torch.cat(all_labels, dim=0)

        results: dict[str, float] = {}

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
        all_pids = []

        with torch.no_grad():

            for batch in loader:

                sequence = batch["sequence"].to(self.device)
                labels = batch["label"].to(self.device)
                pids = batch["patient_id"]

                # [B,T,F] -> [B,1,T,F]
                segments = sequence.unsqueeze(1)

                # Tutti i timestep della sequenza completa sono validi
                segment_masks = torch.ones(
                    sequence.size(0),
                    1,
                    sequence.size(1),
                    dtype=torch.bool,
                    device=self.device,
                )

                # Un solo elemento nella sequenza Mamba
                sequence_mask = torch.ones(
                    sequence.size(0),
                    1,
                    dtype=torch.bool,
                    device=self.device,
                )

                logits = model(
                    segments=segments,
                    segment_masks=segment_masks,
                    sequence_mask=sequence_mask,
                )

                loss = criterion(logits, labels)

                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)

                all_logits.append(logits.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())
                all_pids.extend(pids)

        if not all_logits:
            raise ValueError(
                "No validation samples were processed."
            )

        # =====================================
        # AGGREGATE LOGITS PER PATIENT
        # =====================================

        patient_logits = aggregate_patient_logits(
            logits=np.concatenate(all_logits, axis=0),
            patient_ids=all_pids,
            method=self.aggregation,
        )

        recording_labels = np.concatenate(all_labels, axis=0)

        pid_to_label: dict[str, int] = {}

        for pid, label in zip(all_pids, recording_labels):

            if pid not in pid_to_label:
                pid_to_label[pid] = int(label)

            elif pid_to_label[pid] != int(label):
                raise ValueError(
                    f"Inconsistent labels for patient {pid}"
                )

        patient_ids_sorted = list(patient_logits.keys())

        y_true = torch.tensor(
            [pid_to_label[pid] for pid in patient_ids_sorted],
            dtype=torch.long,
            device=self.device,
        )

        y_pred = torch.tensor(
            np.array([patient_logits[pid] for pid in patient_ids_sorted]),
            dtype=torch.float32,
            device=self.device,
        )

        results: dict[str, float] = {}

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
            pieces.append(f"train_{name}={value:.4f}")

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

        valid_segments = []

        for idx in range(len(dataset)):

            sample = dataset[idx]

            x = sample["segments"].float()  # [S, T, F]
            mask = sample["segment_masks"].bool()  # [S, T]

            valid_values = x[mask]  # [N_valid, F]

            if valid_values.numel() > 0:
                valid_segments.append(valid_values)

        if not valid_segments:
            raise ValueError(
                "Cannot compute normalization stats on empty dataset."
            )

        stacked = torch.cat(valid_segments, dim=0)

        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0, unbiased=False)

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

    def _build_summary(
        self,
        all_fold_results: list[dict[str, Any]],
    ) -> dict[str, Any]:

        averages: dict[str, float] = {}

        averages["train_loss"] = float(
            np.mean([fold["train_loss"] for fold in all_fold_results])
        )

        averages["val_loss"] = float(
            np.mean([fold["val_loss"] for fold in all_fold_results])
        )

        for name in self.metric_names:
            averages[f"train_{name}"] = float(
                np.mean([fold[f"train_{name}"] for fold in all_fold_results])
            )

            averages[f"val_{name}"] = float(
                np.mean([fold[f"val_{name}"] for fold in all_fold_results])
            )

        # ----- last epoch -----
        averages["last_epoch"] = float(
            np.mean([fold["last_epoch"] for fold in all_fold_results])
        )

        return {
            "experiment": "HmmCnnMamba",

            "cross_validation": {
                "splits": self.n_splits,
                "patient_wise": True,
                "early_stopping": False,
            },

            "representation": {
                "train": "hmm_segments",
                "validation": "full_time_series",
                "aggregation": self.aggregation,
            },

            "hmm": {
                "n_states": self.n_states,
                "covariance_type": self.covariance_type,
                "n_iter": self.n_iter,
                "tol": self.tol,
                "algorithm": self.algorithm,
                "pad_to": self.pad_to,
            },

            "normalize": self.normalize,

            "features": {
                "base": list(self.feature_names),
                "engineered": list(self.engineered_features),
            },

            "hmm_features": {
                "separate_from_cnn": self._separate_hmm_features,
                "base": list(self.hmm_feature_names),
                "engineered": list(self.hmm_engineered_features),
            },

            "model": {
                "segment_embedding_dim": self.segment_embedding_dim,
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
