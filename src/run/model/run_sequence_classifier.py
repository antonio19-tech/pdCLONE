from __future__ import annotations

import json
import random
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from src.data.pahaw_dataset import PaHaWDataset
from src.data.pahaw_sequence_dataset import PaHaWSequenceDataset
from src.metrics import build_metric
from src.metrics.base_metric import BaseMetric
from src.run.reporting import build_result_writer
from src.run.reporting.base_result_writer import BaseResultWriter
from src.visualization.vis_enum import PaHaWFeatures


class RunSequenceClassifier:
    """
    Runs patient-level K-Fold cross validation for raw-sequence classifiers.
    """

    _REPO_ROOT = Path(__file__).resolve().parents[3]

    def __init__(
        self,
        model_class,
        dataset: PaHaWDataset,
        valid_patients: list[str],
        experiment_cfgs: list[dict[str, Any]],
        model_cfg: dict[str, Any],
        training_cfg: dict[str, Any],
        seed: int,
    ) -> None:
        self._model_class = model_class
        self._dataset = dataset
        self._valid_patients = valid_patients
        self._experiment_cfgs = experiment_cfgs
        self._model_cfg = model_cfg
        self._training_cfg = training_cfg
        self._device = self._resolve_device(training_cfg["device"])
        self._seed = seed
        self._metrics: list[BaseMetric] = [build_metric(m) for m in training_cfg["metrics"]]
        self._result_writer: BaseResultWriter | None = build_result_writer(training_cfg)
        self._normalize_sequences = training_cfg.get("normalize_sequences", True)
        self._stats_batch_size = int(
            training_cfg.get("stats_batch_size", training_cfg["batch_size"])
        )
        self._log_every_n_batches = int(training_cfg.get("log_every_n_batches", 0))

    def run(self) -> None:
        for experiment_cfg in self._experiment_cfgs:
            print(f"\n[RunSequenceClassifier] Starting experiment: {experiment_cfg['name']}")
            self._run_experiment(experiment_cfg)

    def _run_experiment(self, experiment_cfg: dict[str, Any]) -> None:
        output_dir = self._resolve_output_dir(experiment_cfg)
        output_dir.mkdir(parents=True, exist_ok=True)

        kf = KFold(
            n_splits=self._training_cfg["k_folds"],
            shuffle=True,
            random_state=self._seed,
        )
        patients = self._valid_patients
        all_fold_results: list[dict[str, Any]] = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(patients)):
            print(f"  [RunSequenceClassifier] Fold {fold + 1}/{self._training_cfg['k_folds']}")

            train_patients = [patients[i] for i in train_idx]
            val_patients = [patients[i] for i in val_idx]

            normalization_stats = self._build_normalization_stats(train_patients, experiment_cfg)
            if normalization_stats is not None:
                self._save_normalization_stats(output_dir, fold, normalization_stats)

            train_loader = self._build_loader(
                patient_ids=train_patients,
                experiment_cfg=experiment_cfg,
                shuffle=True,
                normalization_stats=normalization_stats,
                split_name="train",
            )
            val_loader = self._build_loader(
                patient_ids=val_patients,
                experiment_cfg=experiment_cfg,
                shuffle=False,
                normalization_stats=normalization_stats,
                split_name="val",
            )

            model = self._build_model(experiment_cfg)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self._training_cfg["learning_rate"],
            )
            criterion = nn.CrossEntropyLoss()

            fold_result = self._train_fold(
                fold=fold,
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                train_loader=train_loader,
                val_loader=val_loader,
                output_dir=output_dir,
            )
            all_fold_results.append(fold_result)

            self._save_fold_result(output_dir, fold_result)

        summary = self._build_summary(experiment_cfg["name"], all_fold_results)
        self._save_experiment_summary(output_dir, summary)
        self._log_summary(summary)

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str.startswith("cuda"):
            if not torch.cuda.is_available():
                print(
                    "[RunSequenceClassifier] WARNING: "
                    f"'{device_str}' requested but CUDA not available. Falling back to CPU."
                )
                return torch.device("cpu")

            device_idx = int(device_str.split(":")[1]) if ":" in device_str else 0
            if device_idx >= torch.cuda.device_count():
                print(
                    "[RunSequenceClassifier] WARNING: "
                    f"'{device_str}' not found ({torch.cuda.device_count()} GPU(s) available). "
                    "Falling back to cuda:0."
                )
                return torch.device("cuda:0")

        return torch.device(device_str)

    def _build_normalization_stats(
        self,
        patient_ids: list[str],
        experiment_cfg: dict[str, Any],
    ) -> dict[str, list[float]] | None:
        if not self._normalize_sequences:
            return None

        dataset = self._build_sequence_dataset(
            patient_ids=patient_ids,
            experiment_cfg=experiment_cfg,
            normalization_stats=None,
        )
        stats = self._compute_normalization_stats(dataset)
        print(
            "    [RunSequenceClassifier] Normalization stats "
            f"mean={stats['mean']} std={stats['std']}"
        )
        return stats

    def _build_loader(
        self,
        patient_ids: list[str],
        experiment_cfg: dict[str, Any],
        shuffle: bool,
        normalization_stats: dict[str, list[float]] | None,
        split_name: str,
    ) -> DataLoader:
        dataset = self._build_sequence_dataset(
            patient_ids=patient_ids,
            experiment_cfg=experiment_cfg,
            normalization_stats=normalization_stats,
            split_name=split_name,
        )
        return DataLoader(
            dataset,
            batch_size=self._training_cfg["batch_size"],
            shuffle=shuffle,
            num_workers=self._training_cfg["num_workers"],
            collate_fn=PaHaWSequenceDataset.collate_fn,
            worker_init_fn=self._worker_init_fn,
            generator=torch.Generator().manual_seed(self._seed),
        )

    def _build_sequence_dataset(
        self,
        patient_ids: list[str],
        experiment_cfg: dict[str, Any],
        normalization_stats: dict[str, list[float]] | None,
        split_name: str = "train",
    ) -> PaHaWSequenceDataset:
        task_indices = self._resolve_task_indices(experiment_cfg, split_name=split_name)
        feature_names = self._resolve_feature_names(experiment_cfg["feature_names"])
        return PaHaWSequenceDataset.from_patient_ids(
            patient_ids=patient_ids,
            pahaw_dataset=self._dataset,
            feature_names=feature_names,
            disease_column=experiment_cfg["disease_column"],
            normalization_stats=normalization_stats,
            task_indices=task_indices,
            downsample_factor=int(experiment_cfg.get("downsample_factor", 1)),
            max_sequence_length=experiment_cfg.get("max_sequence_length"),
        )

    @staticmethod
    def _resolve_feature_names(feature_names: list[str]) -> list[str]:
        resolved = []
        for feature_name in feature_names:
            if feature_name in PaHaWFeatures.__members__:
                resolved.append(PaHaWFeatures[feature_name].value)
            else:
                resolved.append(feature_name)
        return resolved

    @staticmethod
    def _resolve_task_indices(
        experiment_cfg: dict[str, Any],
        split_name: str,
    ) -> list[int] | None:
        split_specific_key = f"{split_name}_task_indices"
        split_specific_numbers_key = f"{split_name}_task_numbers"

        if split_specific_key in experiment_cfg:
            return [int(task_idx) for task_idx in experiment_cfg[split_specific_key]]
        if split_specific_numbers_key in experiment_cfg:
            return [int(task_number) - 1 for task_number in experiment_cfg[split_specific_numbers_key]]
        if "task_indices" in experiment_cfg:
            return [int(task_idx) for task_idx in experiment_cfg["task_indices"]]
        if "task_numbers" in experiment_cfg:
            return [int(task_number) - 1 for task_number in experiment_cfg["task_numbers"]]
        return None

    def _compute_normalization_stats(
        self,
        dataset: PaHaWSequenceDataset,
    ) -> dict[str, list[float]]:
        stats_loader = DataLoader(
            dataset,
            batch_size=self._stats_batch_size,
            shuffle=False,
            num_workers=self._training_cfg["num_workers"],
            collate_fn=PaHaWSequenceDataset.collate_fn,
            worker_init_fn=self._worker_init_fn,
            generator=torch.Generator().manual_seed(self._seed),
        )

        feature_sum = None
        feature_sum_sq = None
        total_timesteps = 0

        for sequences, _, _, mask in stats_loader:
            masked_sequences = sequences.to(dtype=torch.float64) * mask.unsqueeze(-1)

            if feature_sum is None or feature_sum_sq is None:
                num_features = masked_sequences.size(-1)
                feature_sum = torch.zeros(num_features, dtype=torch.float64)
                feature_sum_sq = torch.zeros(num_features, dtype=torch.float64)

            feature_sum += masked_sequences.sum(dim=(0, 1))
            feature_sum_sq += masked_sequences.pow(2).sum(dim=(0, 1))
            total_timesteps += int(mask.sum().item())

        if feature_sum is None or feature_sum_sq is None or total_timesteps == 0:
            raise ValueError("Cannot compute normalization stats on an empty dataset.")

        mean = feature_sum / total_timesteps
        variance = (feature_sum_sq / total_timesteps) - (mean ** 2)
        std = torch.sqrt(variance.clamp(min=1e-12))

        return {
            "mean": mean.tolist(),
            "std": std.tolist(),
        }

    @staticmethod
    def _worker_init_fn(worker_id: int) -> None:
        worker_seed = torch.initial_seed() % (2 ** 32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    def _build_model(self, experiment_cfg: dict[str, Any]) -> nn.Module:
        model_kwargs = dict(self._model_cfg)
        model_kwargs["input_dim"] = len(self._resolve_feature_names(experiment_cfg["feature_names"]))
        return self._model_class(**model_kwargs).to(self._device)

    def _resolve_output_dir(self, experiment_cfg: dict[str, Any]) -> Path:
        general_type = experiment_cfg.get("general_type_of_input")
        specific_type = experiment_cfg.get("specific_type_of_input")

        if general_type and specific_type:
            experiments_root = self._resolve_experiments_root()
            model_name = self._resolve_model_name()
            return experiments_root / model_name / general_type / specific_type

        if "output_dir" in experiment_cfg:
            return Path(experiment_cfg["output_dir"])

        raise ValueError(
            "Experiment config must provide either 'output_dir' or both "
            "'general_type_of_input' and 'specific_type_of_input'."
        )

    def _resolve_experiments_root(self) -> Path:
        experiments_root = Path(self._training_cfg.get("experiments_root", "experiments"))
        if experiments_root.is_absolute():
            return experiments_root
        return self._REPO_ROOT / experiments_root

    def _resolve_model_name(self) -> str:
        module_name = self._model_class.__module__.split(".")[-1]
        if module_name:
            return module_name

        class_name = self._model_class.__name__
        first_pass = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", class_name)
        return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first_pass).lower()

    def _train_fold(
        self,
        fold: int,
        model: nn.Module,
        optimizer,
        criterion,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: Path,
    ) -> dict[str, Any]:
        best_val_scores = {m.name: float("-inf") for m in self._metrics}
        best_val_epochs = {m.name: -1 for m in self._metrics}
        epoch_history: list[dict[str, float]] = []

        for epoch in range(self._training_cfg["epochs"]):
            train_loss, train_scores = self._train_epoch(model, optimizer, criterion, train_loader)
            val_loss, val_scores = self._val_epoch(model, criterion, val_loader)

            train_metrics_str = " | ".join(
                f"Train {name}: {score:.4f}" for name, score in train_scores.items()
            )
            val_metrics_str = " | ".join(
                f"Val {name}: {score:.4f}" for name, score in val_scores.items()
            )
            print(
                f"  Epoch {epoch + 1:02d}/{self._training_cfg['epochs']} | "
                f"Train loss: {train_loss:.4f} | "
                f"Val loss: {val_loss:.4f} | "
                f"{train_metrics_str} | "
                f"{val_metrics_str}"
            )

            for name, score in val_scores.items():
                if score > best_val_scores[name]:
                    best_val_scores[name] = score
                    best_val_epochs[name] = epoch + 1
                    self._save_best_checkpoint(
                        fold=fold,
                        metric_name=name,
                        model=model,
                        output_dir=output_dir,
                    )

            epoch_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    **{f"train_{name}": score for name, score in train_scores.items()},
                    **{f"val_{name}": score for name, score in val_scores.items()},
                }
            )

        return {
            "fold": fold + 1,
            "history": epoch_history,
            **{f"best_val_{key}": value for key, value in best_val_scores.items()},
            **{f"best_epoch_{key}": value for key, value in best_val_epochs.items()},
        }

    def _train_epoch(
        self,
        model,
        optimizer,
        criterion,
        loader: DataLoader,
    ) -> tuple[float, dict[str, float]]:
        model.train()
        total_loss = 0.0
        start_time = time.perf_counter()
        epoch_metrics = self._fresh_metrics()

        for batch_idx, (sequences, labels, lengths, mask) in enumerate(loader, start=1):
            sequences = sequences.to(self._device)
            labels = labels.to(self._device)
            lengths = lengths.to(self._device)
            mask = mask.to(self._device)

            optimizer.zero_grad()
            outputs = model(sequences, mask=mask, lengths=lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            self._update_metrics(epoch_metrics, outputs, labels)

            if self._log_every_n_batches > 0 and (
                batch_idx % self._log_every_n_batches == 0 or batch_idx == len(loader)
            ):
                elapsed = time.perf_counter() - start_time
                batch_scores = self._compute_metric_scores(outputs, labels)
                metrics_str = " | ".join(
                    f"{name}: {score:.4f}" for name, score in batch_scores.items()
                )
                print(
                    "    [RunSequenceClassifier] "
                    f"Train batch {batch_idx}/{len(loader)} | "
                    f"loss: {loss.item():.4f} | "
                    f"{metrics_str} | "
                    f"elapsed: {elapsed:.1f}s"
                )

        return total_loss / len(loader), self._finalize_metrics(epoch_metrics)

    def _val_epoch(
        self,
        model,
        criterion,
        loader: DataLoader,
    ) -> tuple[float, dict[str, float]]:
        model.eval()
        total_loss = 0.0
        start_time = time.perf_counter()
        epoch_metrics = self._fresh_metrics()

        with torch.no_grad():
            for batch_idx, (sequences, labels, lengths, mask) in enumerate(loader, start=1):
                sequences = sequences.to(self._device)
                labels = labels.to(self._device)
                lengths = lengths.to(self._device)
                mask = mask.to(self._device)

                outputs = model(sequences, mask=mask, lengths=lengths)
                batch_loss = criterion(outputs, labels).item()
                total_loss += batch_loss
                self._update_metrics(epoch_metrics, outputs, labels)

                if self._log_every_n_batches > 0 and (
                    batch_idx % self._log_every_n_batches == 0 or batch_idx == len(loader)
                ):
                    elapsed = time.perf_counter() - start_time
                    batch_scores = self._compute_metric_scores(outputs, labels)
                    metrics_str = " | ".join(
                        f"{name}: {score:.4f}" for name, score in batch_scores.items()
                    )
                    print(
                        "    [RunSequenceClassifier] "
                        f"Val batch {batch_idx}/{len(loader)} | "
                        f"loss: {batch_loss:.4f} | "
                        f"{metrics_str} | "
                        f"elapsed: {elapsed:.1f}s"
                    )

        return total_loss / len(loader), self._finalize_metrics(epoch_metrics)

    def _save_best_checkpoint(
        self,
        fold: int,
        metric_name: str,
        model: nn.Module,
        output_dir: Path,
    ) -> None:
        path = output_dir / f"fold_{fold + 1}_best_{metric_name}_checkpoint.pth"
        state_dict = {
            key: value.detach().cpu()
            for key, value in deepcopy(model.state_dict()).items()
        }
        torch.save(state_dict, path)
        print(
            "    [RunSequenceClassifier] "
            f"Best {metric_name} checkpoint saved -> {path}"
        )

    def _save_fold_result(self, output_dir: Path, fold_result: dict[str, Any]) -> None:
        if self._result_writer is None:
            return
        self._result_writer.save_fold_result(output_dir, fold_result)

    def _build_summary(self, name: str, all_fold_results: list[dict[str, Any]]) -> dict[str, Any]:
        metric_keys = [f"best_val_{metric.name}" for metric in self._metrics]
        averages = {
            key: sum(result[key] for result in all_fold_results) / len(all_fold_results)
            for key in metric_keys
        }
        return {
            "experiment": name,
            "folds": all_fold_results,
            "averages": averages,
        }

    def _save_experiment_summary(self, output_dir: Path, summary: dict[str, Any]) -> None:
        if self._result_writer is None:
            return
        self._result_writer.save_experiment_summary(output_dir, summary)

    def _save_normalization_stats(
        self,
        output_dir: Path,
        fold: int,
        normalization_stats: dict[str, list[float]],
    ) -> None:
        path = output_dir / f"fold_{fold + 1}_normalization.json"
        path.write_text(json.dumps(normalization_stats, indent=2), encoding="utf-8")

    def _log_summary(self, summary: dict[str, Any]) -> None:
        print(f"\n[RunSequenceClassifier] Summary - {summary['experiment']}")
        print("-" * 40)
        metric_keys = list(summary["averages"].keys())
        for fold_result in summary["folds"]:
            scores_str = " | ".join(f"{key}: {fold_result[key]:.4f}" for key in metric_keys)
            print(f"  Fold {fold_result['fold']} | {scores_str}")
        for key, avg in summary["averages"].items():
            print(f"  Average {key}: {avg:.4f}")

    def _fresh_metrics(self) -> list[BaseMetric]:
        return [build_metric(metric.name) for metric in self._metrics]

    @staticmethod
    def _update_metrics(metrics: list[BaseMetric], outputs: torch.Tensor, labels: torch.Tensor) -> None:
        for metric in metrics:
            metric.update(outputs, labels)

    @staticmethod
    def _finalize_metrics(metrics: list[BaseMetric]) -> dict[str, float]:
        return {metric.name: metric.compute() for metric in metrics}

    def _compute_metric_scores(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, float]:
        batch_metrics = self._fresh_metrics()
        self._update_metrics(batch_metrics, outputs, labels)
        return self._finalize_metrics(batch_metrics)
