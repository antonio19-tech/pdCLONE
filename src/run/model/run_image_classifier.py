import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from src.data.pahaw_dataset import PaHaWDataset
from src.data.pahaw_image_dataset import PaHaWImageDataset
from src.metrics import build_metric
from src.metrics.base_metric import BaseMetric
from src.run.reporting import build_result_writer
from src.run.reporting.base_result_writer import BaseResultWriter


class RunImageClassifier:
    """
    Pure executor: receives an already-initialized dataset and config,
    runs patient-level K-Fold cross validation for image classification models.

    Responsibilities:
        - Patient-level K-Fold splitting
        - Train / validate loop per fold
        - Train-fold normalization
        - Save per-fold checkpoints and metrics

    NOT responsible for:
        - Loading YAML / building configs
        - Initializing PaHaWDataset
        - Defining the model architecture
    """

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
        self._metrics: list[BaseMetric] = [
            build_metric(m) for m in training_cfg["metrics"]
        ]
        self._result_writer: BaseResultWriter | None = build_result_writer(training_cfg)
        self._input_size = int(training_cfg.get("input_size", 64))
        self._normalize_images = training_cfg.get("normalize_images", True)
        self._stats_batch_size = int(training_cfg.get("stats_batch_size", training_cfg["batch_size"]))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run K-Fold cross validation for all experiments."""
        for experiment_cfg in self._experiment_cfgs:
            print(f"\n[RunImageClassifier] Starting experiment: {experiment_cfg['name']}")
            self._run_experiment(experiment_cfg)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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
            print(f"  [RunImageClassifier] Fold {fold + 1}/{self._training_cfg['k_folds']}")

            train_patients = [patients[i] for i in train_idx]
            val_patients = [patients[i] for i in val_idx]

            normalization_stats = self._build_normalization_stats(train_patients, experiment_cfg)
            if normalization_stats is not None:
                self._save_normalization_stats(output_dir, fold, normalization_stats)

            train_loader = self._build_loader(
                train_patients,
                experiment_cfg,
                shuffle=True,
                normalization_stats=normalization_stats,
            )
            val_loader = self._build_loader(
                val_patients,
                experiment_cfg,
                shuffle=False,
                normalization_stats=normalization_stats,
            )

            model = self._build_model()
            optimizer = torch.optim.Adam(model.parameters(), lr=self._training_cfg["learning_rate"])
            criterion = nn.CrossEntropyLoss()

            fold_result = self._train_fold(fold, model, optimizer, criterion, train_loader, val_loader)
            all_fold_results.append(fold_result)

            self._save_checkpoint(fold, model, output_dir)
            self._save_fold_result(output_dir, fold_result)

        summary = self._build_summary(experiment_cfg["name"], all_fold_results)
        self._save_experiment_summary(output_dir, summary)
        self._log_summary(summary)

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        """
        Resolves device string from config.
        Falls back to CPU if the requested CUDA device is unavailable.
        """
        if device_str.startswith("cuda"):
            if not torch.cuda.is_available():
                print(f"[RunImageClassifier] WARNING: '{device_str}' requested but CUDA not available. Falling back to CPU.")
                return torch.device("cpu")
            device_idx = int(device_str.split(":")[1]) if ":" in device_str else 0
            if device_idx >= torch.cuda.device_count():
                print(f"[RunImageClassifier] WARNING: '{device_str}' not found ({torch.cuda.device_count()} GPU(s) available). Falling back to cuda:0.")
                return torch.device("cuda:0")
        return torch.device(device_str)

    def _build_normalization_stats(
        self,
        patient_ids: list[str],
        experiment_cfg: dict[str, Any],
    ) -> dict[str, list[float]] | None:
        if not self._normalize_images:
            return None

        base_dataset = self._build_image_dataset(
            patient_ids=patient_ids,
            experiment_cfg=experiment_cfg,
            transform=PaHaWImageDataset.build_transform(image_size=self._input_size),
        )
        stats = self._compute_normalization_stats(base_dataset)
        print(
            "    [RunImageClassifier] Normalization stats "
            f"mean={stats['mean']} std={stats['std']}"
        )
        return stats

    def _build_loader(
        self,
        patient_ids: list[str],
        experiment_cfg: dict[str, Any],
        shuffle: bool,
        normalization_stats: dict[str, list[float]] | None,
    ) -> DataLoader:
        transform = PaHaWImageDataset.build_transform(
            image_size=self._input_size,
            normalization_stats=normalization_stats,
        )
        image_dataset = self._build_image_dataset(
            patient_ids=patient_ids,
            experiment_cfg=experiment_cfg,
            transform=transform,
        )
        return DataLoader(
            image_dataset,
            batch_size=self._training_cfg["batch_size"],
            shuffle=shuffle,
            num_workers=self._training_cfg["num_workers"],
            worker_init_fn=self._worker_init_fn,
            generator=torch.Generator().manual_seed(self._seed),
        )

    def _build_image_dataset(
        self,
        patient_ids: list[str],
        experiment_cfg: dict[str, Any],
        transform,
    ) -> PaHaWImageDataset:
        return PaHaWImageDataset.from_patient_ids(
            patient_ids=patient_ids,
            processed_dir=experiment_cfg["processed_dir"],
            pahaw_dataset=self._dataset,
            disease_column=experiment_cfg["disease_column"],
            transform=transform,
        )

    def _compute_normalization_stats(
        self,
        image_dataset: PaHaWImageDataset,
    ) -> dict[str, list[float]]:
        stats_loader = DataLoader(
            image_dataset,
            batch_size=self._stats_batch_size,
            shuffle=False,
            num_workers=self._training_cfg["num_workers"],
            worker_init_fn=self._worker_init_fn,
            generator=torch.Generator().manual_seed(self._seed),
        )

        channel_sum = None
        channel_sum_sq = None
        total_pixels = 0

        for images, _ in stats_loader:
            if channel_sum is None or channel_sum_sq is None:
                num_channels = images.size(1)
                channel_sum = torch.zeros(num_channels, dtype=torch.float64)
                channel_sum_sq = torch.zeros(num_channels, dtype=torch.float64)

            channel_sum += images.sum(dim=(0, 2, 3), dtype=torch.float64)
            channel_sum_sq += (images ** 2).sum(dim=(0, 2, 3), dtype=torch.float64)
            total_pixels += images.size(0) * images.size(2) * images.size(3)

        if channel_sum is None or channel_sum_sq is None or total_pixels == 0:
            raise ValueError("Cannot compute normalization stats on an empty dataset.")

        mean = channel_sum / total_pixels
        variance = (channel_sum_sq / total_pixels) - (mean ** 2)
        std = torch.sqrt(variance.clamp(min=1e-12))

        return {
            "mean": mean.tolist(),
            "std": std.tolist(),
        }

    @staticmethod
    def _worker_init_fn(worker_id: int) -> None:
        """Ensures each DataLoader worker is seeded deterministically."""
        worker_seed = torch.initial_seed() % (2 ** 32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    def _build_model(self) -> nn.Module:
        return self._model_class(
            num_classes=self._model_cfg["num_classes"]
        ).to(self._device)

    def _resolve_output_dir(self, experiment_cfg: dict[str, Any]) -> Path:
        general_type = experiment_cfg.get("general_type_of_input")
        specific_type = experiment_cfg.get("specific_type_of_input")

        if general_type and specific_type:
            experiments_root = Path(self._training_cfg.get("experiments_root", "experiments"))
            model_name = self._resolve_model_name()
            return experiments_root / model_name / general_type / specific_type

        if "output_dir" in experiment_cfg:
            return Path(experiment_cfg["output_dir"])

        raise ValueError(
            "Experiment config must provide either 'output_dir' or both "
            "'general_type_of_input' and 'specific_type_of_input'."
        )

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
    ) -> dict[str, Any]:
        best_val_scores = {m.name: 0.0 for m in self._metrics}
        epoch_history: list[dict[str, float]] = []

        for epoch in range(self._training_cfg["epochs"]):
            train_loss = self._train_epoch(model, optimizer, criterion, train_loader)
            val_loss, val_scores = self._val_epoch(model, criterion, val_loader)

            metrics_str = " | ".join(
                f"Val {name}: {score:.4f}"
                for name, score in val_scores.items()
            )
            print(
                f"  Epoch {epoch + 1:02d}/{self._training_cfg['epochs']} | "
                f"Train loss: {train_loss:.4f} | "
                f"Val loss: {val_loss:.4f} | "
                f"{metrics_str}"
            )

            for name, score in val_scores.items():
                if score > best_val_scores[name]:
                    best_val_scores[name] = score

            epoch_history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{f"val_{name}": score for name, score in val_scores.items()},
            })
        return {
            "fold": fold + 1,
            "history": epoch_history,
            **{f"best_val_{k}": v for k, v in best_val_scores.items()},
        }

    def _train_epoch(self, model, optimizer, criterion, loader: DataLoader) -> float:
        model.train()
        total_loss = 0.0

        for images, labels in loader:
            images, labels = images.to(self._device), labels.to(self._device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    def _val_epoch(
        self,
        model,
        criterion,
        loader: DataLoader,
    ) -> tuple[float, dict[str, float]]:
        model.eval()
        total_loss = 0.0

        for m in self._metrics:
            m.reset()

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self._device), labels.to(self._device)
                outputs = model(images)
                total_loss += criterion(outputs, labels).item()
                for m in self._metrics:
                    m.update(outputs, labels)

        scores = {m.name: m.compute() for m in self._metrics}
        return total_loss / len(loader), scores

    def _save_checkpoint(self, fold: int, model: nn.Module, output_dir: Path) -> None:
        path = output_dir / f"fold_{fold + 1}_checkpoint.pth"
        torch.save(model.state_dict(), path)
        print(f"    [RunImageClassifier] Checkpoint saved → {path}")

    def _save_fold_result(self, output_dir: Path, fold_result: dict[str, Any]) -> None:
        if self._result_writer is None:
            return
        self._result_writer.save_fold_result(output_dir, fold_result)

    def _build_summary(self, name: str, all_fold_results: list[dict[str, Any]]) -> dict[str, Any]:
        metric_keys = [f"best_val_{m.name}" for m in self._metrics]
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
        print(f"\n[RunImageClassifier] Summary — {summary['experiment']}")
        print("-" * 40)
        metric_keys = list(summary["averages"].keys())
        for fold_result in summary["folds"]:
            scores_str = " | ".join(
                f"{k}: {fold_result[k]:.4f}" for k in metric_keys
            )
            print(f"  Fold {fold_result['fold']} | {scores_str}")
        for key, avg in summary["averages"].items():
            print(f"  Average {key}: {avg:.4f}")
