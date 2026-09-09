from pathlib import Path
from typing import Any

from src.data.pahaw_dataset import PaHaWDataset
from src.preprocess.time_serie_to_image import TimeSerieToImage
from src.visualization.vis_enum import PaHaWFeatures, PaHaWTasks
from src.visualization.visualizer import PlotType


class RunPreprocessingToImage:
    """
    Pure executor: receives an already-initialized dataset and config,
    iterates over experiments and produces image files.

    Supported experiment kinds:
        - plot-based images from trajectory features
        - time-series transforms such as GAF and recurrence plots

    Responsibilities:
        - Iterate over valid patients and tasks
        - Generate image files for each (patient, task, experiment) triple
        - Write output files to the configured directories

    NOT responsible for:
        - Loading data or building mappings
        - Reading YAML / constructing configs
        - Initializing the dataset or visualizer / transforms
    """

    def __init__(
        self,
        dataset: PaHaWDataset,
        experiments: list[dict[str, Any]],
        valid_patients: list[str],
        global_cfg: dict[str, Any],
    ) -> None:
        self._dataset = dataset
        self._experiments = experiments
        self._valid_patients = valid_patients
        self._global_cfg = global_cfg
        self._task_values = [t.value for t in PaHaWTasks]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run all experiments and write output files."""
        for experiment in self._experiments:
            self._run_experiment(experiment)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_experiment(self, experiment: dict[str, Any]) -> None:
        output_dir = experiment["output_dir"]
        experiment_type = self._resolve_experiment_type(experiment)

        print(f"[PaHaW] Running experiment: {experiment['name']}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for pid_str in self._valid_patients:
            for task_idx in self._task_values:
                if experiment_type == "plot":
                    self._visualize_patient_task(
                        pid_str=pid_str,
                        task_idx=task_idx,
                        output_dir=output_dir,
                        plot_type=PlotType[experiment["plot_type"]],
                        x_feature=PaHaWFeatures[experiment["x_feature"]].value,
                        y_feature=PaHaWFeatures[experiment["y_feature"]].value,
                        z_feature=PaHaWFeatures[experiment["z_feature"]].value if experiment.get("z_feature") else None,
                    )
                elif experiment_type == "time_series_transform":
                    self._transform_patient_task(
                        pid_str=pid_str,
                        task_idx=task_idx,
                        output_dir=output_dir,
                        transform_type=experiment["transform_type"],
                        series_feature=PaHaWFeatures[experiment["series_feature"]].value,
                        transform_params=self._build_transform_params(experiment),
                        cmap=experiment.get("cmap", self._global_cfg.get("cmap", "gray")),
                    )
                else:
                    raise ValueError(
                        f"Unsupported experiment type '{experiment_type}'. "
                        "Available: ['plot', 'time_series_transform']"
                    )

    @staticmethod
    def _resolve_experiment_type(experiment: dict[str, Any]) -> str:
        if experiment.get("image_type"):
            return experiment["image_type"]
        if "plot_type" in experiment:
            return "plot"
        if "transform_type" in experiment:
            return "time_series_transform"
        raise ValueError(
            "Unable to infer experiment type. "
            "Expected 'image_type', 'plot_type', or 'transform_type'."
        )

    def _visualize_patient_task(
        self,
        pid_str: str,
        task_idx: int,
        output_dir: str,
        plot_type: PlotType,
        x_feature: str,
        y_feature: str,
        z_feature: str | None,
    ) -> None:
        filename = f"{output_dir}/patient_{pid_str}_task_{task_idx + 1}.png"
        self._dataset.visualize(
            patient_id=pid_str,
            task_idx=task_idx,
            plot_type=plot_type,
            x_feature=x_feature,
            y_feature=y_feature,
            z_feature=z_feature,
            filename=filename,
            linestyle=self._global_cfg["linestyle"],
        )

    @staticmethod
    def _build_transform_params(experiment: dict[str, Any]) -> dict[str, Any]:
        transform_type = experiment["transform_type"]
        if transform_type == "gaf":
            return {
                "image_size": experiment.get("image_size", 64),
                "method": experiment.get("method", "summation"),
            }
        if transform_type == "recurrence_plot":
            return {
                "dimension": experiment.get("dimension", 1),
                "threshold": experiment.get("threshold", "point"),
                "percentage": experiment.get("percentage", 20),
            }
        raise ValueError(
            f"Unsupported transform_type '{transform_type}'. "
            "Available: ['gaf', 'recurrence_plot']"
        )

    def _transform_patient_task(
        self,
        pid_str: str,
        task_idx: int,
        output_dir: str,
        transform_type: str,
        series_feature: str,
        transform_params: dict[str, Any],
        cmap: str,
    ) -> None:
        sample = self._dataset[pid_str, task_idx]
        if series_feature not in sample:
            raise ValueError(f"Feature '{series_feature}' not found in sample.")

        series = sample[series_feature].detach().cpu().numpy()
        image = TimeSerieToImage(series).to_image(
            transform_type=transform_type,
            **transform_params,
        )

        filename = f"{output_dir}/patient_{pid_str}_task_{task_idx + 1}.png"
        TimeSerieToImage.save(image, filename=filename, cmap=cmap)
