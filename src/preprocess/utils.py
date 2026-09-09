from typing import List
from src.visualization.vis_enum import PaHaWFeatures, PlotType


def generate_pahaw_paths(pahaw_dataset, base_dir: str, tasks_enum) -> List[str]:
    """
    Generate full file paths for all patients in the dataset and all tasks.

    Args:
        pahaw_dataset: PaHaWDataset object (wraps a DataFrame with "ID" column)
        base_dir: base directory containing patient folders
        tasks_enum: Enum class with tasks (e.g., PaHaWTasks)

    Returns:
        List of full paths as strings
    """
    patient_ids = pahaw_dataset.get("ID")  # get patient IDs from DataFrame
    
    paths = [
        f"{base_dir}/{str(pid).zfill(5)}/{str(pid).zfill(5)}__{task.value + 1}_1.svc"
        for pid in patient_ids
        for task in tasks_enum
    ]
    
    return paths


def validate_experiments(experiments: list[dict]) -> None:
    """Fail fast if any feature name or plot type in the YAML is invalid."""
    valid_features  = {f.name for f in PaHaWFeatures}
    valid_plot_types = {p.name for p in PlotType}

    for exp in experiments:
        assert exp["plot_type"] in valid_plot_types, \
            f"[{exp['name']}] Unknown plot_type '{exp['plot_type']}'"
        for key in ("x_feature", "y_feature"):
            assert exp[key] in valid_features, \
                f"[{exp['name']}] Unknown {key} '{exp[key]}'"
        if exp.get("z_feature"):
            assert exp["z_feature"] in valid_features, \
                f"[{exp['name']}] Unknown z_feature '{exp['z_feature']}'"
