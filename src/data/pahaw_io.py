from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


# PaHaW .svc column order.
FEATURE: tuple[str, ...] = (
    "y",
    "x",
    "timestamp",
    "button",
    "azimuth",
    "altitude",
    "pressure",
)

FEATURE_ALIASES: dict[str, str] = {
    "time": "timestamp",
    "t": "timestamp",
    "ts": "timestamp",
}

ENGINEERED_FEATURES: tuple[str, ...] = (
    "speed",
    "acceleration",
    "pressure_derivative",
    "velocity",
)


@dataclass(frozen=True, slots=True)
class PaHaWRecording:
    """Metadata for one PaHaW .svc recording."""

    patient_id: str
    task_id: int
    file_path: Path
    label: int
    n_samples: int


def normalize_patient_id(patient_id: str | int) -> str:
    return str(patient_id).zfill(5)


def load_metadata(metadata_path: str | Path) -> pd.DataFrame:
    """Load the PaHaW corpus metadata and index it by subject ID."""
    metadata_path = Path(metadata_path)
    df = pd.read_excel(metadata_path, engine="openpyxl")
    if "ID" not in df.columns:
        raise ValueError("The metadata file must contain an 'ID' column")
    return df.set_index("ID", drop=False)


def patient_label(
    metadata: pd.DataFrame,
    patient_id: str,
    label_column: str = "Disease",
    positive_value: str = "PD",
) -> int:
    patient_key = int(patient_id)
    if patient_key not in metadata.index:
        raise KeyError(f"Patient ID {patient_id} not found in metadata")
    value = metadata.loc[patient_key, label_column]
    return int(value == positive_value)


def list_patient_folders(data_root: str | Path) -> list[Path]:
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    return [p for p in sorted(data_root.iterdir()) if p.is_dir() and p.name.isdigit()]


def extract_task_id(file_path: str | Path) -> int:
    # Example: 00001__1_1.svc -> task 1.
    file_path = Path(file_path)
    parts = file_path.stem.split("__")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse task ID from file name: {file_path.name}")
    task_part = parts[1]
    task_str = task_part.split("_")[0]
    return int(task_str)


def read_n_samples(file_path: str | Path) -> int:
    file_path = Path(file_path)
    with file_path.open("r", encoding="utf-8-sig") as handle:
        header = handle.readline().strip()
        if not header:
            raise ValueError(f"Empty SVC file: {file_path}")
        return int(header)


def discover_svc_files(
    data_root: str | Path,
    patient_ids: Sequence[str] | None = None,
    tasks: Sequence[int] | None = None,
) -> list[Path]:
    """Discover all .svc files under the PaHaW public folder."""
    selected_patients = (
        {normalize_patient_id(pid) for pid in patient_ids}
        if patient_ids is not None
        else None
    )
    selected_tasks = set(tasks) if tasks is not None else None

    files: list[Path] = []
    for patient_folder in list_patient_folders(data_root):
        if selected_patients is not None and patient_folder.name not in selected_patients:
            continue

        for file_path in sorted(patient_folder.glob("*.svc")):
            task_id = extract_task_id(file_path)
            if selected_tasks is not None and task_id not in selected_tasks:
                continue
            files.append(file_path)

    return files


def build_recording_index(
    data_root: str | Path,
    metadata_path: str | Path,
    patient_ids: Sequence[str] | None = None,
    tasks: Sequence[int] | None = None,
    label_column: str = "Disease",
    positive_value: str = "PD",
    skip_empty: bool = True,
) -> list[PaHaWRecording]:
    """Build a shared file/metadata index for PaHaW recordings."""
    metadata = load_metadata(metadata_path)
    selected_patients = (
        {normalize_patient_id(pid) for pid in patient_ids}
        if patient_ids is not None
        else None
    )
    selected_tasks = set(tasks) if tasks is not None else None

    recordings: list[PaHaWRecording] = []
    for patient_folder in list_patient_folders(data_root):
        patient_id = patient_folder.name
        if selected_patients is not None and patient_id not in selected_patients:
            continue

        label = patient_label(
            metadata=metadata,
            patient_id=patient_id,
            label_column=label_column,
            positive_value=positive_value,
        )

        for file_path in sorted(patient_folder.glob("*.svc")):
            task_id = extract_task_id(file_path)
            if selected_tasks is not None and task_id not in selected_tasks:
                continue

            n_samples = read_n_samples(file_path)
            if skip_empty and n_samples == 0:
                continue

            recordings.append(
                PaHaWRecording(
                    patient_id=patient_id,
                    task_id=task_id,
                    file_path=file_path,
                    label=label,
                    n_samples=n_samples,
                )
            )

    return recordings


def load_svc_file(path: str | Path) -> np.ndarray:
    """Load a PaHaW .svc file as an array with shape (n_samples, 7)."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig") as handle:
        header = handle.readline().strip()
        if not header:
            raise ValueError(f"Empty SVC file: {path}")

        n_samples = int(header)
        data = np.loadtxt(handle, dtype=np.float32, max_rows=n_samples)

    data = np.atleast_2d(data).astype(np.float32, copy=False)
    if data.shape[0] != n_samples:
        print(
            f"[WARNING] Mismatch in {path.name}: expected {n_samples}, "
            f"found {data.shape[0]}."
        )
    if data.shape[1] != len(FEATURE):
        raise ValueError(
            f"Expected {len(FEATURE)} columns in {path.name}, found {data.shape[1]}."
        )
    return data


def normalise_feature_name(feature_name: str) -> str:
    name = feature_name.strip().lower()
    return FEATURE_ALIASES.get(name, name)


def _finite_difference(values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    timestamps = np.asarray(timestamps, dtype=np.float32)

    if values.ndim != 1:
        raise ValueError("values must be 1D")
    if timestamps.ndim != 1:
        raise ValueError("timestamps must be 1D")
    if len(values) != len(timestamps):
        raise ValueError("values and timestamps must have the same length")
    if len(values) == 0:
        return values.copy()
    if len(values) == 1:
        return np.zeros_like(values)

    dt = np.diff(timestamps, prepend=timestamps[0])
    dt[0] = dt[1] if len(dt) > 1 else 1.0
    dt[dt == 0] = 1.0

    dv = np.diff(values, prepend=values[0])
    return (dv / dt).astype(np.float32)


def compute_engineered_feature(raw: np.ndarray, feature_name: str) -> np.ndarray:
    """Compute a single engineered feature from the raw PaHaW channels."""
    if raw.ndim != 2 or raw.shape[1] != len(FEATURE):
        raise ValueError(
            f"raw must have shape (n_samples, {len(FEATURE)}); got {raw.shape}."
        )

    feature_name = normalise_feature_name(feature_name)
    y = raw[:, 0]
    x = raw[:, 1]
    timestamp = raw[:, 2]
    pressure = raw[:, 6]

    if feature_name == "speed":
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        dt = np.diff(timestamp, prepend=timestamp[0])
        dt[0] = dt[1] if len(dt) > 1 else 1.0
        dt[dt == 0] = 1.0
        speed = np.sqrt(dx**2 + dy**2) / dt
        return speed.astype(np.float32)

    if feature_name == "acceleration":
        speed = compute_engineered_feature(raw, "speed")
        return _finite_difference(speed, timestamp)

    if feature_name == "pressure_derivative":
        return _finite_difference(pressure, timestamp)

    if feature_name == "velocity":
        dy = np.diff(y, prepend=y[0])
        dt = np.diff(timestamp, prepend=timestamp[0])
        dt[0] = dt[1] if len(dt) > 1 else 1.0
        dt[dt == 0] = 1.0
        velocity = dy / dt
        return velocity.astype(np.float32)

    raise ValueError(f"Unsupported engineered feature: {feature_name}")


def select_features(raw: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    """Build a feature matrix from base and/or engineered PaHaW features."""
    if not feature_names:
        raise ValueError("feature_names cannot be empty")

    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != len(FEATURE):
        raise ValueError(
            f"raw must have shape (n_samples, {len(FEATURE)}); got {raw.shape}."
        )

    columns: list[np.ndarray] = []
    for name in feature_names:
        normalised = normalise_feature_name(name)
        if normalised in FEATURE:
            idx = FEATURE.index(normalised)
            columns.append(raw[:, idx].astype(np.float32))
            continue

        if normalised in ENGINEERED_FEATURES:
            columns.append(compute_engineered_feature(raw, normalised))
            continue

        raise ValueError(
            f"Unknown feature '{name}'. Supported base features: {list(FEATURE)}; "
            f"engineered features: {list(ENGINEERED_FEATURES)}"
        )

    return np.column_stack(columns).astype(np.float32, copy=False)
