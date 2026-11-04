import numpy as np
import torch
import os
import glob
from pathlib import Path
from ruamel.yaml import YAML
import random

from sympy import true


def load_svc_file(path: str, feature_names: list[str] | None = None) -> dict:
    """
    Load an SVC file and return a dictionary where each column is named according to feature_names.

    Args:
        path (str): Path to the .svc file
        feature_names (list of str, optional): Names for each column. If None, defaults to f1..fN.

    Returns:
        dict: keys = feature names + 'raw' + 'path', values = torch tensors
    """

    with open(path, "r") as f:
        lines = f.readlines()

    n_samples = int(lines[0].strip())
    data = [list(map(float, line.strip().split())) for line in lines[1:1+n_samples]]
    arr = np.array(data)

    n_cols = arr.shape[1]

    if feature_names is None:
        feature_names = [f"f{i+1}" for i in range(n_cols)]
    elif len(feature_names) != n_cols:
        raise ValueError(f"Number of feature names ({len(feature_names)}) does not match columns in data ({n_cols})")

    sample_dict = {name: torch.tensor(arr[:, i], dtype=torch.float32) for i, name in enumerate(feature_names)}
    sample_dict["raw"] = torch.tensor(arr, dtype=torch.float32)
    sample_dict["path"] = path

    return sample_dict


def load_svc_as_array(file_path: str) -> np.ndarray:
    """
    Load a .svc file and convert it to a 2D NumPy array suitable for pyts.
    Returns shape (1, n_timestamps).
    """
    sample = load_svc_file(file_path)

    # Get raw 2D tensor
    raw_tensor = sample["raw"]  # shape: (n_rows, n_cols)

    # Flatten to 1D and reshape to (1, -1)
    X = raw_tensor.flatten().numpy().reshape(1, -1)

    return X


def build_pahaw_mapping(base_dir, feature_names):
    mapping = {}

    for patient_folder in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, patient_folder)

        if os.path.isdir(folder_path):
            task_paths = sorted(glob.glob(os.path.join(folder_path, "*.svc")))
            samples = [
                load_svc_file(path, feature_names=feature_names)
                for path in task_paths
            ]
            mapping[patient_folder] = samples

    return mapping


def load_ruamel(path: str, typ: str = "safe") -> dict:
    yaml = YAML(typ=typ)
    return yaml.load(Path(path))


def set_determinism(seed: int) -> None:
    """
    Sets global determinism across Python, NumPy, and PyTorch (CPU and GPU).
    Call this once at the top of your entry point script before
    any data loading, model initialization, or training.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32       = False

    torch.use_deterministic_algorithms(True, warn_only=True)
