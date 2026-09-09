from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _build_experiment_metadata(experiments_path: Path, result_path: Path) -> dict[str, str | None]:
    relative_dir = result_path.parent.relative_to(experiments_path)
    parts = relative_dir.parts

    metadata = {
        "experiment": relative_dir.as_posix(),
        "model_name": None,
        "general_type_of_input": None,
        "specific_type_of_input": None,
    }

    if len(parts) >= 1:
        metadata["model_name"] = parts[0]
    if len(parts) >= 2:
        metadata["general_type_of_input"] = parts[1]
    if len(parts) >= 3:
        metadata["specific_type_of_input"] = parts[2]

    return metadata


def load_experiment_summaries(experiments_dir: str | Path = "experiments") -> pd.DataFrame:
    """
    Load one row per experiment from each summary.csv file.
    """
    experiments_path = Path(experiments_dir)
    summary_frames = []

    for summary_path in sorted(experiments_path.glob("**/summary.csv")):
        summary_df = pd.read_csv(summary_path)
        metadata = _build_experiment_metadata(experiments_path, summary_path)
        summary_df["original_experiment"] = summary_df["experiment"]
        for key, value in metadata.items():
            summary_df[key] = value
        summary_frames.append(summary_df)

    if not summary_frames:
        raise FileNotFoundError(f"No summary.csv files found in '{experiments_path}'.")

    return pd.concat(summary_frames, ignore_index=True)


def load_fold_summaries(experiments_dir: str | Path = "experiments") -> pd.DataFrame:
    """
    Load one row per fold from each fold_summary.csv file.
    """
    experiments_path = Path(experiments_dir)
    fold_frames = []

    for fold_summary_path in sorted(experiments_path.glob("**/fold_summary.csv")):
        fold_df = pd.read_csv(fold_summary_path)
        metadata = _build_experiment_metadata(experiments_path, fold_summary_path)
        for key, value in metadata.items():
            fold_df[key] = value
        fold_frames.append(fold_df)

    if not fold_frames:
        raise FileNotFoundError(f"No fold_summary.csv files found in '{experiments_path}'.")

    return pd.concat(fold_frames, ignore_index=True)


def build_experiment_comparison(experiments_dir: str | Path = "experiments") -> pd.DataFrame:
    """
    Build a comparison table with average and fold-variability metrics.
    """
    summary_df = load_experiment_summaries(experiments_dir)
    fold_df = load_fold_summaries(experiments_dir)

    metric_columns = list(summary_df.select_dtypes(include="number").columns)

    fold_std_df = (
        fold_df
        .groupby("experiment")[metric_columns]
        .std()
        .reset_index()
        .rename(columns={col: f"{col}_std" for col in metric_columns})
    )

    comparison_df = summary_df.merge(fold_std_df, on="experiment", how="left")
    comparison_df = comparison_df.sort_values(
        by=["best_val_f1", "best_val_accuracy"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return comparison_df


def plot_experiment_comparison(
    comparison_df: pd.DataFrame,
    metric: str = "best_val_f1",
    with_error_bars: bool = True,
    figsize: tuple[int, int] = (10, 5),
    label_column: str = "experiment",
):
    """
    Plot a bar chart comparing experiments on a selected metric.
    """
    if metric not in comparison_df.columns:
        raise ValueError(f"Metric '{metric}' not found in comparison dataframe.")
    if label_column not in comparison_df.columns:
        raise ValueError(f"Label column '{label_column}' not found in comparison dataframe.")

    plot_df = comparison_df.sort_values(metric, ascending=False)
    error_col = f"{metric}_std"
    yerr = plot_df[error_col] if with_error_bars and error_col in plot_df.columns else None

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        plot_df[label_column],
        plot_df[metric],
        yerr=yerr,
        capsize=5 if yerr is not None else 0,
        color="#4C956C",
        edgecolor="#2F5233",
    )
    ax.set_title(f"Experiment Comparison by {metric}")
    ax.set_xlabel("Experiment")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    return ax
