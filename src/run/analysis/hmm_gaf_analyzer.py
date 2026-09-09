from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import math
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.common.utils import set_determinism
from src.data.pahaw_io import build_recording_index, load_svc_file, select_features
from src.preprocess.hmm.hmm_segmenter import HMMTimeSeriesSegmenter


# ============================================================
# GAF utilities
# ============================================================

def _resample_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    """Linear resampling to a fixed length."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    if x.size == target_len:
        return x
    old_idx = np.linspace(0.0, 1.0, num=x.size)
    new_idx = np.linspace(0.0, 1.0, num=target_len)
    return np.interp(new_idx, old_idx, x).astype(np.float32)


def gramian_angular_field(series: np.ndarray, target_len: int = 128) -> np.ndarray:
    """Compute the Gramian Angular Summation Field (GASF).

    The input is resampled to a fixed length and scaled to [-1, 1].
    Output shape: (target_len, target_len)
    """
    x = _resample_1d(np.asarray(series, dtype=np.float32), target_len=target_len)

    min_v = float(np.min(x))
    max_v = float(np.max(x))
    if math.isclose(max_v, min_v):
        scaled = np.zeros_like(x, dtype=np.float32)
    else:
        scaled = 2.0 * (x - min_v) / (max_v - min_v) - 1.0

    scaled = np.clip(scaled, -1.0, 1.0)
    phi = np.arccos(scaled)
    return np.cos(phi[:, None] + phi[None, :]).astype(np.float32)


def save_gaf_image(gaf: np.ndarray, path: Path, title: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.imshow(gaf, origin="lower", interpolation="nearest")
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.02)
    plt.close()


# ============================================================
# Small helpers for HMM state analysis
# ============================================================

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
        raw = load_svc_file(recording.file_path)
        features = select_features(raw, feature_columns)
        sequences.append(np.asarray(features, dtype=np.float32))
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
        patient_labels.setdefault(recording.patient_id, int(recording.label))

    pids = np.array(sorted(patient_labels.keys()))
    labels = np.array([patient_labels[pid] for pid in pids], dtype=int)
    return pids, labels


def _ensure_state_sequence(segmenter: HMMTimeSeriesSegmenter, sequence: np.ndarray) -> np.ndarray:
    """Try a few common method names to obtain state ids from the HMM segmenter."""
    candidates = ["predict", "segment", "decode", "predict_states", "transform"]
    for name in candidates:
        if hasattr(segmenter, name):
            fn = getattr(segmenter, name)
            try:
                states = fn(sequence)
                if isinstance(states, tuple):
                    states = states[0]
                states = np.asarray(states)
                if states.ndim == 1 and states.size > 0:
                    return states.astype(int)
            except Exception:
                continue
    raise AttributeError(
        "Cannot obtain state sequence from HMMTimeSeriesSegmenter. "
        "Expected one of: predict, segment, decode, predict_states, transform."
    )


def _segments_from_state_sequence(sequence: np.ndarray, states: np.ndarray) -> list[tuple[int, int, int]]:
    """Return (state_id, start, end) intervals over time indices.

    end is exclusive.
    """
    if len(sequence) == 0 or len(states) == 0:
        return []
    if len(sequence) != len(states):
        n = min(len(sequence), len(states))
        sequence = sequence[:n]
        states = states[:n]

    intervals: list[tuple[int, int, int]] = []
    start = 0
    current = int(states[0])
    for i in range(1, len(states)):
        st = int(states[i])
        if st != current:
            intervals.append((current, start, i))
            start = i
            current = st
    intervals.append((current, start, len(states)))
    return intervals


@dataclass
class StateStats:
    n_segments: int = 0
    total_points: int = 0
    total_duration: int = 0
    mean_duration: float = 0.0
    median_duration: float = 0.0


@dataclass
class FoldAnalysisResult:
    fold: int
    n_states: int
    train_patients: int
    val_patients: int
    state_stats: dict[int, dict[str, float]]
    label_state_occupancy: dict[str, dict[int, float]]
    mean_gaf_distance_by_state: dict[int, float]


# ============================================================
# Main analysis class
# ============================================================

class HMMGAFAnalyzer:
    """Analyze HMM states without training a CNN.

    What it saves:
    - HMM state occupancy per class
    - duration / dwell-time summaries
    - representative GAF images per state and per label
    - optional fold-wise CSV / JSON summaries

    The goal is exploratory analysis, not final classification.
    """

    def __init__(
        self,
        data_root: str | Path,
        metadata_path: str | Path,
        patient_ids: Sequence[str] | None = None,
        tasks: Sequence[int] | None = None,
        feature_names: Sequence[str] | None = None,
        engineered_features: Sequence[str] = (),
        n_states: int = 2,
        covariance_type: str = "diag",
        n_iter: int = 50,
        tol: float = 1e-2,
        random_state: int = 42,
        algorithm: str = "viterbi",
        verbose: bool = False,
        gaf_target_len: int = 128,
        max_gaf_per_state: int = 12,
        output_dir: str | Path | None = None,
        seed: int = 42,
        n_splits: int = 5,
        label_column: str = "Disease",
        positive_value: str = "PD",
    ) -> None:
        self.data_root = Path(data_root)
        self.metadata_path = Path(metadata_path)
        self.patient_ids = [str(p).zfill(5) for p in patient_ids] if patient_ids else None
        self.tasks = list(tasks) if tasks else None
        self.feature_names = tuple(feature_names) if feature_names is not None else ("x", "y")
        self.engineered_features = tuple(engineered_features)

        self.n_states = int(n_states)
        self.covariance_type = covariance_type
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.random_state = int(random_state)
        self.algorithm = algorithm
        self.verbose = verbose

        self.gaf_target_len = int(gaf_target_len)
        self.max_gaf_per_state = int(max_gaf_per_state)
        self.output_dir = Path(output_dir) if output_dir else None
        self.seed = int(seed)
        self.n_splits = int(n_splits)
        self.label_column = label_column
        self.positive_value = positive_value

        if self.n_states < 2:
            raise ValueError("n_states must be at least 2")
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")

    @property
    def num_features(self) -> int:
        return len(self.feature_names) + len(self.engineered_features)

    def _split_patient_wise(self, patient_ids: np.ndarray, labels: np.ndarray):
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.seed,
        )
        return skf.split(patient_ids, labels)

    def _fit_hmm_on_sequences(self, sequences: list[np.ndarray]) -> HMMTimeSeriesSegmenter:
        segmenter = HMMTimeSeriesSegmenter(
            n_states=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,
            algorithm=self.algorithm,
            verbose=self.verbose,
        )
        segmenter.fit(sequences)
        return segmenter

    def _analyze_one_sequence(
        self,
        sequence: np.ndarray,
        segmenter: HMMTimeSeriesSegmenter,
        label: int,
        patient_id: str,
        save_gaf_dir: Path | None = None,
    ) -> dict[str, Any]:
        states = _ensure_state_sequence(segmenter, sequence)
        intervals = _segments_from_state_sequence(sequence, states)

        per_state: dict[int, dict[str, Any]] = {
            s: {
                "count": 0,
                "durations": [],
                "segments": [],
            }
            for s in range(self.n_states)
        }

        for state_id, start, end in intervals:
            if state_id not in per_state:
                continue
            seg = sequence[start:end]
            if seg.size == 0:
                continue
            per_state[state_id]["count"] += 1
            per_state[state_id]["durations"].append(end - start)
            per_state[state_id]["segments"].append(seg)

        # GAF summaries per state: we compute GAF for each segment using the first feature
        # and keep a mean pairwise distance to measure within-state consistency.
        state_gaf_samples: dict[int, list[np.ndarray]] = {s: [] for s in range(self.n_states)}
        state_label_occupancy: dict[int, float] = {}

        for state_id in range(self.n_states):
            segments = per_state[state_id]["segments"]
            if len(segments) == 0:
                state_label_occupancy[state_id] = 0.0
                continue

            occupancy = float(sum(len(seg) for seg in segments)) / float(len(sequence))
            state_label_occupancy[state_id] = occupancy

            for seg_idx, seg in enumerate(segments[: self.max_gaf_per_state]):
                # GAF on first feature by default, so that visual comparison remains interpretable.
                series = seg[:, 0]
                gaf = gramian_angular_field(series, target_len=self.gaf_target_len)
                state_gaf_samples[state_id].append(gaf)
                if save_gaf_dir is not None:
                    img_name = f"pid_{patient_id}_label_{label}_state_{state_id}_seg_{seg_idx}.png"
                    save_gaf_image(
                        gaf,
                        save_gaf_dir / f"label_{label}" / f"state_{state_id}" / img_name,
                        title=f"PID {patient_id} | label={label} | state={state_id}",
                    )

        # State-wise summary
        state_summary: dict[int, dict[str, float]] = {}
        for state_id in range(self.n_states):
            durations = np.asarray(per_state[state_id]["durations"], dtype=np.float32)
            gafs = state_gaf_samples[state_id]
            mean_pairwise_dist = float("nan")
            if len(gafs) >= 2:
                flattened = np.stack([g.reshape(-1) for g in gafs], axis=0)
                center = flattened.mean(axis=0, keepdims=True)
                mean_pairwise_dist = float(np.mean(np.linalg.norm(flattened - center, axis=1)))
            elif len(gafs) == 1:
                mean_pairwise_dist = 0.0

            state_summary[state_id] = {
                "n_segments": float(per_state[state_id]["count"]),
                "mean_duration": float(durations.mean()) if durations.size else 0.0,
                "median_duration": float(np.median(durations)) if durations.size else 0.0,
                "total_duration": float(durations.sum()) if durations.size else 0.0,
                "occupancy": float(state_label_occupancy.get(state_id, 0.0)),
                "mean_gaf_distance": mean_pairwise_dist,
                "n_gaf_segments": float(len(gafs)),
            }

        return {
            "patient_id": patient_id,
            "label": int(label),
            "state_summary": state_summary,
            "state_label_occupancy": state_label_occupancy,
            "intervals": intervals,
        }

    def run(self) -> dict[str, Any]:
        """
        Run fold-wise HMM + GAF analysis without a classifier.

        For each fold:
            - fit the HMM only on training patients;
            - analyze only validation patients;
            - save fold-level HMM convergence, segmentation quality,
              occupancy separation and GAF diagnostics.

        State identifiers are only interpreted inside each fold.
        Cross-fold selection uses permutation-invariant metrics.
        """
        set_determinism(self.seed)

        pids, labels = _collect_patient_labels(
            data_root=self.data_root,
            metadata_path=self.metadata_path,
            patient_ids=self.patient_ids,
            tasks=self.tasks,
            label_column=self.label_column,
            positive_value=self.positive_value,
        )

        all_fold_results: list[dict[str, Any]] = []

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        for fold, (train_idx, val_idx) in enumerate(
            self._split_patient_wise(pids, labels),
            1,
        ):
            train_patients = pids[train_idx].tolist()
            val_patients = pids[val_idx].tolist()

            train_sequences = _load_sequences_for_patients(
                data_root=self.data_root,
                metadata_path=self.metadata_path,
                patient_ids=train_patients,
                tasks=self.tasks,
                feature_names=self.feature_names,
                engineered_features=self.engineered_features,
                label_column=self.label_column,
                positive_value=self.positive_value,
            )

            # Fit HMM exclusively on the training portion of this fold.
            hmm = self._fit_hmm_on_sequences(train_sequences)

            monitor_history = list(hmm.model.monitor_.history)

            hmm_fit = {
                "converged": bool(hmm.model.monitor_.converged),
                "iterations": int(hmm.model.monitor_.iter),
                "final_log_likelihood": (
                    float(monitor_history[-1])
                    if monitor_history
                    else float("nan")
                ),
            }

            fold_dir = None
            save_gaf_dir = None

            if self.output_dir is not None:
                fold_dir = self.output_dir / f"fold_{fold:02d}"
                save_gaf_dir = fold_dir / "gaf"

                fold_dir.mkdir(parents=True, exist_ok=True)
                save_gaf_dir.mkdir(parents=True, exist_ok=True)

            fold_patient_reports: list[dict[str, Any]] = []

            # Analyze only validation patients.
            for pid in val_patients:
                recs = build_recording_index(
                    data_root=self.data_root,
                    metadata_path=self.metadata_path,
                    patient_ids=[pid],
                    tasks=self.tasks,
                    label_column=self.label_column,
                    positive_value=self.positive_value,
                    skip_empty=True,
                )

                if not recs:
                    continue

                # This version is intentionally patient-level with one selected task.
                # It is correct for the current notebook configuration: tasks=[8].
                if len(recs) != 1:
                    raise ValueError(
                        "HMMGAFAnalyzer.run expects exactly one selected recording "
                        "per patient. Run one task at a time or implement explicit "
                        "multi-task aggregation."
                    )

                recording = recs[0]

                raw = load_svc_file(recording.file_path)
                features = select_features(
                    raw,
                    list(self.feature_names) + list(self.engineered_features),
                )

                report = self._analyze_one_sequence(
                    sequence=np.asarray(features, dtype=np.float32),
                    segmenter=hmm,
                    label=int(recording.label),
                    patient_id=pid,
                    save_gaf_dir=save_gaf_dir,
                )

                fold_patient_reports.append(report)

            if not fold_patient_reports:
                raise ValueError(
                    f"Fold {fold} does not contain validation reports."
                )

            fold_labels = {
                int(report["label"])
                for report in fold_patient_reports
            }

            if fold_labels != {0, 1}:
                raise ValueError(
                    f"Fold {fold} validation set must contain both classes "
                    "to compute PD/HC occupancy separation."
                )

            fold_state_stats: dict[int, list[dict[str, float]]] = {
                state: []
                for state in range(self.n_states)
            }

            fold_label_occupancy: dict[str, dict[int, list[float]]] = {
                "PD": {
                    state: []
                    for state in range(self.n_states)
                },
                "HC": {
                    state: []
                    for state in range(self.n_states)
                },
            }

            for report in fold_patient_reports:
                label_name = "PD" if int(report["label"]) == 1 else "HC"

                for state in range(self.n_states):
                    state_report = report["state_summary"][state]

                    fold_state_stats[state].append(state_report)
                    fold_label_occupancy[label_name][state].append(
                        float(state_report["occupancy"])
                    )

            state_summary_fold: dict[int, dict[str, float]] = {}
            mean_gaf_distances: dict[int, float] = {}

            for state in range(self.n_states):
                rows = fold_state_stats[state]

                if not rows:
                    state_summary_fold[state] = {
                        "n_segments": 0.0,
                        "mean_duration": 0.0,
                        "median_duration": 0.0,
                        "total_duration": 0.0,
                        "occupancy": 0.0,
                        "mean_gaf_distance": float("nan"),
                    }
                    mean_gaf_distances[state] = float("nan")
                    continue

                gaf_values = np.asarray(
                    [row["mean_gaf_distance"] for row in rows],
                    dtype=np.float64,
                )
                gaf_counts = np.asarray(
                    [row["n_gaf_segments"] for row in rows],
                    dtype=np.float64,
                )

                mask = np.isfinite(gaf_values) & (gaf_counts >= 2)

                if mask.any():
                    mean_gaf_distance = float(
                        np.average(gaf_values[mask], weights=gaf_counts[mask])
                    )
                else:
                    mean_gaf_distance = float("nan")

                state_summary_fold[state] = {
                    "n_segments": float(
                        np.mean([row["n_segments"] for row in rows])
                    ),
                    "mean_duration": float(
                        np.mean([row["mean_duration"] for row in rows])
                    ),
                    "median_duration": float(
                        np.mean([row["median_duration"] for row in rows])
                    ),
                    "total_duration": float(
                        np.mean([row["total_duration"] for row in rows])
                    ),
                    "occupancy": float(
                        np.mean([row["occupancy"] for row in rows])
                    ),
                    "mean_gaf_distance": mean_gaf_distance,
                    "n_gaf_segments": float(np.mean([row["n_gaf_segments"] for row in rows])),
                }

                mean_gaf_distances[state] = mean_gaf_distance

            pd_occupancy = np.asarray(
                [
                    float(np.mean(fold_label_occupancy["PD"][state]))
                    for state in range(self.n_states)
                ],
                dtype=np.float64,
            )

            hc_occupancy = np.asarray(
                [
                    float(np.mean(fold_label_occupancy["HC"][state]))
                    for state in range(self.n_states)
                ],
                dtype=np.float64,
            )

            # Total variation distance between class occupancy distributions.
            # It is invariant to a permutation of the state identifiers and
            # remains comparable when testing different numbers of states.
            occupancy_total_variation = float(
                0.5 * np.abs(pd_occupancy - hc_occupancy).sum()
            )

            segment_lengths = np.asarray(
                [
                    int(end) - int(start)
                    for report in fold_patient_reports
                    for _, start, end in report["intervals"]
                ],
                dtype=np.int64,
            )

            segments_per_patient = np.asarray(
                [
                    len(report["intervals"])
                    for report in fold_patient_reports
                ],
                dtype=np.int64,
            )

            if segment_lengths.size == 0:
                raise ValueError(
                    f"Fold {fold} does not contain HMM segments."
                )

            segmentation_quality = {
                "total_segments": int(segment_lengths.size),
                "mean_segment_length": float(np.mean(segment_lengths)),
                "median_segment_length": float(np.median(segment_lengths)),
                "p90_segment_length": float(np.percentile(segment_lengths, 90)),
                "max_segment_length": int(segment_lengths.max()),
                "fraction_length_one": float(np.mean(segment_lengths == 1)),
                "mean_segments_per_patient": float(np.mean(segments_per_patient)),
                "median_segments_per_patient": float(np.median(segments_per_patient)),
                "max_segments_per_patient": int(segments_per_patient.max()),
            }

            valid_gaf_values = [
                v for v in mean_gaf_distances.values()
                if np.isfinite(v)
            ]

            mean_gaf_distance = (
                float(np.mean(valid_gaf_values))
                if len(valid_gaf_values) > 0
                else float("nan")
            )

            fold_payload = {
                "fold": fold,
                "n_states": self.n_states,
                "train_patients": len(train_patients),
                "val_patients": len(val_patients),
                "validation_only": True,
                "hmm_fit": hmm_fit,
                "occupancy_total_variation": occupancy_total_variation,
                "segmentation_quality": segmentation_quality,
                "state_stats": state_summary_fold,
                "label_state_occupancy": {
                    "PD": {
                        state: float(pd_occupancy[state])
                        for state in range(self.n_states)
                    },
                    "HC": {
                        state: float(hc_occupancy[state])
                        for state in range(self.n_states)
                    },
                },
                "mean_gaf_distance": mean_gaf_distance,
                "mean_gaf_distance_by_state": mean_gaf_distances,
                "patients": fold_patient_reports,
            }

            all_fold_results.append(fold_payload)

            if fold_dir is not None:
                with (fold_dir / "analysis.json").open(
                    "w",
                    encoding="utf-8",
                ) as output_file:
                    json.dump(fold_payload, output_file, indent=2)

                state_rows = []

                for state in range(self.n_states):
                    state_rows.append(
                        {
                            "fold": fold,
                            "state": state,
                            **state_summary_fold[state],
                            "pd_occupancy": float(pd_occupancy[state]),
                            "hc_occupancy": float(hc_occupancy[state]),
                        }
                    )

                pd.DataFrame(state_rows).to_csv(
                    fold_dir / "state_summary.csv",
                    index=False,
                )

        fold_separations = np.asarray(
            [
                fold_result["occupancy_total_variation"]
                for fold_result in all_fold_results
            ],
            dtype=np.float64,
        )

        quality_rows = [
            fold_result["segmentation_quality"]
            for fold_result in all_fold_results
        ]

        non_converged_folds = [
            fold_result["fold"]
            for fold_result in all_fold_results
            if not fold_result["hmm_fit"]["converged"]
        ]

        fold_table = [
            {
                "fold": fold_result["fold"],
                "n_states": fold_result["n_states"],
                "train_patients": fold_result["train_patients"],
                "val_patients": fold_result["val_patients"],
                "occupancy_total_variation": fold_result[
                    "occupancy_total_variation"
                ],
                "converged": fold_result["hmm_fit"]["converged"],
                "iterations": fold_result["hmm_fit"]["iterations"],
                "final_log_likelihood": fold_result["hmm_fit"][
                    "final_log_likelihood"
                ],
                **fold_result["segmentation_quality"],
            }
            for fold_result in all_fold_results
        ]

        summary = {
            "analysis": "HMM_GAF",
            "validation_only": True,
            "data": {
                "tasks": (
                    list(self.tasks)
                    if self.tasks is not None
                    else None
                ),
                "feature_names": list(self.feature_names),
                "engineered_features": list(self.engineered_features),
                "label_column": self.label_column,
                "positive_value": self.positive_value,
            },
            "hmm": {
                "n_states": self.n_states,
                "covariance_type": self.covariance_type,
                "n_iter": self.n_iter,
                "tol": self.tol,
                "algorithm": self.algorithm,
            },
            "selection_metrics": {
                "metric": "occupancy_total_variation",
                "mean_occupancy_total_variation": float(fold_separations.mean()),
                "std_occupancy_total_variation": float(fold_separations.std()),
                "fold_occupancy_total_variation": (fold_separations.tolist()),
                "mean_gaf_distance": float(np.mean([f["mean_gaf_distance"] for f in all_fold_results])),
                "fold_gaf_distance": [f["mean_gaf_distance"] for f in all_fold_results],
            },
            "segmentation_quality": {
                "mean_fraction_length_one": float(
                    np.mean(
                        [
                            row["fraction_length_one"]
                            for row in quality_rows
                        ]
                    )
                ),
                "mean_median_segment_length": float(
                    np.mean(
                        [
                            row["median_segment_length"]
                            for row in quality_rows
                        ]
                    )
                ),
                "mean_median_segments_per_patient": float(
                    np.mean(
                        [
                            row["median_segments_per_patient"]
                            for row in quality_rows
                        ]
                    )
                ),
                "mean_segments_per_patient": float(
                    np.mean(
                        [
                            row["mean_segments_per_patient"]
                            for row in quality_rows
                        ]
                    )
                ),
                "max_segments_per_patient": int(
                    max(
                        row["max_segments_per_patient"]
                        for row in quality_rows
                    )
                ),
            },
            "convergence": {
                "all_folds_converged": len(non_converged_folds) == 0,
                "non_converged_folds": non_converged_folds,
            },
            "fold_table": fold_table,
            "folds": all_fold_results,
        }

        if self.output_dir is not None:
            with (self.output_dir / "summary.json").open(
                "w",
                encoding="utf-8",
            ) as output_file:
                json.dump(summary, output_file, indent=2)

            pd.DataFrame(summary["fold_table"]).to_csv(
                self.output_dir / "summary_table.csv",
                index=False,
            )

        return summary

    def _build_summary(self, all_fold_results: list[dict[str, Any]]) -> dict[str, Any]:
        fold_table = []
        for fold_payload in all_fold_results:
            row = {
                "fold": fold_payload["fold"],
                "n_states": fold_payload["n_states"],
                "train_patients": fold_payload["train_patients"],
                "val_patients": fold_payload["val_patients"],
            }
            for s in range(self.n_states):
                row[f"state{s}_occupancy_PD"] = fold_payload["label_state_occupancy"]["PD"][s]
                row[f"state{s}_occupancy_HC"] = fold_payload["label_state_occupancy"]["HC"][s]
                row[f"state{s}_mean_duration"] = fold_payload["state_stats"][s]["mean_duration"]
                row[f"state{s}_mean_gaf_distance"] = fold_payload["state_stats"][s]["mean_gaf_distance"]
            fold_table.append(row)

        averages: dict[str, float] = {}
        for s in range(self.n_states):
            averages[f"state{s}_occupancy_PD"] = float(np.mean([f["label_state_occupancy"]["PD"][s] for f in all_fold_results]))
            averages[f"state{s}_occupancy_HC"] = float(np.mean([f["label_state_occupancy"]["HC"][s] for f in all_fold_results]))
            averages[f"state{s}_mean_duration"] = float(np.mean([f["state_stats"][s]["mean_duration"] for f in all_fold_results]))
            averages[f"state{s}_mean_gaf_distance"] = float(np.nanmean([f["state_stats"][s]["mean_gaf_distance"] for f in all_fold_results]))

        return {
            "experiment": "HMM_GAF_analysis",
            "data": {
                "features": {
                    "base": list(self.feature_names),
                    "engineered": list(self.engineered_features),
                },
                "tasks": list(self.tasks) if self.tasks is not None else None,
                "patient_ids": list(self.patient_ids) if self.patient_ids is not None else None,
            },
            "hmm": {
                "n_states": self.n_states,
                "covariance_type": self.covariance_type,
                "n_iter": self.n_iter,
                "tol": self.tol,
                "random_state": self.random_state,
                "algorithm": self.algorithm,
            },
            "gaf": {
                "target_len": self.gaf_target_len,
                "max_gaf_per_state": self.max_gaf_per_state,
            },
            "folds": all_fold_results,
            "fold_table": fold_table,
            "averages": averages,
        }

