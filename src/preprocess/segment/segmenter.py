from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.data.pahaw_io import (
    ENGINEERED_FEATURES,
    FEATURE,
    FEATURE_ALIASES,
    compute_engineered_feature,
    load_svc_file,
    normalise_feature_name,
    select_features,
)


_normalise_feature_name = normalise_feature_name

__all__ = [
    "ENGINEERED_FEATURES",
    "FEATURE",
    "FEATURE_ALIASES",
    "SegmentBounds",
    "aggregate_patient_logits",
    "compute_engineered_feature",
    "load_svc_file",
    "normalise_feature_name",
    "pad_segment",
    "patient_prediction_from_logits",
    "segment_bounds",
    "segment_timeseries",
    "select_features",
]


@dataclass(frozen=True, slots=True)
class SegmentBounds:
    """Container for a fixed-size time-series segment."""

    start: int
    end: int

def segment_bounds(
    n_samples: int,             # total data length
    window_size: int,           # window size
    stride: int | None = None,  # how much the window moves each time
    include_last: bool = True,  # includes the last window, even if overlapping
) -> list[SegmentBounds]:
    """Return fixed-size windows for a sequence of length ``n_samples``.

    The last segment is optionally included even when it overlaps with the
    previous one, so that the tail of the sequence is not lost.
    """

    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    stride = window_size if stride is None else stride
    if stride <= 0:
        raise ValueError("stride must be positive")

    if n_samples == 0:
        return []

    if n_samples <= window_size:
        return [SegmentBounds(0, window_size)] if include_last else []

    starts = list(range(0, n_samples - window_size + 1, stride))
    if include_last:
        last_start = n_samples - window_size
        if not starts or starts[-1] != last_start:
            starts.append(last_start)

    # Remove duplicates while preserving order.
    starts = list(dict.fromkeys(starts))
    return [SegmentBounds(start, start + window_size) for start in starts]


def pad_segment(segment: np.ndarray, target_length: int):

    segment = np.asarray(segment, dtype=np.float32)

    if segment.ndim == 1:
        segment = segment[:, None]

    original_length = min(segment.shape[0], target_length)

    mask = np.zeros(target_length, dtype=np.float32)
    mask[:original_length] = 1.0

    if segment.shape[0] >= target_length:
        return segment[:target_length], mask

    pad_width = target_length - segment.shape[0]

    pad_block = np.zeros(
        (pad_width, segment.shape[1]),
        dtype=np.float32,
    )

    padded = np.concatenate([segment, pad_block], axis=0)

    return padded, mask

def segment_timeseries(
    signal: np.ndarray,         # time series
    window_size: int,           # Number of point in each windows
    stride: int | None = None,  # How much move the windows each time
    include_last: bool = True,  # Include the last windows
    pad_mode: str = "edge",     # How to fill windows
    flatten: bool = False,      # Whether to flatten each window into a 1D vector
) -> np.ndarray:
    """Split a 1D or 2D time series into fixed-size windows.

    Parameters
    ----------
    signal:
        Array with shape ``(n_samples,)`` or ``(n_samples, n_features)``.
    window_size:
        Number of samples per segment.
    stride:
        Step between consecutive windows. Defaults to ``window_size``.
    include_last:
        Whether to append a final window aligned with the tail of the signal.
    pad_mode:
        Padding strategy for incomplete windows (``edge`` or ``constant``).
    flatten:
        If ``True``, each segment is flattened into a 1D vector.
    """

    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim == 1:
        signal = signal[:, None] # 2D transformation
    if signal.ndim != 2:
        raise ValueError("signal must be 1D or 2D")

    bounds = segment_bounds(
        n_samples=signal.shape[0],
        window_size=window_size,
        stride=stride,
        include_last=include_last,
    )
    if not bounds:
        if flatten:
            return np.empty((0, window_size * signal.shape[1]), dtype=np.float32)
        return np.empty((0, window_size, signal.shape[1]), dtype=np.float32)

    segments = []
    for bound in bounds:
        segment = signal[bound.start : bound.end]
        segment = pad_segment(segment, window_size, mode=pad_mode)
        if flatten:
            segments.append(segment.reshape(-1))
        else:
            segments.append(segment)

    return np.asarray(segments, dtype=np.float32)


def aggregate_patient_logits(
    logits: np.ndarray,
    patient_ids: Sequence[str],
    method: str = "mean",
) -> dict[str, np.ndarray]:
    """Aggregate segment-level logits at patient level.

    Parameters
    ----------
    logits:
        Array with shape ``(n_segments, n_classes)``.
    patient_ids:
        Sequence mapping each segment to the corresponding patient.
    method:
        Aggregation strategy. Currently supports ``mean`` and ``median``.
    """

    logits = np.asarray(logits, dtype=np.float32)
    patient_ids = list(patient_ids)

    if logits.ndim != 2:
        raise ValueError("logits must have shape (n_segments, n_classes)")
    if len(logits) != len(patient_ids):
        raise ValueError("logits and patient_ids must have the same length")

    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for logit, patient_id in zip(logits, patient_ids, strict=True):
        grouped[str(patient_id)].append(np.asarray(logit, dtype=np.float32))

    aggregated: dict[str, np.ndarray] = {}
    for patient_id, values in grouped.items():
        stacked = np.stack(values, axis=0)
        if method in {"mean", "average"}:
            aggregated[patient_id] = stacked.mean(axis=0)
        elif method == "median":
            aggregated[patient_id] = np.median(stacked, axis=0)
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")

    return aggregated


def patient_prediction_from_logits(
    logits: np.ndarray,
    patient_ids: Sequence[str],
    method: str = "mean",
) -> dict[str, int]:
    """Convert segment logits into patient-level class predictions."""

    aggregated = aggregate_patient_logits(logits=logits, patient_ids=patient_ids, method=method)
    return {patient_id: int(np.argmax(values)) for patient_id, values in aggregated.items()}
