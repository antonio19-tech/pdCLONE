from dataclasses import dataclass
from typing import Sequence

import numpy as np
from hmmlearn.hmm import GaussianHMM
from src.data.pahaw_sequence_dataset import PaHaWSequenceDataset

@dataclass(frozen=True)
class SegmentChunk:
    start: int
    end: int
    state: int
    segment: np.ndarray
    mask: np.ndarray


class HMMTimeSeriesSegmenter:

    def __init__(
        self,
        # numero di stati nascosti dell'HMM
        n_states: int,
        # struttura della matrice di covarianza delle distribuzioni Gaussiane
        # associate agli stati dell'HMM
        covariance_type: str = "diag",
        # numero massimo di iterazioni consentite durante l'addestramento
        n_iter: int = 200,
        # soglia di tolleranza utilizzata per determinare la convergenza:
        # l'addestramento si arresta quando il miglioramento della log-likelihood
        # diventa sufficientemente piccolo
        tol: float = 1e-2,
        random_state: int | None = 42,
        # algoritmo utilizzato per determinare la sequenza più probabile
        # degli stati nascosti date le osservazioni
        algorithm: str = "viterbi",
        verbose: bool = False,
    ) -> None:

        if n_states <= 0:
            raise ValueError("n_states must be positive")

        self.n_states = n_states

        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            random_state=random_state,
            algorithm=algorithm,
            verbose=verbose,
        )

        self._is_fitted = False

    @staticmethod
    def _as_2d(sequence: np.ndarray) -> np.ndarray:

        seq = np.asarray(sequence, dtype=np.float64)

        if seq.ndim == 1:
            seq = seq[:, None]

        if seq.ndim != 2:
            raise ValueError(
                f"Expected shape (T, F), got {seq.shape}"
            )

        return seq

    @staticmethod
    def _stack_sequences(
        sequences: Sequence[np.ndarray] | np.ndarray,
        lengths: Sequence[int] | None = None,
    ) -> tuple[np.ndarray, list[int]]:

        if isinstance(sequences, np.ndarray):

            X = HMMTimeSeriesSegmenter._as_2d(sequences)

            if lengths is None:
                return X, [len(X)]
            lengths_list = [int(x) for x in lengths]

            if sum(lengths_list) != len(X):
                raise ValueError(
                    "sum(lengths) must match len(X)"
                )
            return X, lengths_list

        seqs = [
            HMMTimeSeriesSegmenter._as_2d(s)
            for s in sequences
        ]
        if not seqs:
            raise ValueError("Empty sequence list")
        n_features = seqs[0].shape[1]
        for i, s in enumerate(seqs):
            if s.shape[1] != n_features:
                raise ValueError(
                    f"Inconsistent feature count "
                    f"(seq0={n_features}, seq{i}={s.shape[1]})"
                )
        X = np.concatenate(seqs, axis=0)
        lengths_list = [len(s) for s in seqs]
        return X, lengths_list

    def fit(
        self,
        sequences: Sequence[np.ndarray] | np.ndarray,
        lengths: Sequence[int] | None = None,
    ):
        X, lengths_list = self._stack_sequences(
            sequences,
            lengths,
        )
        self.model.fit(X, lengths=lengths_list)
        self._is_fitted = True
        return self

    def score(self, sequence: np.ndarray) -> float:

        self._ensure_fitted()
        X = self._as_2d(sequence)
        return float(self.model.score(X))

    def predict_states(
        self,
        sequence: np.ndarray,
    ) -> np.ndarray:

        self._ensure_fitted()
        X = self._as_2d(sequence)
        return self.model.predict(X)

    def predict_proba(
        self,
        sequence: np.ndarray,
    ) -> np.ndarray:

        self._ensure_fitted()
        X = self._as_2d(sequence)
        return self.model.predict_proba(X)

    def decode(
        self,
        sequence: np.ndarray,
    ) -> tuple[float, np.ndarray]:

        self._ensure_fitted()
        X = self._as_2d(sequence)
        return self.model.decode(X)

    @staticmethod
    def states_to_bounds(
        states: np.ndarray,
    ) -> list[tuple[int, int, int]]:

        states = np.asarray(states, dtype=np.int64)

        if states.ndim != 1:
            raise ValueError("states must be 1D")
        if len(states) == 0:
            return []
        bounds = []

        start = 0
        current = int(states[0])

        for i in range(1, len(states)):

            s = int(states[i])

            if s != current:

                bounds.append(
                    (start, i, current)
                )

                start = i
                current = s

        bounds.append(
            (start, len(states), current)
        )

        return bounds

    @staticmethod
    def pad_timeseries(
        segment: np.ndarray,
        target_length: int,
        pad_value: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:

        if target_length <= 0:
            raise ValueError(
                "target_length must be positive"
            )

        seg = np.asarray(segment, dtype=np.float32)

        if seg.ndim == 1:
            seg = seg[:, None]

        if seg.ndim != 2:
            raise ValueError(
                f"Expected shape (T, F), got {seg.shape}"
            )

        real_len = min(len(seg), target_length)

        mask = np.zeros(
            target_length,
            dtype=np.float32,
        )

        mask[:real_len] = 1.0

        if len(seg) >= target_length:

            return (
                seg[:target_length].copy(),
                mask,
            )

        pad_len = target_length - len(seg)

        pad_block = np.full(
            (pad_len, seg.shape[1]),
            fill_value=pad_value,
            dtype=np.float32,
        )

        padded = np.concatenate(
            [seg, pad_block],
            axis=0,
        )

        return padded, mask

    def segment_sequence(
        self,
        sequence: np.ndarray,
        pad_to: int | None = None,
        pad_value: float = 0.0,
    ) -> list[SegmentChunk]:

        states = self.predict_states(sequence)
        bounds = self.states_to_bounds(states)
        x = self._as_2d(sequence)

        chunks = []

        for start, end, state in bounds:
            if pad_to is None:
                segment = x[start:end].copy()
                mask = np.ones(len(segment), dtype=np.float32)

                chunks.append(
                    SegmentChunk(
                        start=start,
                        end=end,
                        state=state,
                        segment=segment,
                        mask=mask,
                    )
                )
                continue

            for chunk_start in range(start, end, pad_to):
                chunk_end = min(chunk_start + pad_to, end)
                raw_chunk = x[chunk_start:chunk_end]

                segment, mask = self.pad_timeseries(
                    raw_chunk,
                    target_length=pad_to,
                    pad_value=pad_value,
                )

                chunks.append(
                    SegmentChunk(
                        start=chunk_start,
                        end=chunk_end,
                        state=state,
                        segment=segment,
                        mask=mask,
                    )
                )

        return chunks

    def _ensure_fitted(self):

        if not self._is_fitted:
            raise RuntimeError(
                "HMM not fitted"
            )

    def fit_from_dataset(
        self,
        dataset,
    ):

        sequences = []

        for idx in range(len(dataset)):
            sequence, _, _ = dataset[idx]

            sequences.append(
                sequence.numpy()
            )

        self.fit(sequences)

        return self

    def segment_dataset(
        self,
        dataset,
        pad_to: int | None = None,
        pad_value: float = 0.0,
    ):

        all_segments = []

        for idx in range(len(dataset)):
            sequence, label, length = dataset[idx]

            chunks = self.segment_sequence(
                sequence=sequence.numpy(),
                pad_to=pad_to,
                pad_value=pad_value,
            )

            sample_payload = {
                "label": label,
                "length": length,
                "segments": chunks,
            }

            all_segments.append(sample_payload)

        return all_segments
