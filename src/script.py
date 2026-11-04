"""
check_hmm_explainability.py

Script ad hoc per:
  1) addestrare l'HMM esattamente come nel runner reale
     (stessi parametri di HMMTimeSeriesSegmenter usati in
     RunFullyHmmCnnMambaClassifier)
  2) decodificare gli stati nascosti (Viterbi) per uno o più pazienti
  3) produrre una spiegabilità "semplice" dell'HMM:
       - parametri globali (means_/covars_/transmat_/startprob_) salvati
         in JSON, con etichette delle feature invece di soli indici
       - interpretazione automatica: quale/i feature caratterizzano
         ciascuno stato, rispetto alla media tra tutti gli stati
       - overlay grafico: segnale grezzo colorato in base allo stato HMM

Non tocca nulla del progetto: legge solo i dati (data_root/metadata_path)
e scrive gli output in una nuova cartella "explainability_out/" accanto
allo script.
"""

from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

from src.data.pahaw_io import build_recording_index, load_svc_file, select_features
from src.preprocess.hmm.hmm_segmenter import HMMTimeSeriesSegmenter
from src.run.preprocess.config import _project_path

# ----------------------------------------------------------------------
# CONFIGURAZIONE — adatta questi valori al tuo esperimento di riferimento
# ----------------------------------------------------------------------
DATA_ROOT = _project_path("PaHaW/PaHaW/PaHaW_public")
METADATA_PATH = _project_path("PaHaW/PaHaW/PaHaW_files/corpus_PaHaW.xlsx")

FEATURE_NAMES = ("y", "x")
ENGINEERED_FEATURES = ("pressure_derivative",)

N_STATES = 2
COVARIANCE_TYPE = "diag"
N_ITER = 50
TOL = 1e-2
RANDOM_STATE = 42
ALGORITHM = "viterbi"

# quanti pazienti usare per addestrare l'HMM (None = tutti quelli trovati)
FIT_PATIENT_IDS = None
FIT_TASKS = None

# paziente/i su cui generare la spiegazione visiva (None = primi 3 trovati)
INSPECT_PATIENT_IDS = ["00027"]
INSPECT_TASKS = [1, 8]

OUTPUT_DIR = Path(__file__).resolve().parent / "explainability_out"

LABEL_COLUMN = "Disease"
POSITIVE_VALUE = "PD"


def load_sequences(patient_ids, tasks):
    recordings = build_recording_index(
        data_root=DATA_ROOT,
        metadata_path=METADATA_PATH,
        patient_ids=patient_ids,
        tasks=tasks,
        label_column=LABEL_COLUMN,
        positive_value=POSITIVE_VALUE,
        skip_empty=True,
    )
    feature_columns = list(FEATURE_NAMES) + list(ENGINEERED_FEATURES)

    items = []
    for recording in recordings:
        raw = load_svc_file(recording.file_path)
        features = select_features(raw, feature_columns)
        items.append((recording, features))
    return items, feature_columns


def fit_hmm(fit_items):
    sequences = [features for _, features in fit_items]
    segmenter = HMMTimeSeriesSegmenter(
        n_states=N_STATES,
        covariance_type=COVARIANCE_TYPE,
        n_iter=N_ITER,
        tol=TOL,
        random_state=RANDOM_STATE,
        algorithm=ALGORITHM,
        verbose=False,
    )
    segmenter.fit(sequences)
    return segmenter


def dump_global_parameters(segmenter: HMMTimeSeriesSegmenter, feature_columns, out_path: Path):
    model = segmenter.model
    report = {
        "n_states": segmenter.n_states,
        "covariance_type": model.covariance_type,
        "converged": bool(model.monitor_.converged),
        "n_iter_run": int(model.monitor_.iter),
        "startprob_": model.startprob_.tolist(),
        "transmat_": model.transmat_.tolist(),
        "feature_order": feature_columns,
        "means_per_state": {
            f"state_{i}": dict(zip(feature_columns, model.means_[i].tolist()))
            for i in range(segmenter.n_states)
        },
    }

    if COVARIANCE_TYPE == "diag":
        report["variances_per_state"] = {
            f"state_{i}": dict(zip(feature_columns, model.covars_[i].tolist()))
            for i in range(segmenter.n_states)
        }
    # NB: per covariance_type "full"/"tied"/"spherical" la shape di covars_
    # cambia — questo script è pensato per "diag" (quello che usi tu).

    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] Parametri globali HMM salvati in: {out_path}")
    return report


def interpret_states(report, feature_columns):
    """
    Spiegabilità 'a basso livello': per ogni stato individua le feature che
    si discostano di più dalla media tra tutti gli stati (z-score), per
    dargli un'etichetta descrittiva automatica (es. 'alta velocità').
    """
    n_states = len(report["means_per_state"])
    means_matrix = np.array([
        [report["means_per_state"][f"state_{i}"][feat] for feat in feature_columns]
        for i in range(n_states)
    ])  # [n_states, n_features]

    global_mean = means_matrix.mean(axis=0)
    global_std = means_matrix.std(axis=0) + 1e-8
    z = (means_matrix - global_mean) / global_std

    print("\n=== Interpretazione automatica degli stati HMM ===")
    for i, feat_z in enumerate(z):
        top_idx = np.argsort(np.abs(feat_z))[::-1]
        highlights = [
            f"{feature_columns[j]} ({'alto' if feat_z[j] > 0 else 'basso'}, z={feat_z[j]:+.2f})"
            for j in top_idx[:2]
        ]
        print(f"Stato {i}: caratterizzato da " + ", ".join(highlights))


def plot_state_overlay(recording, sequence, segmenter, feature_columns, out_path: Path):
    states = segmenter.predict_states(sequence)
    n_features = sequence.shape[1]

    fig, axes = plt.subplots(n_features, 1, figsize=(12, 2.2 * n_features), sharex=True)
    if n_features == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10")
    t = np.arange(len(sequence))
    bounds = segmenter.states_to_bounds(states)

    for feat_idx, ax in enumerate(axes):
        ax.plot(t, sequence[:, feat_idx], color="black", linewidth=0.8)
        for start, end, state in bounds:
            ax.axvspan(start, end, color=cmap(state % 10), alpha=0.25)
        ax.set_ylabel(feature_columns[feat_idx])

    axes[-1].set_xlabel("timestep")
    fig.suptitle(
        f"Paziente {recording.patient_id} - task {recording.task_id} "
        f"(label={'PD' if recording.label == 1 else 'HC'}) - stati HMM sovrapposti"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Grafico salvato in: {out_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Carico le sequenze per l'addestramento dell'HMM...")
    fit_items, feature_columns = load_sequences(FIT_PATIENT_IDS, FIT_TASKS)
    if not fit_items:
        raise RuntimeError("Nessuna sequenza trovata. Controlla DATA_ROOT/METADATA_PATH.")
    print(f"    -> {len(fit_items)} sequenze trovate, feature: {feature_columns}")

    print("[2/4] Addestro l'HMM...")
    segmenter = fit_hmm(fit_items)

    print("[3/4] Estraggo i parametri globali e li interpreto...")
    report = dump_global_parameters(segmenter, feature_columns, OUTPUT_DIR / "hmm_parameters.json")
    interpret_states(report, feature_columns)

    print("[4/4] Genero l'overlay stato/segnale per i pazienti selezionati...")
    inspect_items, _ = load_sequences(INSPECT_PATIENT_IDS, INSPECT_TASKS)
    if not inspect_items:
        inspect_items = fit_items[:3]

    for recording, sequence in inspect_items[:3]:
        out_path = OUTPUT_DIR / f"states_{recording.patient_id}_task{recording.task_id}.png"
        plot_state_overlay(recording, sequence, segmenter, feature_columns, out_path)

    print(f"\nFatto. Tutti gli output sono in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
