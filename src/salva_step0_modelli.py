import os
import sys
import shutil

# =========================================================
# STEP 0: salvataggio dei modelli per la spiegabilità
#
# Riaddestra lo step 0 (finestra fissa sul training set, validation set
# completo) con ESATTAMENTE la configurazione dell'esperimento originale
# e salva, per ciascun fold, i pesi della rete all'ultima epoca, i parametri
# di normalizzazione e i pazienti di validazione.
#
# Va eseguito NELLO STESSO AMBIENTE in cui è stato eseguito l'esperimento
# originale (stessa macchina, CPU, stessa versione di PyTorch): solo così
# le reti coincidono con quelle che hanno prodotto i risultati della tesi.
# Il controllo finale lo verifica fold per fold.
#
# Output: <OUT_DIR>/modelli/step0/fold_k.pt e fold_k_normalization.json,
#         <OUT_DIR>/controllo_riproducibilita.csv, log.txt e uno zip
#         da caricare su Kaggle come dataset.
# =========================================================

import torch

from run.preprocess.config import _project_path
from run.model.run_gradcam_explainer import (
    RunGradCamExplainer,
    ModelSpec,
    CaptureSegmentCNN,
)

OUT_DIR = os.path.abspath("step0_modelli")

shutil.rmtree(OUT_DIR, ignore_errors=True)
os.makedirs(OUT_DIR, exist_ok=True)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


_log_file = open(os.path.join(OUT_DIR, "log.txt"), "w", encoding="utf-8")
_stdout = sys.stdout
sys.stdout = _Tee(_stdout, _log_file)

print(f"PyTorch {torch.__version__} | CUDA disponibile: {torch.cuda.is_available()} | dispositivo usato: cpu")


# configurazione identica alla cella dell'esperimento originale
# (notebook "raw segment + cnn(train_seg - val_no)", Task_8/y_speed_velocity_pressure_derivative)
step0_kwargs = dict(
    data_root=_project_path("PaHaW/PaHaW/PaHaW_public"),
    metadata_path=_project_path("PaHaW/PaHaW/PaHaW_files/corpus_PaHaW.xlsx"),
    tasks=[8],
    feature_names=("y",),
    engineered_features=("speed", "velocity", "pressure_derivative"),
    window_size=256,
    epochs=50,
    batch_size=16,
    learning_rate=1e-3,
    regularization=1e-3,
    n_splits=5,
    normalize=True,
    device="cpu",
    aggregation="mean",
    training_cfg={
        "metrics": ["accuracy", "f1", "auc", "precision", "recall", "specificity"],
        "monitor_metric": "auc",
        "early_stopping_patience": float("inf"),
        "min_delta": 0.0,
        "results_writer": "csv",
        "save_metrics": True,
    },
)

# metriche dell'ultima epoca salvate in experiments/ant/.../fold_summary.csv
REF_STEP0 = {1: (0.8667, 0.8571, 0.8178), 2: (0.8667, 0.9286, 0.5286), 3: (0.6667, 0.7857, 1.4472),
             4: (0.8000, 0.7500, 0.9008), 5: (0.7333, 0.7500, 0.9133)}

spec = ModelSpec(
    name="step0",
    description="finestra fissa, val completo",
    runner_cls=CaptureSegmentCNN,
    kwargs=step0_kwargs,
    reference={f: {"accuracy": a, "auc": u, "loss": l} for f, (a, u, l) in REF_STEP0.items()},
)

try:
    explainer = RunGradCamExplainer(specs=[spec], output_dir=OUT_DIR, show=False).fit(save_models=True)

    if explainer.reference_check_["coincide"].all():
        print("\nOK: tutte le reti coincidono con quelle dell'esperimento originale. "
              "Carica lo zip su Kaggle e usa la cartella modelli/step0 come STEP0_MODELS_DIR.")
    else:
        print("\n[ATTENZIONE] Le reti NON coincidono con l'esperimento originale: questo ambiente è "
              "diverso da quello in cui è stato eseguito lo step 0, e le spiegazioni non si "
              "riferirebbero ai modelli della tesi.")
finally:
    sys.stdout = _stdout
    _log_file.close()
    zip_path = shutil.make_archive(
        base_name=OUT_DIR,
        format="zip",
        root_dir=os.path.dirname(OUT_DIR),
        base_dir=os.path.basename(OUT_DIR),
    )
    print(f"\nArchivio creato: {zip_path} ({os.path.getsize(zip_path) / 1024:.1f} KB)")
