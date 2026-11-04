"""
Runner per la spiegazione con Grad-CAM delle decisioni a livello di paziente.

I risultati salvati degli esperimenti contengono solo metriche, storie di
training e parametri di normalizzazione, non i pesi delle reti. Il runner
riaddestra quindi ciascun modello con lo stesso runner, la stessa
configurazione, lo stesso seed e lo stesso dispositivo dell'esperimento
originale (CPU o GPU), verifica fold per fold che le metriche dell'ultima
epoca coincidano con quelle salvate e salva i pesi, così da non doverli più
riaddestrare. Le spiegazioni si riferiscono quindi esattamente ai modelli che
hanno prodotto i risultati riportati nella tesi.

Modalità di validazione supportate:
- "full":       registrazioni complete (RunHmmCnnClassifier, RunSegmentCNN);
- "segmented":  segmenti dell'HMM con media delle pre-attivazioni
                (RunFullyHmmCnnClassifier).

In entrambi i casi la decisione del paziente è la media dei logit delle
"unità" che lo compongono (registrazioni o segmenti); la CAM di ciascuna unità
viene pesata 1/N, cioè per il suo contributo alla media.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.pahaw_io import build_recording_index, load_svc_file, select_features
from src.run.model.run_fully_hmm_cnn_classifier import RunFullyHmmCnnClassifier
from src.run.model.run_hmm_cnn_classifier import RunHmmCnnClassifier
from src.run.model.run_segment_cnn_classifier import RunSegmentCNN
from src.run.preprocess.run_preprocessing_timeseries import (
    TimeSeriesDataConfig,
    build_timeseries_dataset,
)


# =========================================================
# CONTESTO DI UN FOLD
# =========================================================

@dataclass
class FoldContext:
    fold: int
    model: nn.Module
    val_dataset: Any
    mean: torch.Tensor
    std: torch.Tensor                    # già limitata inferiormente da eps
    device: torch.device
    validation_mode: str                 # "full" | "segmented"
    hmm: Any = None                      # HMM del fold (None per la finestra fissa)
    hmm_features: tuple[str, ...] = ()   # feature date all'HMM
    last_metrics: dict[str, float] = field(default_factory=dict)


class _StopAfterFold(Exception):
    """Interrompe la cross-validation dopo l'ultimo fold richiesto."""


class _CaptureMixin:
    """
    Si inserisce prima del runner nella MRO: non modifica l'addestramento,
    ma conserva il modello creato in _build_training_components e,
    al termine di _fit, il contesto del fold.
    """

    validation_mode = "full"

    def _init_capture(self, stop_after_fold: int | None) -> None:
        self.fold_contexts_: dict[int, FoldContext] = {}
        self._stop_after_fold = stop_after_fold
        self._last_model: nn.Module | None = None
        self._last_hmm = None

    def _build_training_components(self):
        model, optimizer, criterion = super()._build_training_components()
        self._last_model = model
        return model, optimizer, criterion

    def _hmm_columns(self) -> tuple[str, ...]:
        return ()

    def _store_context(self, fold, payload, val_dataset, normalization) -> None:
        model = self._last_model
        model.eval()

        eps = float(self.normalization_eps)
        mean = torch.tensor(normalization["mean"], dtype=torch.float32)
        std = torch.tensor(normalization["std"], dtype=torch.float32).clamp_min(eps)

        self.fold_contexts_[fold] = FoldContext(
            fold=fold,
            model=model,
            val_dataset=val_dataset,
            mean=mean,
            std=std,
            device=self.device,
            validation_mode=self.validation_mode,
            hmm=self._last_hmm,
            hmm_features=self._hmm_columns(),
            # metriche di validazione all'ultima epoca
            last_metrics={
                k[len("val_"):]: float(v)
                for k, v in payload.items()
                if k.startswith("val_")
            },
        )

        if self._stop_after_fold is not None and fold >= self._stop_after_fold:
            raise _StopAfterFold()


class _HmmCaptureMixin(_CaptureMixin):

    def _build_hmm_segment_dataset(self, patient_ids, hmm_segmenter):
        self._last_hmm = hmm_segmenter
        return super()._build_hmm_segment_dataset(patient_ids, hmm_segmenter)

    def _hmm_columns(self) -> tuple[str, ...]:
        return tuple(self.hmm_feature_names) + tuple(self.hmm_engineered_features)

    def _fit(self, *args, **kwargs):
        payload = super()._fit(*args, **kwargs)
        self._store_context(
            fold=kwargs["fold"],
            payload=payload,
            val_dataset=kwargs["validation_dataset"],
            normalization=kwargs["normalization_payload"],
        )
        return payload


class CaptureHmmCnnClassifier(_HmmCaptureMixin, RunHmmCnnClassifier):
    """Step 6: HMM sul training set, validation set completo."""

    validation_mode = "full"

    def __init__(self, *args, stop_after_fold: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_capture(stop_after_fold)


class CaptureFullyHmmCnnClassifier(_HmmCaptureMixin, RunFullyHmmCnnClassifier):
    """Step 7: HMM sul training set e sul validation set."""

    validation_mode = "segmented"

    def __init__(self, *args, stop_after_fold: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_capture(stop_after_fold)


class CaptureSegmentCNN(_CaptureMixin, RunSegmentCNN):
    """Step 0: finestra fissa sul training set, validation set completo."""

    validation_mode = "full"

    def __init__(self, *args, stop_after_fold: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_capture(stop_after_fold)

    def _fit(self, *args, **kwargs):
        payload = super()._fit(*args, **kwargs)

        # stesse operazioni eseguite all'interno del runner (nessuna casualità)
        train_ds = kwargs["training_dataset"].subset_by_patients(kwargs["train_patients"])
        val_ds = kwargs["validation_dataset"].subset_by_patients(kwargs["val_patients"])
        normalization = self._compute_normalization_stats(train_ds)

        self._store_context(
            fold=kwargs["fold"],
            payload=payload,
            val_dataset=val_ds,
            normalization=normalization,
        )
        return payload


# =========================================================
# GRAD-CAM 1D
# =========================================================

class GradCAM1D:
    """
    Grad-CAM per la VanillaCNN1D: i pesi dei canali sono la media temporale
    del gradiente del logit della classe (prima del softmax) rispetto alle
    attivazioni del layer scelto; la mappa viene interpolata linearmente
    alla lunghezza dell'input.
    """

    def __init__(self, model: nn.Module, target_layer: str = "features.8"):
        self.model = model
        modules = dict(model.named_modules())
        if target_layer not in modules:
            raise KeyError(f"Layer '{target_layer}' non trovato nel modello.")

        # le ReLU in-place impediscono di leggere attivazioni e gradienti
        for m in model.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

        self._activation = None
        self._gradient = None
        self._handle = modules[target_layer].register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self._activation = output
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self._gradient = grad

    def remove(self) -> None:
        self._handle.remove()

    def __call__(self, x: torch.Tensor, mask: torch.Tensor | None = None,
                 target_class: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """x: (1, T, F) già normalizzato. Restituisce (cam grezza [T], logit [C])."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            logits = self.model(x, mask=mask)
            cls = int(logits[0].argmax()) if target_class is None else int(target_class)
            logits[0, cls].backward()

        A = self._activation.detach()[0]          # (C, T')
        G = self._gradient.detach()[0]            # (C, T')
        weights = G.mean(dim=1)                   # (C,)
        cam = F.relu((weights[:, None] * A).sum(dim=0))
        cam = F.interpolate(cam[None, None], size=x.shape[1],
                            mode="linear", align_corners=False)[0, 0]

        return cam.cpu().numpy(), logits.detach()[0].cpu().numpy()


# =========================================================
# UNITÀ DI UN PAZIENTE (registrazioni o segmenti)
# =========================================================

def _val_patient_ids(ctx: FoldContext) -> list[str]:
    return sorted({str(ctx.val_dataset[i]["patient_id"]).zfill(5) for i in range(len(ctx.val_dataset))})


def _patient_units(ctx: FoldContext, patient_id: str, with_states: bool = True) -> tuple[list[dict], int]:
    ds = ctx.val_dataset
    pid = str(patient_id).zfill(5)
    units = []
    label = None

    for i in range(len(ds)):
        s = ds[i]
        if str(s["patient_id"]).zfill(5) != pid:
            continue
        label = int(s["label"])
        task = int(s["task_id"])

        if ctx.validation_mode == "segmented":
            mask = s["mask"].bool()
            units.append(dict(
                task=task, start=int(s["start"]), x_raw=s["segment"].float(),
                mask=mask, valid=int(mask.sum()),
                states=np.full(int(mask.sum()), int(s["state"])),
            ))
        else:
            x_raw = s["sequence"].float()
            states = None
            if with_states and ctx.hmm is not None and ctx.hmm_features:
                raw = load_svc_file(s["file_path"])
                states = np.asarray(ctx.hmm.predict_states(
                    select_features(raw, list(ctx.hmm_features))))
            units.append(dict(
                task=task, start=0, x_raw=x_raw, mask=None,
                valid=int(x_raw.shape[0]), states=states,
            ))

    if not units:
        raise RuntimeError(f"Paziente {pid} non presente nel validation set del fold {ctx.fold}.")

    units.sort(key=lambda u: (u["task"], u["start"]))
    return units, label


def _normalize(ctx: FoldContext, u: dict) -> torch.Tensor:
    x = (u["x_raw"] - ctx.mean) / ctx.std
    if u["mask"] is not None:
        x = x * u["mask"].unsqueeze(-1)          # il padding resta nullo, come nel runner
    return x


def _logits(ctx: FoldContext, u: dict) -> np.ndarray:
    x = _normalize(ctx, u).unsqueeze(0).to(ctx.device)
    mask = None if u["mask"] is None else u["mask"].unsqueeze(0).to(ctx.device)
    with torch.no_grad():
        z = ctx.model(x, mask=mask)
    return z[0].cpu().numpy()


def _patient_cam(ctx: FoldContext, units: list[dict], target_class: int | str | None,
                 label: int, target_layer: str) -> dict:
    """
    Decisione del paziente e mappa Grad-CAM sull'intera registrazione
    (per lo step 7: mappe dei segmenti pesate 1/N e affiancate nel tempo).
    """
    for u in units:
        u["z"] = _logits(ctx, u)

    Z = np.stack([u["z"] for u in units])
    N = len(units)
    patient_logits = Z.mean(axis=0)
    pred = int(patient_logits.argmax())

    if target_class is None or target_class == "pred":
        cls = pred
    elif target_class == "true":
        cls = int(label)
    else:
        cls = int(target_class)
    other = 1 - cls

    gradcam = GradCAM1D(ctx.model, target_layer=target_layer)
    try:
        for u in units:
            x = _normalize(ctx, u).unsqueeze(0).to(ctx.device)
            mask = None if u["mask"] is None else u["mask"].unsqueeze(0).to(ctx.device)
            cam, _ = gradcam(x, mask=mask, target_class=cls)
            u["cam"] = (cam / N)[: u["valid"]]
            u["contrib_margin"] = float((u["z"][cls] - u["z"][other]) / N)
    finally:
        gradcam.remove()

    return dict(units=units, N=N, logits=patient_logits, pred=pred, cls=cls, other=other)


# =========================================================
# DECISIONI SU TUTTI I PAZIENTI DEL FOLD
# =========================================================

def patient_decisions(ctx: FoldContext, class_names: Sequence[str]) -> pd.DataFrame:
    rows = []
    for pid in _val_patient_ids(ctx):
        units, label = _patient_units(ctx, pid, with_states=False)
        z = np.stack([_logits(ctx, u) for u in units]).mean(axis=0)
        pred = int(z.argmax())
        rows.append(dict(fold=ctx.fold, patient_id=pid, label=class_names[label], pred=class_names[pred],
                         correct=pred == label, n_units=len(units),
                         logit_0=float(z[0]), logit_1=float(z[1]),
                         margin=float(z[pred] - z[1 - pred])))
    return pd.DataFrame(rows)


# =========================================================
# SPIEGAZIONE DELLA DECISIONE DI UN PAZIENTE (analisi qualitativa)
# =========================================================

def explain_decision(
    ctx: FoldContext,
    patient_id: str,
    class_names: Sequence[str],
    feature_names_plot: Sequence[str],
    model_name: str = "",
    target_class: int | str | None = None,
    target_layer: str = "features.8",
    save_dir: str | Path | None = None,
    show: bool = True,
):
    units, label = _patient_units(ctx, patient_id)
    pid = str(patient_id).zfill(5)
    res = _patient_cam(ctx, units, target_class, label, target_layer)
    N, pred, cls, other = res["N"], res["pred"], res["cls"], res["other"]
    tasks_present = sorted({u["task"] for u in units})

    print(f"\n[{model_name}] Paziente {pid} | fold {ctx.fold} | unità N={N} (task {tasks_present})")
    print(f"label vera     : {class_names[label]}")
    print(f"logit medi     : {np.round(res['logits'], 4)}  ->  decisione: {class_names[pred]}"
          f"  ({'corretta' if pred == label else 'ERRATA'})")
    print(f"classe spiegata: {class_names[cls]}")

    gmax = max(float(u["cam"].max()) for u in units)
    for u in units:
        u["cam_n"] = u["cam"] / gmax if gmax > 0 else u["cam"]        # normalizzazione globale sul paziente

    table = pd.DataFrame([{
        "task": u["task"], "start": u["start"], "valid": u["valid"],
        "state": (int(u["states"][0]) if ctx.validation_mode == "segmented" else None),
        "logit_0": u["z"][0], "logit_1": u["z"][1],
        "contrib_margin": u["contrib_margin"], "cam_max": float(u["cam_n"].max()),
    } for u in units])
    print(table.round(4).to_string(index=False))
    print("Somma contributi al margine:", round(float(table["contrib_margin"].sum()), 4),
          f"(>0 => decisione per {class_names[cls]})")

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(save_dir / f"patient_{pid}_units.csv", index=False)

    cmap = plt.get_cmap("tab10")
    n_features = units[0]["x_raw"].shape[1]

    for task in tasks_present:
        us = [u for u in units if u["task"] == task]
        signal = np.concatenate([u["x_raw"][: u["valid"]].numpy() for u in us], axis=0)
        cam = np.concatenate([u["cam_n"] for u in us])
        has_states = all(u["states"] is not None for u in us)
        states = np.concatenate([u["states"][: u["valid"]] for u in us]) if has_states else None
        t = np.arange(len(cam))

        fig, axes = plt.subplots(n_features + 1, 1, figsize=(14, 2.2 * (n_features + 1)), sharex=True)

        if states is not None:
            for a in axes:
                start = 0
                for k in range(1, len(states) + 1):
                    if k == len(states) or states[k] != states[start]:
                        a.axvspan(start, k, color=cmap(int(states[start])), alpha=0.10, linewidth=0)
                        start = k

        if ctx.validation_mode == "segmented":
            edges = np.cumsum([u["valid"] for u in us])[:-1]
            for a in axes:
                for e in edges:
                    a.axvline(e, color="k", linewidth=0.4, alpha=0.3)

        for j in range(n_features):
            axes[j].plot(t, signal[:, j], color="C0", linewidth=1.0)
            axes[j].set_ylabel(feature_names_plot[j])
            axes[j].grid(True, alpha=0.3)

        axes[0].set_title(f"{model_name} | paziente {pid} → {class_names[pred]} (vera: {class_names[label]}) | "
                          f"task {task} | Grad-CAM sulla classe {class_names[cls]}")
        axes[-1].plot(t, cam, color="C3", linewidth=1.5)
        axes[-1].fill_between(t, 0, cam, color="C3", alpha=0.35)
        axes[-1].set_ylabel("Grad-CAM")
        axes[-1].set_xlabel("Campione della registrazione")
        axes[-1].set_ylim(-0.05, 1.05)
        axes[-1].grid(True, alpha=0.3)
        plt.tight_layout()

        if save_dir is not None:
            fig.savefig(save_dir / f"patient_{pid}_task{task}_gradcam.png", dpi=150)
        plt.show() if show else plt.close(fig)

    if N > 1:
        fig = plt.figure(figsize=(12, 3.2))
        colors = [cmap(int(u["states"][0])) if ctx.validation_mode == "segmented" else "C0" for u in units]
        plt.bar(np.arange(N), [u["contrib_margin"] for u in units], color=colors)
        plt.axhline(0, color="k", linewidth=0.8)
        plt.xlabel("Segmento (ordinato per task e tempo; colore = stato HMM)")
        plt.ylabel(f"(z[{class_names[cls]}] − z[{class_names[other]}]) / N")
        plt.title(f"{model_name} | contributo dei segmenti alla decisione del paziente {pid}")
        plt.tight_layout()
        if save_dir is not None:
            fig.savefig(save_dir / f"patient_{pid}_contributi.png", dpi=150)
        plt.show() if show else plt.close(fig)

    return units, table


# =========================================================
# ANALISI QUANTITATIVA
# =========================================================

def _run_lengths(states: np.ndarray) -> np.ndarray:
    """Per ogni campione, la lunghezza del tratto consecutivo nello stesso stato."""
    out = np.empty(len(states), dtype=int)
    start = 0
    for k in range(1, len(states) + 1):
        if k == len(states) or states[k] != states[start]:
            out[start:k] = k - start
            start = k
    return out


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    kernel = np.ones(2 * radius + 1)
    return np.convolve(mask.astype(float), kernel, mode="same") > 0


def _variance_order(hmm) -> dict[int, str]:
    """Etichetta gli stati dell'HMM per varianza di emissione (gli indici cambiano tra i fold)."""
    covars = np.asarray(hmm.model.covars_)
    var = [float(np.mean(np.diag(np.atleast_2d(c)))) for c in covars]
    hi = int(np.argmax(var))
    return {k: ("alta" if k == hi else "bassa") for k in range(len(var))}


def _distribution_metrics(r: np.ndarray, annot: dict, short_len: int) -> dict[str, float]:
    """
    r: rilevanza (>= 0) sui T campioni. Restituisce, per ciascuna regione,
    la quota di rilevanza, la quota di tempo e il loro rapporto (arricchimento:
    > 1 se la regione riceve più rilevanza di quanta ne spetterebbe in base
    alla sua durata).
    """
    out: dict[str, float] = {}
    total = float(r.sum())
    T = len(r)
    if total <= 0:
        return {"rilevanza_nulla": 1.0}
    p = r / total

    for name, mask in annot["regions"].items():
        time_share = float(mask.mean())
        mass = float(p[mask].sum())
        out[f"quota_rilevanza_{name}"] = mass
        out[f"quota_tempo_{name}"] = time_share
        out[f"arricchimento_{name}"] = mass / time_share if time_share > 0 else np.nan

    # concentrazione: frazione minima di campioni che contiene il 50% della rilevanza
    cum = np.cumsum(np.sort(p)[::-1])
    out["frazione_campioni_50pct"] = float((np.searchsorted(cum, 0.5) + 1) / T)
    out["rilevanza_nulla"] = 0.0
    return out


# =========================================================
# CARICAMENTO DI MODELLI SALVATI (step 0, finestra fissa)
# =========================================================

def load_saved_contexts(spec: "ModelSpec", folds: Sequence[int] | None = None) -> dict[int, FoldContext]:
    """
    Ricostruisce i contesti dei fold a partire dai pesi salvati, senza
    riaddestrare. Supportato per le configurazioni con validation set
    completo e senza HMM (RunSegmentCNN, step 0).
    """
    if not issubclass(spec.runner_cls, RunSegmentCNN):
        raise NotImplementedError("Il caricamento è implementato per lo step 0 (RunSegmentCNN).")

    load_dir = Path(spec.load_dir)
    kwargs = dict(spec.kwargs)
    kwargs["output_dir"] = None
    runner = RunSegmentCNN(**kwargs)

    val_full = build_timeseries_dataset(TimeSeriesDataConfig(
        data_root=runner.data_root,
        metadata_path=runner.metadata_path,
        patient_ids=runner.patient_ids,
        tasks=runner.tasks,
        feature_names=runner.feature_names,
        engineered_features=runner.engineered_features,
        cache_files=True,
    ))

    ctxs: dict[int, FoldContext] = {}
    for meta_path in sorted(load_dir.glob("fold_*_normalization.json")):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        fold = int(meta["fold"])
        if folds is not None and fold not in folds:
            continue
        if "val_patient_ids" not in meta:
            raise KeyError(f"{meta_path.name}: mancano i pazienti di validazione (salvare con la versione aggiornata).")

        model = runner._build_training_components()[0]
        state = torch.load(load_dir / f"fold_{fold}.pt", map_location=runner.device)
        model.load_state_dict(state)
        model.eval()

        ctxs[fold] = FoldContext(
            fold=fold,
            model=model,
            val_dataset=val_full.subset_by_patients(meta["val_patient_ids"]),
            mean=torch.tensor(meta["mean"], dtype=torch.float32),
            std=torch.tensor(meta["std_clamped"], dtype=torch.float32),
            device=runner.device,
            validation_mode="full",
        )
        print(f"[{spec.name}] fold {fold}: pesi caricati da {load_dir} "
              f"(salvati con torch {meta.get('torch_version', '?')} su {meta.get('device', '?')})")
    if not ctxs:
        raise FileNotFoundError(f"Nessun modello trovato in {load_dir}")
    return ctxs


def evaluate_context(ctx: FoldContext) -> dict[str, float]:
    """Accuracy, AUC e loss sul validation set del fold, come nel runner."""
    from sklearn.metrics import roc_auc_score

    logits, labels = [], []
    for pid in _val_patient_ids(ctx):
        units, label = _patient_units(ctx, pid, with_states=False)
        logits.append(np.stack([_logits(ctx, u) for u in units]).mean(axis=0))
        labels.append(label)
    z = torch.tensor(np.stack(logits), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    prob = torch.softmax(z, dim=1)[:, 1].numpy()
    return {
        "accuracy": float((z.argmax(dim=1) == y).float().mean()),
        "auc": float(roc_auc_score(y.numpy(), prob)) if len(set(labels)) > 1 else float("nan"),
        "loss": float(F.cross_entropy(z, y)),
    }


class RunGradCamExplainer:
    """
    Riaddestra i modelli indicati, verifica che coincidano con gli esperimenti
    originali, ne salva i pesi e ne analizza le decisioni con Grad-CAM.
    """

    def __init__(
        self,
        specs: Sequence["ModelSpec"],
        folds: Sequence[int] | None = None,          # None = tutti i fold
        class_names: Sequence[str] = ("Sano", "PD"),
        feature_names_plot: Sequence[str] = ("y", "speed", "velocity", "pressure derivative"),
        target_layer: str = "features.8",
        output_dir: str | Path = "gradcam",
        show: bool = True,
    ):
        self.specs = list(specs)
        self.folds = None if folds is None else sorted(int(f) for f in folds)
        self.class_names = tuple(class_names)
        self.feature_names_plot = tuple(feature_names_plot)
        self.target_layer = target_layer
        self.output_dir = Path(output_dir)
        self.show = show
        self.contexts_: dict[str, dict[int, FoldContext]] = {}
        self.reference_check_: pd.DataFrame | None = None

    # ---------------------------------------------------------
    def fit(self, save_models: bool = True) -> "RunGradCamExplainer":
        checks = []
        for spec in self.specs:
            print("\n" + "#" * 80)
            print(f"# {spec.name}: {spec.description} | dispositivo: {spec.kwargs.get('device')}")
            print("#" * 80)

            if spec.load_dir is not None:
                # pesi salvati in precedenza: nessun riaddestramento
                ctxs = load_saved_contexts(spec, self.folds)
                for ctx in ctxs.values():
                    ctx.last_metrics = evaluate_context(ctx)
                self.contexts_[spec.name] = ctxs
                for fold, ctx in ctxs.items():
                    checks.append(self._check_reference(spec, ctx))
                continue

            kwargs = dict(spec.kwargs)
            kwargs["output_dir"] = None            # non sovrascrive i risultati originali
            stop = None if self.folds is None else max(self.folds)
            runner = spec.runner_cls(stop_after_fold=stop, **kwargs)

            try:
                runner.run()
            except _StopAfterFold:
                pass

            ctxs = {f: c for f, c in runner.fold_contexts_.items()
                    if self.folds is None or f in self.folds}
            self.contexts_[spec.name] = ctxs

            for fold, ctx in ctxs.items():
                checks.append(self._check_reference(spec, ctx))
                if save_models:
                    self._save_model(spec, ctx)

        self.reference_check_ = pd.DataFrame([c for c in checks if c is not None])
        if not self.reference_check_.empty:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.reference_check_.to_csv(self.output_dir / "controllo_riproducibilita.csv", index=False)
            print("\n=== Controllo rispetto agli esperimenti originali ===")
            print(self.reference_check_.round(4).to_string(index=False))
            if not self.reference_check_["coincide"].all():
                print("\n[ATTENZIONE] Alcuni modelli riaddestrati NON coincidono con quelli degli esperimenti: "
                      "verificare dispositivo (CPU/GPU) e configurazione prima di usarne le spiegazioni.")
        return self

    def _check_reference(self, spec: "ModelSpec", ctx: FoldContext) -> dict | None:
        ref = spec.reference.get(ctx.fold) if spec.reference else None
        if ref is None:
            return None
        row = {"modello": spec.name, "fold": ctx.fold}
        same = True
        for m, ref_v in ref.items():
            new_v = ctx.last_metrics.get(m, np.nan)
            row[f"{m}_originale"] = ref_v
            row[f"{m}_riaddestrato"] = new_v
            same &= bool(abs(ref_v - new_v) < 5e-4)        # valori di riferimento arrotondati a 4 cifre
        row["coincide"] = same
        return row

    def _save_model(self, spec: "ModelSpec", ctx: FoldContext) -> None:
        d = self.output_dir / "modelli" / spec.name
        d.mkdir(parents=True, exist_ok=True)
        torch.save(ctx.model.state_dict(), d / f"fold_{ctx.fold}.pt")
        with open(d / f"fold_{ctx.fold}_normalization.json", "w", encoding="utf-8") as f:
            json.dump({
                "fold": ctx.fold,
                "mean": ctx.mean.tolist(),
                "std_clamped": ctx.std.tolist(),
                "val_patient_ids": _val_patient_ids(ctx),
                "last_metrics": ctx.last_metrics,
                "torch_version": torch.__version__,
                "device": str(ctx.device),
            }, f, indent=2)

    # ---------------------------------------------------------
    def decisions(self) -> pd.DataFrame:
        """Decisione di ciascun modello su tutti i pazienti di validazione."""
        merged = None
        for spec in self.specs:
            df = pd.concat([patient_decisions(ctx, self.class_names)
                            for ctx in self.contexts_[spec.name].values()], ignore_index=True)
            df = df[["fold", "patient_id", "label", "pred", "margin"]].rename(
                columns={"pred": f"pred_{spec.name}", "margin": f"margin_{spec.name}"})
            merged = df if merged is None else merged.merge(df, on=["fold", "patient_id", "label"])
        preds = merged[[f"pred_{s.name}" for s in self.specs]]
        merged["concordi"] = preds.nunique(axis=1) == 1
        return merged

    def explain(self, patient_id: str, fold: int, target_class: int | str | None = None):
        tables = {}
        for spec in self.specs:
            _, table = explain_decision(
                ctx=self.contexts_[spec.name][fold],
                patient_id=patient_id,
                class_names=self.class_names,
                feature_names_plot=self.feature_names_plot,
                model_name=f"{spec.name} ({spec.description})",
                target_class=target_class,
                target_layer=self.target_layer,
                save_dir=self.output_dir / "qualitativa" / spec.name,
                show=self.show,
            )
            tables[spec.name] = table
        return tables

    # ---------------------------------------------------------
    def _annotation(self, fold: int, file_path: Path, T: int, short_len: int,
                    abrupt_quantile: float, abrupt_radius: int) -> dict:
        """
        Annotazioni comuni a tutti i modelli per la registrazione di un paziente:
        stati dell'HMM del fold (etichettati per varianza), tratti di penna
        sollevata (pressione nulla), variazioni brusche della pressione,
        tratti brevi dell'HMM.
        """
        hmm_ctx = next((c[fold] for c in self.contexts_.values() if c[fold].hmm is not None), None)
        raw = load_svc_file(file_path)
        pressure = select_features(raw, ["pressure"])[:, 0]
        dpress = select_features(raw, ["pressure_derivative"])[:, 0]
        n = min(T, len(pressure))

        regions: dict[str, np.ndarray] = {}
        regions["penna_sollevata"] = (pressure[:n] <= 0)
        regions["penna_appoggiata"] = ~regions["penna_sollevata"]

        thr = np.quantile(np.abs(dpress[:n]), abrupt_quantile)
        regions["variazioni_brusche_pressione"] = _dilate(np.abs(dpress[:n]) >= thr, abrupt_radius)

        states = None
        if hmm_ctx is not None:
            states = np.asarray(hmm_ctx.hmm.predict_states(
                select_features(raw, list(hmm_ctx.hmm_features))))[:n]
            names = _variance_order(hmm_ctx.hmm)
            for k, name in names.items():
                regions[f"stato_varianza_{name}"] = states == k
            regions[f"tratti_brevi_lt{short_len}"] = _run_lengths(states) < short_len

        return dict(n=n, regions=regions, states=states)

    def quantitative(
        self,
        target_class: int | str | None = "pred",
        short_len: int = 32,
        abrupt_quantile: float = 0.95,
        abrupt_radius: int = 5,
        save_maps: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Calcola la mappa Grad-CAM di ogni paziente di validazione per ogni
        modello e fold e ne misura la distribuzione rispetto a regioni della
        registrazione definite in modo indipendente dal modello.
        """
        recordings = {}
        spec0 = self.specs[0].kwargs
        for rec in build_recording_index(
            data_root=spec0["data_root"], metadata_path=spec0["metadata_path"],
            tasks=spec0.get("tasks"), skip_empty=True,
        ):
            recordings[str(rec.patient_id).zfill(5)] = rec.file_path

        rows, corr_rows, maps = [], [], {}
        folds = sorted(next(iter(self.contexts_.values())).keys())

        for fold in folds:
            pids = _val_patient_ids(next(iter(self.contexts_.values()))[fold])
            print(f"\n[Analisi quantitativa] fold {fold}: {len(pids)} pazienti")

            for pid in pids:
                patient_maps = {}
                annot = None
                for spec in self.specs:
                    ctx = self.contexts_[spec.name][fold]
                    units, label = _patient_units(ctx, pid, with_states=False)
                    res = _patient_cam(ctx, units, target_class, label, self.target_layer)
                    r = np.concatenate([u["cam"] for u in res["units"]])

                    if annot is None:
                        annot = self._annotation(fold, recordings[pid], len(r), short_len,
                                                 abrupt_quantile, abrupt_radius)
                    n = annot["n"]
                    if len(r) != n:
                        print(f"  [avviso] {spec.name} paziente {pid}: lunghezza mappa {len(r)} ≠ {n}, troncata")
                    r = r[:n] if len(r) >= n else np.pad(r, (0, n - len(r)))
                    patient_maps[spec.name] = r

                    row = dict(modello=spec.name, fold=fold, patient_id=pid,
                               label=self.class_names[label], pred=self.class_names[res["pred"]],
                               corretto=res["pred"] == label, classe_spiegata=self.class_names[res["cls"]],
                               margine=float(res["logits"][res["pred"]] - res["logits"][1 - res["pred"]]),
                               lunghezza=n)
                    row.update(_distribution_metrics(r, annot, short_len))
                    rows.append(row)

                names = list(patient_maps)
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        a, b = patient_maps[names[i]], patient_maps[names[j]]
                        c = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
                        corr_rows.append(dict(fold=fold, patient_id=pid, coppia=f"{names[i]}-{names[j]}",
                                              correlazione=c))
                maps[f"fold{fold}_{pid}"] = patient_maps

        per_patient = pd.DataFrame(rows)
        correlations = pd.DataFrame(corr_rows)

        enrich_cols = [c for c in per_patient.columns if c.startswith("arricchimento_")]
        summary = (per_patient.groupby("modello", sort=False)[enrich_cols + ["frazione_campioni_50pct"]]
                   .agg(["mean", "std", "median"]))
        by_outcome = (per_patient.groupby(["modello", "corretto"], sort=False)[enrich_cols]
                      .mean())
        by_class = (per_patient.groupby(["modello", "label"], sort=False)[enrich_cols]
                    .mean())
        corr_summary = correlations.groupby("coppia")["correlazione"].agg(["mean", "std", "median", "count"])

        out = self.output_dir / "quantitativa"
        out.mkdir(parents=True, exist_ok=True)
        per_patient.to_csv(out / "metriche_per_paziente.csv", index=False)
        summary.to_csv(out / "riepilogo_per_modello.csv")
        by_outcome.to_csv(out / "riepilogo_per_esito.csv")
        by_class.to_csv(out / "riepilogo_per_classe.csv")
        correlations.to_csv(out / "correlazioni_tra_modelli.csv", index=False)
        corr_summary.to_csv(out / "riepilogo_correlazioni.csv")
        if save_maps:
            np.savez_compressed(out / "mappe_gradcam.npz",
                                **{f"{k}_{m}": v for k, d in maps.items() for m, v in d.items()})

        print("\n=== Arricchimento della rilevanza (media sui pazienti; 1 = proporzionale alla durata) ===")
        print(per_patient.groupby("modello", sort=False)[enrich_cols].mean().round(3).T.to_string())
        print("\n=== Frazione di campioni che contiene il 50% della rilevanza (media) ===")
        print(per_patient.groupby("modello", sort=False)["frazione_campioni_50pct"].mean().round(3).to_string())
        print("\n=== Correlazione tra le mappe dei modelli (stesso paziente) ===")
        print(corr_summary.round(3).to_string())

        self._plot_quantitative(per_patient, enrich_cols, out)

        return dict(per_patient=per_patient, summary=summary, by_outcome=by_outcome,
                    by_class=by_class, correlations=correlations, corr_summary=corr_summary)

    def _plot_quantitative(self, per_patient: pd.DataFrame, enrich_cols: list[str], out: Path) -> None:
        models = list(dict.fromkeys(per_patient["modello"]))
        n = len(enrich_cols)
        fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4), sharey=False)
        axes = np.atleast_1d(axes)
        for ax, col in zip(axes, enrich_cols):
            data = [per_patient.loc[per_patient["modello"] == m, col].dropna().values for m in models]
            ax.boxplot(data, showfliers=False)
            ax.set_xticks(range(1, len(models) + 1))
            ax.set_xticklabels(models)
            ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8)
            ax.set_title(col.replace("arricchimento_", "").replace("_", " "), fontsize=9)
            ax.tick_params(axis="x", labelsize=8)
            ax.grid(True, axis="y", alpha=0.3)
        axes[0].set_ylabel("arricchimento della rilevanza")
        plt.tight_layout()
        fig.savefig(out / "arricchimento_per_regione.png", dpi=150)
        plt.show() if self.show else plt.close(fig)

    # ---------------------------------------------------------
    def zip_results(self, base_name: str | Path) -> str:
        return shutil.make_archive(
            base_name=str(base_name),
            format="zip",
            root_dir=str(self.output_dir.parent),
            base_dir=self.output_dir.name,
        )


@dataclass
class ModelSpec:
    name: str                                          # es. "step6"
    description: str
    runner_cls: type
    kwargs: dict[str, Any]
    reference: dict[int, dict[str, float]] | None = None   # metriche originali per fold (ultima epoca)
    load_dir: str | Path | None = None   # cartella con fold_k.pt e fold_k_normalization.json: carica invece di addestrare
