"""
Attribuzione della decisione alle singole feature in input alla CNN 1D.

Complementa l'analisi Grad-CAM (run_gradcam_explainer.py), che indica in quali
istanti si concentra la rilevanza ma non quale feature la determina.

Due analisi, sugli stessi modelli (stessi pesi) dell'analisi Grad-CAM:

1) Integrated Gradients (Sundararajan et al., 2017) rispetto al margine della
   classe predetta, z_c - z_c', con baseline nulla nello spazio normalizzato
   (ovvero la media di training di ciascuna feature). Si ottiene una matrice
   di attribuzione A[t, d] (campione t, feature d) che soddisfa la completezza:
   sum_{t,d} A[t, d] = margine(x) - margine(baseline).
   Per ciascun paziente vengono calcolate:
     - la quota di attribuzione di ciascuna feature sull'intera serie;
     - la quota di ciascuna feature dentro e fuori ciascuna regione
       (le stesse regioni dell'analisi Grad-CAM);
     - l'arricchimento di ciascuna feature in ciascuna regione.
   L'integrale lungo il percorso viene calcolato con punti più fitti vicino
   alla baseline (alpha = s^3), dove si concentra il gradiente.
   Poiché IG moltiplica il gradiente per il valore dell'input, viene calcolata
   anche la composizione della sola sensibilità |gradiente|, che non favorisce
   i campioni nei quali una feature assume valori elevati (come la pressure
   derivative nelle variazioni brusche).

2) Occlusione delle feature: il canale di una feature viene posto a zero
   (= media di training) e si misura l'effetto sulla decisione:
     - sull'intera serie: accuracy, AUC e margine medio;
     - soltanto all'interno di una regione (es. variazioni brusche), confrontato
       con l'occlusione della stessa feature in una regione di controllo con la
       stessa struttura (la maschera della regione traslata circolarmente).

Step 7: la decisione è la media dei logit dei segmenti, per cui le
attribuzioni di ciascun segmento vengono pesate 1/N e affiancate nel tempo,
come per Grad-CAM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from src.data.pahaw_io import build_recording_index
from src.run.model.run_gradcam_explainer import (
    FoldContext,
    RunGradCamExplainer,
    _normalize,
    _patient_units,
    _val_patient_ids,
)


# =========================================================
# UTILITÀ
# =========================================================

def _disable_inplace(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """p-value corretti con il metodo di Holm."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    order = np.argsort(q)
    m = len(q)
    adj = np.empty(m)
    running = 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * q[k]))
        adj[k] = running
    out[ok] = adj
    return out


def _safe_wilcoxon(x: np.ndarray, y: np.ndarray | None = None) -> float:
    d = np.asarray(x, float) - (0.0 if y is None else np.asarray(y, float))
    d = d[~np.isnan(d)]
    if len(d) < 5 or np.allclose(d, 0):
        return np.nan
    return float(wilcoxon(d).pvalue)


class _PatientModel:
    """Decisione di un paziente: media dei logit delle sue unità (serie o segmenti)."""

    def __init__(self, ctx: FoldContext, units: list[dict]):
        self.ctx = ctx
        self.units = units
        self.N = len(units)
        self.X = [_normalize(ctx, u) for u in units]                       # (L, F) ciascuna
        self.masks = [u["mask"] for u in units]
        self.valid = [u["valid"] for u in units]
        self.offsets = np.concatenate([[0], np.cumsum(self.valid)[:-1]]).astype(int)
        self.T = int(sum(self.valid))
        same = len({tuple(x.shape) for x in self.X}) == 1
        self._stackable = same and all(m is not None for m in self.masks) or self.N == 1

    def logits(self, X: list[torch.Tensor] | None = None) -> np.ndarray:
        X = self.X if X is None else X
        dev = self.ctx.device
        with torch.no_grad():
            if self._stackable:
                xb = torch.stack(X).to(dev)
                mb = None if self.masks[0] is None else torch.stack(self.masks).to(dev)
                z = self.ctx.model(xb, mask=mb)
            else:
                z = torch.cat([self.ctx.model(x[None].to(dev),
                                              mask=None if m is None else m[None].to(dev))
                               for x, m in zip(X, self.masks)])
        return z.mean(dim=0).cpu().numpy()

    # ---------------------------------------------------------
    def occlude(self, feature: int | None, region: np.ndarray | None) -> list[torch.Tensor]:
        """Pone a zero (= media di training) la feature indicata (None = tutte)
        nei campioni della regione (None = intera serie)."""
        out = []
        for x, off, v in zip(self.X, self.offsets, self.valid):
            x = x.clone()
            sel = slice(None) if feature is None else feature
            if region is None:
                x[:v, sel] = 0.0
            else:
                r = torch.as_tensor(region[off:off + v], dtype=torch.bool)
                idx = torch.nonzero(r).squeeze(1)
                if len(idx):
                    if feature is None:
                        x[idx, :] = 0.0
                    else:
                        x[idx, feature] = 0.0
            out.append(x)
        return out

    # ---------------------------------------------------------
    def integrated_gradients(self, cls: int, steps: int = 128, power: float = 3.0,
                             batch: int = 32) -> tuple[np.ndarray, float, float]:
        """
        Attribuzione del margine z_cls - z_other del paziente a ciascun campione e
        feature (baseline nulla). Restituisce A (T, F), margine(x), margine(baseline).

        L'integrale lungo il percorso alpha in [0, 1] viene approssimato con il
        cambio di variabile alpha = s^power (s uniforme, regola del punto medio),
        che infittisce i punti vicino alla baseline: con la group normalization
        la rete è quasi invariante alla scala dell'input, per cui il gradiente
        lungo il percorso si concentra per alpha piccoli.
        """
        dev = self.ctx.device
        model = self.ctx.model
        other = 1 - cls
        s = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
        alphas = s ** power
        weights = power * s ** (power - 1) / steps          # d alpha = power s^(power-1) ds
        parts = []

        if self.N > 1 and self._stackable:
            # segmenti della stessa lunghezza (step 7): tutti insieme, un valore di alpha alla volta
            X = torch.stack(self.X)                                           # (N, L, F)
            M = torch.stack(self.masks).to(dev)
            grad_sum = torch.zeros_like(X)
            for a, w in zip(alphas, weights):
                xb = (a * X).to(dev).requires_grad_(True)
                with torch.enable_grad():
                    z = model(xb, mask=M)
                    margin = (z[:, cls] - z[:, other]).sum()
                    (g,) = torch.autograd.grad(margin, xb)
                grad_sum += w * g.cpu()
            A_all = X * grad_sum / self.N                                    # contributo alla media
            parts = [A_all[i, :v].numpy() for i, v in enumerate(self.valid)]
        else:
            for x, m, v in zip(self.X, self.masks, self.valid):
                grad_sum = torch.zeros_like(x)
                for a_chunk, w_chunk in zip(alphas.split(batch), weights.split(batch)):
                    xb = (a_chunk[:, None, None] * x[None]).to(dev).requires_grad_(True)
                    mb = None if m is None else m[None].expand(len(a_chunk), -1).to(dev)
                    with torch.enable_grad():
                        z = model(xb, mask=mb)
                        margin = (z[:, cls] - z[:, other]).sum()
                        (g,) = torch.autograd.grad(margin, xb)
                    grad_sum += (w_chunk[:, None, None] * g.cpu()).sum(dim=0)
                A = x * grad_sum / self.N
                parts.append(A[:v].numpy())

        A = np.concatenate(parts, axis=0)
        z_x = self.logits()
        z_0 = self.logits([torch.zeros_like(x) for x in self.X])
        return A, float(z_x[cls] - z_x[other]), float(z_0[cls] - z_0[other])

    # ---------------------------------------------------------
    def saliency(self, cls: int) -> np.ndarray:
        """
        Sensibilità del margine z_cls - z_other del paziente a ciascun campione e
        feature, |d margine / d x|, calcolata sulla serie stessa. A differenza di
        Integrated Gradients non è moltiplicata per il valore dell'input, per cui
        non favorisce i campioni nei quali una feature assume valori elevati.
        """
        dev = self.ctx.device
        model = self.ctx.model
        other = 1 - cls
        parts = []
        if self.N > 1 and self._stackable:
            X = torch.stack(self.X).to(dev).requires_grad_(True)
            M = torch.stack(self.masks).to(dev)
            with torch.enable_grad():
                z = model(X, mask=M)
                (g,) = torch.autograd.grad((z[:, cls] - z[:, other]).sum(), X)
            g = g.abs().cpu() / self.N
            parts = [g[i, :v].numpy() for i, v in enumerate(self.valid)]
        else:
            for x, m, v in zip(self.X, self.masks, self.valid):
                xb = x[None].to(dev).requires_grad_(True)
                mb = None if m is None else m[None].to(dev)
                with torch.enable_grad():
                    z = model(xb, mask=mb)
                    (g,) = torch.autograd.grad((z[0, cls] - z[0, other]), xb)
                parts.append((g[0].abs().cpu() / self.N)[:v].numpy())
        return np.concatenate(parts, axis=0)


def _fit_length(a: np.ndarray, n: int) -> np.ndarray:
    if len(a) >= n:
        return a[:n]
    pad = [(0, n - len(a))] + [(0, 0)] * (a.ndim - 1)
    return np.pad(a, pad)


# =========================================================
# RUNNER
# =========================================================

class RunFeatureAttribution(RunGradCamExplainer):
    """
    Stessi modelli dell'analisi Grad-CAM (fit identico: riaddestramento
    verificato o caricamento dei pesi salvati), con attribuzione per feature.
    """

    def _recordings(self) -> dict[str, Path]:
        spec0 = self.specs[0].kwargs
        return {
            str(rec.patient_id).zfill(5): rec.file_path
            for rec in build_recording_index(
                data_root=spec0["data_root"], metadata_path=spec0["metadata_path"],
                tasks=spec0.get("tasks"), skip_empty=True,
            )
        }

    def _feat_keys(self) -> list[str]:
        return [f.replace(" ", "_") for f in self.feature_names_plot]

    # ---------------------------------------------------------
    def attribution(
        self,
        short_len: int = 32,
        abrupt_quantile: float = 0.95,
        abrupt_radius: int = 5,
        ig_steps: int = 128,
        ig_power: float = 3.0,
        occlusion_regions: Sequence[str] = ("variazioni_brusche_pressione", "penna_sollevata"),
        n_controls: int = 10,
        seed: int = 42,
    ) -> dict[str, pd.DataFrame]:
        feats = self._feat_keys()
        F_ = len(feats)
        recordings = self._recordings()
        rng = np.random.default_rng(seed)

        ig_rows, occ_rows, reg_rows = [], [], []
        folds = sorted(next(iter(self.contexts_.values())).keys())

        for fold in folds:
            pids = _val_patient_ids(next(iter(self.contexts_.values()))[fold])
            print(f"\n[Attribuzione per feature] fold {fold}: {len(pids)} pazienti")

            for pid in pids:
                annot = None
                for spec in self.specs:
                    ctx = self.contexts_[spec.name][fold]
                    _disable_inplace(ctx.model)
                    ctx.model.eval()
                    units, label = _patient_units(ctx, pid, with_states=False)
                    pm = _PatientModel(ctx, units)

                    z = pm.logits()
                    pred = int(z.argmax())
                    cls, other = pred, 1 - pred
                    margin0 = float(z[cls] - z[other])

                    if annot is None:
                        annot = self._annotation(fold, recordings[pid], pm.T, short_len,
                                                 abrupt_quantile, abrupt_radius)
                    n = annot["n"]
                    regions = {k: _fit_length(v, pm.T) for k, v in annot["regions"].items()}

                    base = dict(modello=spec.name, fold=fold, patient_id=pid,
                                label=self.class_names[label], pred=self.class_names[pred],
                                corretto=pred == label)

                    # ---------- 1) Integrated Gradients ----------
                    A, m_x, m_0 = pm.integrated_gradients(cls, steps=ig_steps, power=ig_power)
                    err = float(A.sum() - (m_x - m_0))                        # completezza, prima del taglio
                    A = _fit_length(A, n)
                    absA = np.abs(A)
                    tot = absA.sum()
                    row = dict(base, margine=m_x, margine_baseline=m_0,
                               errore_completezza=err,
                               errore_completezza_rel=float(abs(err) / (abs(m_x - m_0) + 1e-12)))
                    per_feat = absA.sum(axis=0)
                    for d, f in enumerate(feats):
                        row[f"quota_{f}"] = float(per_feat[d] / tot) if tot > 0 else np.nan
                        row[f"contributo_{f}"] = float(A[:, d].sum())
                    for rname, rmask in annot["regions"].items():
                        rmask = rmask[:n]
                        if rmask.sum() == 0 or (~rmask).sum() == 0:
                            continue
                        inside = absA[rmask].sum(axis=0)
                        outside = absA[~rmask].sum(axis=0)
                        dur = float(rmask.mean())
                        for d, f in enumerate(feats):
                            row[f"quota_{f}_dentro_{rname}"] = float(inside[d] / inside.sum()) if inside.sum() > 0 else np.nan
                            row[f"quota_{f}_fuori_{rname}"] = float(outside[d] / outside.sum()) if outside.sum() > 0 else np.nan
                            row[f"arricchimento_{f}_{rname}"] = (
                                float(inside[d] / per_feat[d] / dur) if per_feat[d] > 0 else np.nan)
                            row[f"contributo_{f}_dentro_{rname}"] = float(A[rmask, d].sum())
                    # sensibilità |gradiente| (non pesata per il valore dell'input)
                    S = _fit_length(pm.saliency(cls), n)
                    for rname, rmask in annot["regions"].items():
                        rmask = rmask[:n]
                        if rmask.sum() == 0 or (~rmask).sum() == 0:
                            continue
                        si = S[rmask].sum(axis=0)
                        so = S[~rmask].sum(axis=0)
                        for d, f in enumerate(feats):
                            row[f"sens_{f}_dentro_{rname}"] = float(si[d] / si.sum()) if si.sum() > 0 else np.nan
                            row[f"sens_{f}_fuori_{rname}"] = float(so[d] / so.sum()) if so.sum() > 0 else np.nan
                    ig_rows.append(row)

                    # ---------- 2a) occlusione sull'intera serie ----------
                    occ_rows.append(dict(base, variante="originale", label_int=label, z0=z[0], z1=z[1]))
                    for d, f in enumerate(feats):
                        zo = pm.logits(pm.occlude(d, None))
                        occ_rows.append(dict(base, variante=f, label_int=label, z0=zo[0], z1=zo[1]))

                    # ---------- 2b) occlusione nelle regioni vs controllo ----------
                    for rname in occlusion_regions:
                        if rname not in regions:
                            continue
                        rmask = regions[rname]
                        T = pm.T
                        if rmask.sum() == 0 or rmask.all():
                            continue
                        shifts = rng.integers(int(0.1 * T), int(0.9 * T) + 1, size=n_controls)
                        for d, f in [(None, "tutte")] + list(enumerate(feats)):
                            zr = pm.logits(pm.occlude(d, rmask))
                            delta_r = margin0 - float(zr[cls] - zr[other])
                            deltas_c = []
                            for s in shifts:
                                zc = pm.logits(pm.occlude(d, np.roll(rmask, int(s))))
                                deltas_c.append(margin0 - float(zc[cls] - zc[other]))
                            reg_rows.append(dict(base, regione=rname, feature=f,
                                                 quota_campioni=float(rmask.mean()),
                                                 delta_margine_regione=delta_r,
                                                 delta_margine_controllo=float(np.mean(deltas_c)),
                                                 cambio_decisione=int(np.argmax(zr)) != pred))

        ig = pd.DataFrame(ig_rows)
        occ = pd.DataFrame(occ_rows)
        reg = pd.DataFrame(reg_rows)

        out = self.output_dir / "attribuzione_feature"
        out.mkdir(parents=True, exist_ok=True)
        ig.to_csv(out / "integrated_gradients_per_paziente.csv", index=False)
        occ.to_csv(out / "occlusione_logit_per_paziente.csv", index=False)
        reg.to_csv(out / "occlusione_regioni_per_paziente.csv", index=False)

        results = dict(integrated_gradients=ig, occlusione=occ, occlusione_regioni=reg)
        results.update(self._summaries(ig, occ, reg, feats, out))
        self._plot(results, feats, out)
        return results

    # ---------------------------------------------------------
    def _summaries(self, ig, occ, reg, feats, out: Path) -> dict[str, pd.DataFrame]:
        res = {}
        models = list(dict.fromkeys(ig["modello"]))

        # controllo della completezza di IG
        comp = ig.groupby("modello", sort=False)["errore_completezza_rel"].agg(["mean", "max"])
        comp.to_csv(out / "controllo_completezza_ig.csv")
        print("\n=== Errore relativo di completezza di Integrated Gradients (deve essere piccolo) ===")
        print(comp.round(4).to_string())

        # quota globale di attribuzione per feature
        quota = ig.groupby("modello", sort=False)[[f"quota_{f}" for f in feats]].agg(["mean", "std"])
        quota.to_csv(out / "quota_attribuzione_globale.csv")
        res["quota_globale"] = quota
        print("\n=== Quota dell'attribuzione |IG| per feature sull'intera serie (media sui pazienti) ===")
        print(ig.groupby("modello", sort=False)[[f"quota_{f}" for f in feats]].mean().round(3).to_string())

        # composizione dentro/fuori ciascuna regione + test appaiato (Holm per regione)
        regions = sorted({c.split("_dentro_")[1] for c in ig.columns if "_dentro_" in c})
        comp_rows = []
        for rname in regions:
            rows = []
            for mname in models:
                sub = ig[ig["modello"] == mname]
                for f in feats:
                    a = sub[f"quota_{f}_dentro_{rname}"].values
                    b = sub[f"quota_{f}_fuori_{rname}"].values
                    e = sub[f"arricchimento_{f}_{rname}"].values
                    cnt = sub[f"contributo_{f}_dentro_{rname}"].values
                    rows.append(dict(regione=rname, modello=mname, feature=f,
                                     quota_dentro=np.nanmean(a), quota_fuori=np.nanmean(b),
                                     diff_mediana=np.nanmedian(a - b),
                                     pct_pazienti_dentro_maggiore=float(np.nanmean(a > b) * 100),
                                     arricchimento_medio=np.nanmean(e),
                                     contributo_netto_dentro_medio=np.nanmean(cnt),
                                     p=_safe_wilcoxon(a, b)))
            df = pd.DataFrame(rows)
            df["p_holm"] = holm(df["p"])
            comp_rows.append(df)
        composition = pd.concat(comp_rows, ignore_index=True)
        composition.to_csv(out / "composizione_per_regione.csv", index=False)
        res["composizione"] = composition
        print("\n=== Composizione dell'attribuzione per feature dentro e fuori le regioni ===")
        print(composition.round(3).to_string(index=False))

        # stessa composizione con la sensibilità |gradiente|
        sens_rows = []
        for rname in regions:
            rows = []
            for mname in models:
                sub = ig[ig["modello"] == mname]
                for f in feats:
                    ca, cb = f"sens_{f}_dentro_{rname}", f"sens_{f}_fuori_{rname}"
                    if ca not in sub:
                        continue
                    a_, b_ = sub[ca].values, sub[cb].values
                    rows.append(dict(regione=rname, modello=mname, feature=f,
                                     quota_dentro=np.nanmean(a_), quota_fuori=np.nanmean(b_),
                                     diff_mediana=np.nanmedian(a_ - b_),
                                     pct_pazienti_dentro_maggiore=float(np.nanmean(a_ > b_) * 100),
                                     p=_safe_wilcoxon(a_, b_)))
            if rows:
                df = pd.DataFrame(rows)
                df["p_holm"] = holm(df["p"])
                sens_rows.append(df)
        if sens_rows:
            sensitivity = pd.concat(sens_rows, ignore_index=True)
            sensitivity.to_csv(out / "composizione_sensibilita_per_regione.csv", index=False)
            res["composizione_sensibilita"] = sensitivity
            print("\n=== Composizione della sensibilità |gradiente| per feature dentro e fuori le regioni ===")
            print(sensitivity.round(3).to_string(index=False))

        # occlusione sull'intera serie: accuracy e AUC (media sui fold, come nella tesi)
        occ = occ.copy()
        occ["prob_pd"] = 1 / (1 + np.exp(-(occ["z1"] - occ["z0"])))
        occ["pred_int"] = (occ["z1"] > occ["z0"]).astype(int)
        rows = []
        for (mname, var), g in occ.groupby(["modello", "variante"], sort=False):
            accs, aucs = [], []
            for _, gf in g.groupby("fold"):
                accs.append(float((gf["pred_int"] == gf["label_int"]).mean()))
                aucs.append(float(roc_auc_score(gf["label_int"], gf["prob_pd"]))
                            if gf["label_int"].nunique() > 1 else np.nan)
            rows.append(dict(modello=mname, variante=var, accuracy=np.mean(accs), auc=np.nanmean(aucs),
                             pazienti_corretti=int((g["pred_int"] == g["label_int"]).sum())))
        occ_summary = pd.DataFrame(rows)
        orig = occ_summary[occ_summary["variante"] == "originale"].set_index("modello")
        occ_summary["delta_accuracy"] = occ_summary.apply(
            lambda r: r["accuracy"] - orig.loc[r["modello"], "accuracy"], axis=1)
        occ_summary["delta_auc"] = occ_summary.apply(
            lambda r: r["auc"] - orig.loc[r["modello"], "auc"], axis=1)
        occ_summary["delta_pazienti"] = occ_summary.apply(
            lambda r: r["pazienti_corretti"] - orig.loc[r["modello"], "pazienti_corretti"], axis=1)

        # calo del margine della decisione originale (per paziente) e test rispetto a 0
        o = occ[occ["variante"] == "originale"][["modello", "fold", "patient_id", "z0", "z1"]]
        merged = occ.merge(o, on=["modello", "fold", "patient_id"], suffixes=("", "_orig"))
        cls = (merged["z1_orig"] > merged["z0_orig"]).astype(int)
        m_orig = np.where(cls == 1, merged["z1_orig"] - merged["z0_orig"], merged["z0_orig"] - merged["z1_orig"])
        m_occ = np.where(cls == 1, merged["z1"] - merged["z0"], merged["z0"] - merged["z1"])
        merged["delta_margine"] = m_orig - m_occ
        dm = (merged[merged["variante"] != "originale"]
              .groupby(["modello", "variante"], sort=False)["delta_margine"]
              .agg(delta_margine_medio="mean", delta_margine_mediano="median",
                   p=lambda x: _safe_wilcoxon(x.values)).reset_index())
        dm["p_holm"] = holm(dm["p"])
        occ_summary = occ_summary.merge(dm, on=["modello", "variante"], how="left")
        occ_summary.to_csv(out / "occlusione_intera_serie.csv", index=False)
        res["occlusione_intera_serie"] = occ_summary
        print("\n=== Occlusione di ciascuna feature sull'intera serie ===")
        print(occ_summary.round(3).to_string(index=False))

        # occlusione nelle regioni vs controllo (test appaiato, Holm per regione)
        if not reg.empty:
            rows = []
            for (rname, mname, f), g in reg.groupby(["regione", "modello", "feature"], sort=False):
                a = g["delta_margine_regione"].values
                b = g["delta_margine_controllo"].values
                rows.append(dict(regione=rname, modello=mname, feature=f,
                                 quota_campioni=g["quota_campioni"].mean(),
                                 delta_regione=np.mean(a), delta_controllo=np.mean(b),
                                 diff_mediana=np.median(a - b),
                                 pct_pazienti_regione_maggiore=float(np.mean(a > b) * 100),
                                 cambi_decisione=int(g["cambio_decisione"].sum()),
                                 p=_safe_wilcoxon(a, b)))
            reg_summary = pd.DataFrame(rows)
            reg_summary["p_holm"] = np.nan
            for rname in reg_summary["regione"].unique():
                idx = reg_summary["regione"] == rname
                reg_summary.loc[idx, "p_holm"] = holm(reg_summary.loc[idx, "p"])
            reg_summary.to_csv(out / "occlusione_regioni.csv", index=False)
            res["occlusione_regioni_riepilogo"] = reg_summary
            print("\n=== Occlusione delle feature nelle regioni rispetto a regioni di controllo ===")
            print(reg_summary.round(3).to_string(index=False))

        return res

    # ---------------------------------------------------------
    def _plot(self, res: dict, feats: list[str], out: Path) -> None:
        ig = res["integrated_gradients"]
        models = list(dict.fromkeys(ig["modello"]))
        comp = res["composizione"]
        regions = list(dict.fromkeys(comp["regione"]))

        # quota per feature: intera serie e dentro ciascuna regione
        cols = ["intera serie"] + regions
        fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, mname in zip(axes, models):
            sub = ig[ig["modello"] == mname]
            bottom = np.zeros(len(cols))
            for f in feats:
                vals = [sub[f"quota_{f}"].mean()] + [
                    comp[(comp["modello"] == mname) & (comp["regione"] == r) & (comp["feature"] == f)]
                    ["quota_dentro"].mean() for r in regions]
                ax.bar(range(len(cols)), vals, bottom=bottom, label=f.replace("_", " "))
                bottom += np.nan_to_num(vals)
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels([c.replace("_", " ") for c in cols], rotation=45, ha="right", fontsize=7)
            ax.set_title(mname)
        axes[0].set_ylabel("quota dell'attribuzione |IG|")
        axes[-1].legend(fontsize=7, loc="upper right")
        plt.tight_layout()
        fig.savefig(out / "quota_feature_per_regione.png", dpi=150)
        plt.show() if self.show else plt.close(fig)

        # occlusione sull'intera serie: calo di AUC per feature
        occ = res["occlusione_intera_serie"]
        occ = occ[occ["variante"] != "originale"]
        fig, ax = plt.subplots(figsize=(6, 4))
        w = 0.8 / len(models)
        for i, mname in enumerate(models):
            sub = occ[occ["modello"] == mname].set_index("variante").reindex(feats)
            ax.bar(np.arange(len(feats)) + i * w, sub["delta_auc"], width=w, label=mname)
        ax.set_xticks(np.arange(len(feats)) + w * (len(models) - 1) / 2)
        ax.set_xticklabels([f.replace("_", " ") for f in feats])
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_ylabel("variazione dell'AUC")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(out / "occlusione_delta_auc.png", dpi=150)
        plt.show() if self.show else plt.close(fig)
