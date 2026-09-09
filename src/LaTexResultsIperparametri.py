from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURAZIONE
# ============================================================

ROOT_DIR = Path(
    r"C:\Users\Utente\parkinson_disease\experiments\test_iperparametri"
)

MODEL_DIRS = [
    ROOT_DIR / "cnn1D(train seg - val no)",
    ROOT_DIR / "cnn1D(train seg - val seg)",
]

OUTPUT_TEX = ROOT_DIR / "results_overleaf.tex"


# ============================================================
# METRICHE: SOLO VALIDAZIONE
# ============================================================

METRICS = [
    "val_loss",
    "val_accuracy",
    "val_f1",
    "val_auc",
    "val_precision",
    "val_recall",
    "val_specificity",
]


# ============================================================
# FEATURE UTILIZZATE
# ============================================================

FEATURES = {
    "cnn1D(train seg - val no)": "xy_coordinate",
    "cnn1D(train seg - val seg)": "xy_speed_pressure_derivative",
}


# ============================================================
# NOMI DEI MODELLI
# ============================================================

MODEL_NAMES = {
    "cnn1D(train seg - val no)":
        "CNN 1D - segmentazione fissa (train seg - val no)",

    "cnn1D(train seg - val seg)":
        "CNN 1D - segmentazione fissa (train seg - val seg)",
}


# ============================================================
# PULIZIA DEL VALORE DELL'ESPERIMENTO
# ============================================================

def clean_experiment_value(value: str) -> str:

    prefixes = [
        "size_",
        "lr_",
        "learning_rate_",
        "epoch_",
        "epochs_",
        "patience_",
        "early_stopping_",
        "delta_",
        "min_delta_",
        "reg_",
        "regularization_",
        "window_",
        "window_size_",
    ]

    for prefix in prefixes:
        if value.startswith(prefix):
            return value[len(prefix):]

    return value


# ============================================================
# ESTRAZIONE ESPERIMENTO E VALORE
# ============================================================

def extract_experiment_info(
    model_dir: Path,
    summary_file: Path,
):
    """
    Esempio:

    batch_size/size_16/summary.csv

    -> Esperimento = batch_size
    -> Valore = 16
    """

    relative_path = summary_file.relative_to(model_dir)
    parts = relative_path.parts

    if len(parts) < 3:
        return None, None

    # Esempio:
    # parts = ("batch_size", "size_16", "summary.csv")

    experiment = parts[-3]
    value = parts[-2]

    value = clean_experiment_value(value)

    return experiment, value


# ============================================================
# LETTURA DEI RISULTATI
# ============================================================

rows = []

for model_dir in MODEL_DIRS:

    print("\n" + "=" * 70)
    print(f"MODELLO: {model_dir.name}")
    print("=" * 70)

    if not model_dir.exists():
        print(f"ERRORE: cartella non trovata:")
        print(model_dir)
        continue

    # Cerca tutti i summary.csv nelle sottocartelle
    summary_files = list(model_dir.rglob("summary.csv"))

    print(f"Trovati {len(summary_files)} summary.csv")

    for summary_file in summary_files:

        relative_path = summary_file.relative_to(model_dir)

        # ----------------------------------------------------
        # IGNORA EVENTUALE SUMMARY FINALE NELLA ROOT
        #
        # Vogliamo SOLO quelli degli esperimenti:
        #
        # batch_size/size_16/summary.csv
        # learning_rate/lr_0.001/summary.csv
        # ecc.
        # ----------------------------------------------------

        if len(relative_path.parts) < 3:
            continue

        print(f"\nLettura: {summary_file}")

        try:
            df = pd.read_csv(summary_file)

        except Exception as e:
            print(f"ERRORE nella lettura: {e}")
            continue

        if df.empty:
            print("CSV vuoto -> ignorato")
            continue

        # ----------------------------------------------------
        # CONTROLLO COLONNE
        # ----------------------------------------------------

        missing_metrics = [
            metric
            for metric in METRICS
            if metric not in df.columns
        ]

        if missing_metrics:
            print(
                f"ATTENZIONE: metriche mancanti: {missing_metrics}"
            )

        # ----------------------------------------------------
        # RIGA DEL SUMMARY
        # ----------------------------------------------------

        row = df.iloc[0]

        # ----------------------------------------------------
        # MODELLO
        # ----------------------------------------------------

        model_name = MODEL_NAMES.get(
            model_dir.name,
            model_dir.name
        )

        # ----------------------------------------------------
        # FEATURE
        # ----------------------------------------------------

        features = FEATURES.get(
            model_dir.name,
            "N/A"
        )

        # ----------------------------------------------------
        # ESPERIMENTO E VALORE
        # ----------------------------------------------------

        experiment, value = extract_experiment_info(
            model_dir,
            summary_file
        )

        if experiment is None:
            continue

        # ----------------------------------------------------
        # RISULTATO
        # ----------------------------------------------------

        result = {
            "Dataset": "test_iperparametri",
            "Modello": model_name,
            "Feature": features,
            "Esperimento": experiment,
            "Valore": value,
        }

        # ----------------------------------------------------
        # SOLO METRICHE DI VALIDAZIONE
        # ----------------------------------------------------

        for metric in METRICS:

            if metric in df.columns:
                result[metric] = row[metric]
            else:
                result[metric] = None

        rows.append(result)


# ============================================================
# CONTROLLO
# ============================================================

print("\n" + "=" * 70)
print(f"NUMERO TOTALE DI ESPERIMENTI LETTI: {len(rows)}")
print("=" * 70)

if not rows:
    raise RuntimeError(
        "\nNessun summary.csv di esperimento trovato.\n"
        f"Controlla questa cartella:\n{ROOT_DIR}"
    )


# ============================================================
# DATAFRAME
# ============================================================

results_df = pd.DataFrame(rows)


# ============================================================
# ORDINE DELLE COLONNE
# ============================================================

column_order = [
    "Dataset",
    "Modello",
    "Feature",
    "Esperimento",
    "Valore",
    "val_loss",
    "val_accuracy",
    "val_f1",
    "val_auc",
    "val_precision",
    "val_recall",
    "val_specificity",
]

results_df = results_df[column_order]



# ============================================================
# ORDINE TEMPORALE DEGLI ESPERIMENTI
# ============================================================

EXPERIMENT_ORDER = {
    "window_size": 0,
    "epochs": 1,
    "learning_rate": 2,
    "regularization": 3,
    "batch_size": 4,
    "early_stopping": 5,
    "min_delta": 6,
}

# Ordine dei modelli
MODEL_ORDER = {
    "CNN 1D - segmentazione fissa (train seg - val no)": 0,
    "CNN 1D - segmentazione fissa (train seg - val seg)": 1,
}

results_df["_model_order"] = (
    results_df["Modello"]
    .map(MODEL_ORDER)
    .fillna(999)
)

results_df["_experiment_order"] = (
    results_df["Esperimento"]
    .map(EXPERIMENT_ORDER)
    .fillna(999)
)

results_df = (
    results_df
    .sort_values(
        by=[
            "_model_order",
            "_experiment_order",
            "Valore",
        ],
        key=lambda col: col.astype(str),
    )
    .drop(
        columns=[
            "_model_order",
            "_experiment_order",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# STAMPA RISULTATI
# ============================================================

print("\nRISULTATI TROVATI:\n")
print(results_df.to_string(index=False))


# ============================================================
# PREPARAZIONE LATEX
# ============================================================

latex_df = results_df.rename(
    columns={
        "Dataset": "Dataset",
        "Modello": "Modello",
        "Feature": "Feature",
        "Esperimento": "Esperimento",
        "Valore": "Valore",
        "val_loss": "Val Loss",
        "val_accuracy": "Accuracy",
        "val_f1": "F1",
        "val_auc": "AUC",
        "val_precision": "Precision",
        "val_recall": "Recall",
        "val_specificity": "Specificity",
    }
)


# ============================================================
# GENERAZIONE LATEX COMPLETO
# ============================================================

latex_df = results_df.rename(
    columns={
        "Dataset": "Dataset",
        "Modello": "Modello",
        "Feature": "Feature",
        "Esperimento": "Esperimento",
        "Valore": "Valore",
        "val_loss": "Val Loss",
        "val_accuracy": "Accuracy",
        "val_f1": "F1",
        "val_auc": "AUC",
        "val_precision": "Precision",
        "val_recall": "Recall",
        "val_specificity": "Specificity",
    }
)

table_latex = latex_df.to_latex(
    index=False,
    escape=True,
    float_format="%.4f",
    column_format="lllllrrrrrrr",
)

latex = rf"""
\documentclass{{article}}

\usepackage[utf8]{{inputenc}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{adjustbox}}

\begin{{document}}

\begin{{table}}[htbp]
\centering
\small
\begin{{adjustbox}}{{max width=\textwidth}}
{table_latex}
\end{{adjustbox}}

\caption{{Risultati degli esperimenti di ottimizzazione degli iperparametri sul dataset test\_iperparametri.}}
\label{{tab:hyperparameter_results}}

\end{{table}}

\end{{document}}
"""

OUTPUT_TEX.write_text(
    latex,
    encoding="utf-8"
)


# ============================================================
# SALVATAGGIO
# ============================================================

OUTPUT_TEX.write_text(
    latex,
    encoding="utf-8"
)


# ============================================================
# FINE
# ============================================================

print("\n" + "=" * 70)
print("TABELLA OVERLEAF GENERATA")
print("=" * 70)

print(f"\nFile salvato in:")
print(OUTPUT_TEX)
