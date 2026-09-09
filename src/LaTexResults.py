from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURAZIONE
# ============================================================

ROOT_DIR = Path(
    r"C:\Users\Utente\parkinson_disease\experiments\cnn1D"
)

MODEL_DIRS = [
    ROOT_DIR / "raw_segment + cnn(train_seg - val_no)",
    ROOT_DIR / "raw_segment + cnn(train_seg - val_seg)",
]

OUTPUT_TEX = ROOT_DIR / "results_overleaf_tasks.tex"


# ============================================================
# METRICHE
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
# NOMI MODELLI
# ============================================================

MODEL_NAMES = {
    "raw_segment + cnn(train_seg - val_no)":
        "CNN 1D - segmentazione fissa (train seg - val no)",

    "raw_segment + cnn(train_seg - val_seg)":
        "CNN 1D - segmentazione fissa (train seg - val seg)",
}


# ============================================================
# ESCAPE TESTO NORMALE LATEX
# ============================================================

def escape_latex(text):
    """
    Escape dei caratteri speciali per testo normale.
    """

    text = str(text)

    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "\\": r"\textbackslash{}",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


# ============================================================
# TESTO PER FEATURE E TASK
# ============================================================

def latex_path(text):
    """
    Mantiene il nome ESATTO della feature/task, compresi
    gli underscore, e permette a LaTeX di andare a capo.

    Esempio:
        xy_speed_pressure_derivative
    rimane:
        xy_speed_pressure_derivative
    """

    text = str(text)

    # \path gestisce automaticamente gli underscore
    return r"\path{" + text + "}"


# ============================================================
# FORMATTAZIONE METRICHE
# ============================================================

def format_metric_column(
    series,
    minimize=False,
):
    """
    Migliore:
        -> grassetto

    Secondo migliore:
        -> sottolineato

    Val Loss:
        -> più basso è migliore

    Tutte le altre:
        -> più alto è migliore
    """

    values = pd.to_numeric(
        series,
        errors="coerce"
    )

    valid_values = values.dropna()

    if valid_values.empty:
        return [""] * len(values)

    # --------------------------------------------------------
    # Valori distinti
    # --------------------------------------------------------

    unique_values = sorted(
        valid_values.unique(),
        reverse=not minimize
    )

    best_value = unique_values[0]

    second_best_value = (
        unique_values[1]
        if len(unique_values) > 1
        else None
    )

    formatted = []

    for value in values:

        if pd.isna(value):

            formatted.append("")

        elif value == best_value:

            formatted.append(
                rf"\textbf{{{value:.4f}}}"
            )

        elif (
            second_best_value is not None
            and value == second_best_value
        ):

            formatted.append(
                rf"\underline{{{value:.4f}}}"
            )

        else:

            formatted.append(
                f"{value:.4f}"
            )

    return formatted


# ============================================================
# LETTURA DEI RISULTATI
# ============================================================

rows = []


for model_dir in MODEL_DIRS:

    print()
    print("=" * 90)
    print(f"MODELLO: {model_dir.name}")
    print("=" * 90)

    if not model_dir.exists():

        print("ERRORE: cartella non trovata:")
        print(model_dir)

        continue

    # --------------------------------------------------------
    # TROVA TUTTI I SUMMARY
    # --------------------------------------------------------

    summary_files = sorted(
        model_dir.rglob("summary.csv")
    )

    print(
        f"summary.csv trovati: {len(summary_files)}"
    )

    valid_for_model = 0

    for summary_file in summary_files:

        relative_path = summary_file.relative_to(
            model_dir
        )

        parts = relative_path.parts

        # ----------------------------------------------------
        # STRUTTURA ATTESA:
        #
        # Task_1/acceleration/summary.csv
        # ----------------------------------------------------

        if len(parts) < 3:

            print(
                f"IGNORATO: percorso non valido -> "
                f"{summary_file}"
            )

            continue

        # ----------------------------------------------------
        # TASK
        # ----------------------------------------------------

        task = parts[-3]

        # Mantieni la forma Task_...
        if task.lower().startswith("task_"):
            task = "Task_" + task[5:]

        # ----------------------------------------------------
        # FEATURE
        # ----------------------------------------------------

        feature = parts[-2]

        print(
            f"Lettura: {summary_file}"
            f" | Task={task}"
            f" | Feature={feature}"
        )

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        try:

            df = pd.read_csv(summary_file)

        except Exception as e:

            print(
                f"ERRORE leggendo {summary_file}: {e}"
            )

            continue

        if df.empty:

            print(
                f"CSV vuoto -> {summary_file}"
            )

            continue

        # ----------------------------------------------------
        # PRIMA RIGA DEL SUMMARY
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
        # RISULTATO
        # ----------------------------------------------------

        result = {
            "Modello": model_name,
            "Feature": feature,
            "Task": task,
        }

        # ----------------------------------------------------
        # SOLO METRICHE DI VALIDAZIONE
        # ----------------------------------------------------

        for metric in METRICS:

            if metric in df.columns:

                result[metric] = row[metric]

            else:

                result[metric] = None

                print(
                    f"ATTENZIONE: {metric} mancante in "
                    f"{summary_file}"
                )

        rows.append(result)
        valid_for_model += 1

    print(
        f"Risultati validi per il modello: "
        f"{valid_for_model}"
    )


# ============================================================
# CONTROLLO
# ============================================================

print()
print("=" * 90)
print(
    f"NUMERO TOTALE DI RISULTATI: {len(rows)}"
)
print("=" * 90)

if not rows:

    raise RuntimeError(
        "\nNessun summary.csv valido trovato.\n"
        f"Controlla:\n{ROOT_DIR}"
    )


# ============================================================
# DATAFRAME
# ============================================================

results_df = pd.DataFrame(rows)


# ============================================================
# CONVERSIONE NUMERICA
# ============================================================

for metric in METRICS:

    results_df[metric] = pd.to_numeric(
        results_df[metric],
        errors="coerce"
    )


# ============================================================
# ORDINE COLONNE
# ============================================================

results_df = results_df[
    [
        "Modello",
        "Feature",
        "Task",
        "val_loss",
        "val_accuracy",
        "val_f1",
        "val_auc",
        "val_precision",
        "val_recall",
        "val_specificity",
    ]
]


# ============================================================
# ORDINA PER ACCURACY
# ============================================================

results_df = (
    results_df
    .sort_values(
        by="val_accuracy",
        ascending=False,
        na_position="last",
    )
    .reset_index(drop=True)
)


# ============================================================
# PREPARAZIONE LATEX
# ============================================================

latex_df = results_df.rename(
    columns={
        "val_loss": "Val Loss",
        "val_accuracy": "Accuracy",
        "val_f1": "F1",
        "val_auc": "AUC",
        "val_precision": "Precision",
        "val_recall": "Recall",
        "val_specificity": "Specificity",
    }
).copy()


# ============================================================
# MODELLO: TESTO NORMALE
# ============================================================

latex_df["Modello"] = (
    latex_df["Modello"]
    .apply(escape_latex)
)


# ============================================================
# FEATURE E TASK:
# MANTIENI IL NOME ESATTO
# ============================================================

latex_df["Feature"] = (
    latex_df["Feature"]
    .apply(latex_path)
)

latex_df["Task"] = (
    latex_df["Task"]
    .apply(latex_path)
)


# ============================================================
# METRICHE
# ============================================================

LATEX_METRICS = [
    "Val Loss",
    "Accuracy",
    "F1",
    "AUC",
    "Precision",
    "Recall",
    "Specificity",
]


# Val Loss -> minimo migliore
MINIMIZE_METRICS = {
    "Val Loss"
}


for column in LATEX_METRICS:

    latex_df[column] = format_metric_column(
        latex_df[column],
        minimize=(column in MINIMIZE_METRICS)
    )


# ============================================================
# COSTRUZIONE MANUALE DELLE RIGHE
# ============================================================

table_rows = []

for _, row in latex_df.iterrows():

    table_row = (
        f"{row['Modello']} & "
        f"{row['Feature']} & "
        f"{row['Task']} & "
        f"{row['Val Loss']} & "
        f"{row['Accuracy']} & "
        f"{row['F1']} & "
        f"{row['AUC']} & "
        f"{row['Precision']} & "
        f"{row['Recall']} & "
        f"{row['Specificity']} "
        r"\\"
    )

    table_rows.append(table_row)


table_rows = "\n".join(table_rows)


# ============================================================
# DOCUMENTO LATEX
# ============================================================

latex = rf"""
\documentclass{{article}}

\usepackage[utf8]{{inputenc}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{array}}
\usepackage{{pdflscape}}
\usepackage{{url}}

\begin{{document}}

\begin{{landscape}}

\scriptsize

\setlength{{\tabcolsep}}{{3pt}}
\renewcommand{{\arraystretch}}{{1.15}}

\begin{{longtable}}{{p{{4.2cm}} p{{5.2cm}} p{{1.4cm}} rrrrrrr}}

\caption{{Risultati dei task della CNN 1D ordinati per accuracy.}}
\label{{tab:cnn_tasks}}
\\

\toprule

Modello &
Feature &
Task &
Val Loss &
Accuracy &
F1 &
AUC &
Precision &
Recall &
Specificity
\\

\midrule

\endfirsthead


\toprule

Modello &
Feature &
Task &
Val Loss &
Accuracy &
F1 &
AUC &
Precision &
Recall &
Specificity
\\

\midrule

\endhead


\midrule
\multicolumn{{10}}{{r}}{{Continua nella pagina successiva}} \\

\endfoot


\bottomrule
\endlastfoot

{table_rows}

\end{{longtable}}

\end{{landscape}}

\end{{document}}
"""


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

print()
print("=" * 90)
print("TABELLA OVERLEAF GENERATA")
print("=" * 90)

print(
    f"\nNumero totale di righe: {len(latex_df)}"
)

print(
    f"\nFile salvato in:\n{OUTPUT_TEX}"
)
