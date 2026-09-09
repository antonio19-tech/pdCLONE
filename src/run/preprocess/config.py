from pathlib import Path

# Root del progetto
_PROJECT_PATH = Path(__file__).resolve().parents[3]


def _project_path(relative_path: str | Path) -> Path:
    """
    Restituisce il percorso assoluto a partire dalla root del progetto.

    Esempio:
        _project_path("PaHaW/PaHaW/PaHaW_public")
    """
    return _PROJECT_PATH / Path(relative_path)
