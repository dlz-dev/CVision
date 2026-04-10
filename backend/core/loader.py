from pathlib import Path


def load_cv(filepath: str) -> str:
    """Charge et nettoie le contenu d'un fichier texte (CV)."""
    path = Path(filepath)

    if not path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")

    if path.suffix != ".txt":
        raise ValueError(f"Format invalide, attendu .txt : {path.suffix}")

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Fichier vide : {filepath}")

    return content


def load_cvs_from_folder(folder_path: str) -> dict[str, str]:
    """Charge l'ensemble des fichiers .txt d'un dossier donné."""
    folder = Path(folder_path)

    if not folder.is_dir():
        raise NotADirectoryError(f"Dossier introuvable : {folder_path}")

    cvs = {txt_file.name: load_cv(str(txt_file)) for txt_file in folder.glob("*.txt")}

    if not cvs:
        raise FileNotFoundError(f"Aucun fichier .txt trouvé dans : {folder_path}")

    return cvs