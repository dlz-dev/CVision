import os
import json
import time
from pathlib import Path

# Import des fonctions nécessaires depuis votre module preprocessor
from preprocessor import _split_sections, compute_distance_km


def update_missing_distances(json_directory: str, txt_directory: str):
    """
    Parcourt un dossier de JSON, identifie ceux dont la distance est null,
    retrouve l'adresse dans le fichier TXT source et met à jour le JSON.
    """
    json_path = Path(json_directory)
    txt_path = Path(txt_directory)

    # Parcourir tous les fichiers JSON du dossier
    for json_file in json_path.glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Erreur de lecture du fichier {json_file.name}. Ignoré.")
                continue

        # Vérifier si la distance est nulle
        if data.get("distance_ville_haute_km") is None:
            print(f"Traitement de {json_file.name}...")

            # 1. Identifier le fichier texte source
            cv_filename = data.get("meta", {}).get("cv_id")
            if not cv_filename:
                print(f"  -> Ignoré : Aucun 'cv_id' trouvé dans les métadonnées.")
                continue

            # Concatène "data/raw" + "cv_XXXX.txt"
            cv_filepath = txt_path / cv_filename

            if not cv_filepath.exists():
                print(f"  -> Erreur : Le fichier texte source '{cv_filepath}' est introuvable.")
                continue

            # 2. Extraire le texte et retrouver l'adresse
            with open(cv_filepath, 'r', encoding='utf-8') as f_txt:
                cv_text = f_txt.read()

            sections = _split_sections(cv_text)
            address = sections.get("Address", "").strip()

            if address:
                # 3. Recalculer la distance
                distance = compute_distance_km(address)

                # 4. Mettre à jour et sauvegarder si une distance est trouvée
                if distance is not None:
                    data["distance_ville_haute_km"] = distance

                    with open(json_file, 'w', encoding='utf-8') as f_out:
                        json.dump(data, f_out, indent=2, ensure_ascii=False)

                    print(f"  -> Succès : Distance mise à jour ({distance} km) pour '{address}'.")
                else:
                    print(f"  -> Échec : L'API n'a pas pu géocoder l'adresse '{address}'.")

                # Pause obligatoire pour respecter le rate limit de Nominatim (1 requête/sec)
                time.sleep(1)
            else:
                print(f"  -> Ignoré : Aucune section 'Address' trouvée dans '{cv_filename}'.")


if __name__ == "__main__":
    # Ajustez DOSSIER_JSON avec le chemin où vous stockez vos fichiers .json traités
    DOSSIER_JSON = "../data/extracted"

    # Pointage direct vers le dossier contenant vos fichiers bruts
    DOSSIER_TXT = "../data/raw"

    update_missing_distances(DOSSIER_JSON, DOSSIER_TXT)