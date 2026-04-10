from pathlib import Path
import traceback
import time
import random

from backend.core import pre_process_cv


def test_cv_processing(data_folder="data/raw", limit=None):
    """
    Test la fonction pre_process_cv sur les fichiers d'un dossier.
    """
    raw_dir = Path(data_folder)

    if not raw_dir.exists():
        print(f"Erreur : Le dossier '{data_folder}' est introuvable.")
        return

    txt_files = list(raw_dir.glob("*.txt"))

    random.shuffle(txt_files)

    if not txt_files:
        print(f"Aucun fichier .txt trouvé dans '{data_folder}'.")
        return

    if limit:
        txt_files = txt_files[:limit]

    total_files = len(txt_files)
    success_count = 0
    error_count = 0
    na_distance_count = 0
    errors = []

    print(f"Démarrage du test sur {total_files} fichier(s)...")
    print(
        "Attention : En raison de l'API Nominatim (géolocalisation), le traitement prendra au moins 1 seconde par adresse trouvée.")
    print("-" * 50)

    start_time = time.time()

    for filepath in txt_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                cv_text = f.read()

            result = pre_process_cv(cv_text)
            success_count += 1

            # 👈 VÉRIFICATION DE LA DISTANCE
            dist = result.get('distance_ville_haute_km')
            if dist is None:
                na_distance_count += 1
                dist_str = 'N/A'
            else:
                dist_str = f"{dist}km"

            role = result.get('target_role') or 'Non défini'
            age = result.get('age') or 'Non défini'

            # Affichage de suivi pour chaque fichier
            print(f"[{success_count}/{total_files}] [OK] {filepath.name} -> Rôle: {role} | Dist: {dist_str}")

        except Exception as e:
            error_count += 1
            errors.append((filepath.name, str(e), traceback.format_exc()))
            print(f"[ERREUR] Échec sur {filepath.name} : {e}")

    elapsed_time = time.time() - start_time

    # --- RÉSUMÉ ---
    print("-" * 50)
    print("📊 RÉSUMÉ DU TEST")
    print(f"Temps d'exécution : {elapsed_time:.2f} secondes")
    print(f"Fichiers traités  : {total_files}")
    print(f"✅ Traités (sans crash) : {success_count}")
    print(
        f"⚠️ Dont distances N/A  : {na_distance_count} ({(na_distance_count / success_count * 100):.1f}% des fichiers traités)" if success_count > 0 else "⚠️ Dont distances N/A  : 0")
    print(f"❌ Échecs (crash)     : {error_count}")

    if errors:
        print("\n🔍 DÉTAIL DES ERREURS (Top 5) :")
        for filename, err_msg, trace in errors[:5]:
            print(f"\nFichier : {filename}")
            print(f"Erreur  : {err_msg}")

if __name__ == "__main__":
    # 💡 Astuce : Laisse limit=5 pour faire un essai rapide avant de lancer les 200.
    # Remplace par limit=None pour tester l'intégralité du dossier.
    test_cv_processing(data_folder="../data/raw", limit=None)