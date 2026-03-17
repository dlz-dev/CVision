import re
import json
import time
from datetime import datetime
from pathlib import Path
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# Initialisation du géolocalisateur (OpenStreetMap)
geolocator = Nominatim(user_agent="cv_distance_calculator_luxembourg")
LUXEMBOURG_VILLE_HAUTE_COORDS = (49.6116, 6.1319)


def parse_date(date_str):
    """
    Essaie de parser plusieurs formats de dates, y compris les formats US.
    Retourne un objet datetime valide ou None.
    """
    if date_str.lower() == 'present':
        return datetime.now()

    # Formats gérés: YYYY-MM-DD, MM/DD/YYYY (US), DD/MM/YYYY, YYYY-MM, MM/YYYY (US)
    formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m', '%m/%Y']
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def obtenir_coordonnees(adresse, geolocator):
    """
    Tente d'obtenir les coordonnées GPS. En cas d'échec, nettoie et reformate
    l'adresse (notamment pour les formats US) et réessaie.
    """
    # Tentative 1 : L'adresse exacte telle qu'écrite dans le CV
    location = geolocator.geocode(adresse, timeout=10)
    if location:
        return location

    # Tentative 2 : Détection de format US/Étranger et ajout du pays
    # Si on détecte un code postal à 5 chiffres ou une abréviation d'état à 2 lettres
    if re.search(r'\b[A-Z]{2}\b|\b\d{5}\b', adresse):
        adresse_usa = adresse
        if "usa" not in adresse.lower() and "united states" not in adresse.lower():
            adresse_usa += ", USA"

        time.sleep(1)  # Pause API obligatoire
        location = geolocator.geocode(adresse_usa, timeout=10)
        if location:
            return location

    # Tentative 3 : Simplification (On enlève la rue/le numéro pour ne garder que la ville)
    parts = [p.strip() for p in adresse.split(',')]
    if len(parts) > 1:
        # Garde tout après la première virgule (souvent City, State, Zip)
        adresse_simplifiee = ", ".join(parts[1:])

        # On force "USA" si ça ressemble toujours à un format américain
        if re.search(r'\b[A-Z]{2}\b|\b\d{5}\b', adresse_simplifiee) and "usa" not in adresse_simplifiee.lower():
            adresse_simplifiee += ", USA"

        time.sleep(1)  # Pause API obligatoire
        location = geolocator.geocode(adresse_simplifiee, timeout=10)
        return location

    return None


def extraire_donnees_cv(chemin_fichier: Path):
    annee_actuelle = datetime.now().year
    mois_actuel = datetime.now().month
    jour_actuel = datetime.now().day

    texte = chemin_fichier.read_text(encoding='utf-8')

    # Nettoyage des balises du système
    texte = re.sub(r'\\s*', '', texte)

    cv_data = {}

    # 1. AGE (Support des dates US)
    dob_match = re.search(r'Date of Birth:\s*([\d\-\/]+)', texte)
    if dob_match:
        dob = parse_date(dob_match.group(1))
        if dob:
            age = annee_actuelle - dob.year - ((mois_actuel, jour_actuel) < (dob.month, dob.day))
            cv_data['AGE'] = age
        else:
            cv_data['AGE'] = None

    # 2. ADRESSE ET DISTANCE (Avec reformatage intelligent)
    addr_match = re.search(r'Address:\s*(.+)', texte)
    if addr_match:
        adresse_candidat = addr_match.group(1).strip()
        cv_data['ADRESSE'] = adresse_candidat

        try:
            # Utilisation de la fonction de recherche avec repli
            location = obtenir_coordonnees(adresse_candidat, geolocator)

            if location:
                candidat_coords = (location.latitude, location.longitude)
                distance_km = geodesic(candidat_coords, LUXEMBOURG_VILLE_HAUTE_COORDS).kilometers
                cv_data['DISTANCE_VILLE_HAUTE_KM'] = round(distance_km, 2)
            else:
                cv_data['DISTANCE_VILLE_HAUTE_KM'] = "Adresse introuvable (Format non reconnu)"

            time.sleep(1)  # Pause API finale par sécurité

        except Exception as e:
            cv_data['DISTANCE_VILLE_HAUTE_KM'] = f"Erreur de calcul API : {str(e)}"
    else:
        cv_data['ADRESSE'] = None
        cv_data['DISTANCE_VILLE_HAUTE_KM'] = None

    # 3. TARGET ROLE
    role_match = re.search(r'Target Role:\s*(.+)', texte)
    target_role = role_match.group(1).strip() if role_match else ""
    cv_data['TARGET_ROLE'] = target_role

    # 4 & 5. ECOLE, DIPLOME ET AGE DU DIPLOME
    edu_match = re.search(r'Education:\n(.*?)(?=\n\n|\nExperience:)', texte, re.DOTALL)
    if edu_match:
        edu_line = edu_match.group(1).strip()
        parts = [p.strip() for p in edu_line.split('—')]
        if len(parts) >= 4:
            cv_data['DIPLOME'] = f"{parts[0]} - {parts[1]}"
            cv_data['ECOLE'] = parts[2]
            try:
                cv_data['AGE_DIPLOME'] = annee_actuelle - int(parts[3])
            except ValueError:
                cv_data['AGE_DIPLOME'] = None

    # 6. EXPERIENCES, ACCOMPLISSEMENTS, PAUSES, CUMUL
    exp_match = re.search(r'Experience:\n(.*?)(?=\n\nSkills:)', texte, re.DOTALL)
    experiences = []
    accomplissements = []
    pauses = []
    cumul_experiences = {}

    if exp_match:
        exp_text = exp_match.group(1)
        job_pattern = r'(.+?)\s*—\s*(.+?)\s*—\s*(.+?)\s*—\s*([\d\-\/]+)\s*to\s*([a-zA-Z]+|[\d\-\/]+)'
        jobs = list(re.finditer(job_pattern, exp_text))

        for i, job in enumerate(jobs):
            titre = job.group(1).strip()
            entreprise = job.group(2).strip()
            start_raw = job.group(4).strip()
            end_raw = job.group(5).strip()

            start_date = parse_date(start_raw)
            end_date = parse_date(end_raw)

            # Reformatage des dates en YYYY-MM
            start_str = start_date.strftime('%Y-%m') if start_date else start_raw
            if end_raw.lower() == 'present':
                end_str = 'Present'
            else:
                end_str = end_date.strftime('%Y-%m') if end_date else end_raw

            duree_mois = 0
            if start_date and end_date:
                duree_mois = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

            mots_target = set(target_role.lower().split())
            mots_titre = set(titre.lower().split())
            est_en_lien = len(mots_target.intersection(mots_titre)) > 0

            start_pos = job.end()
            end_pos = jobs[i + 1].start() if i + 1 < len(jobs) else len(exp_text)
            description = exp_text[start_pos:end_pos]
            acc = [a.strip() for a in re.findall(r'\*\s*(.+)', description)]
            accomplissements.extend(acc)

            experiences.append({
                'titre': titre,
                'entreprise': entreprise,
                'dates': f"{start_str} to {end_str}",
                'duree_mois': duree_mois,
                'en_lien_avec_target_role': est_en_lien,
                'accomplissements': acc
            })

            categorie = titre.split()[-1]
            cumul_experiences[categorie] = cumul_experiences.get(categorie, 0) + duree_mois

            if i > 0:
                fin_job_precedent_raw = jobs[i - 1].group(5).strip()

                if fin_job_precedent_raw.lower() != 'present':
                    d1 = parse_date(fin_job_precedent_raw)
                    d2 = start_date

                    if d1 and d2:
                        gap_mois = (d2.year - d1.year) * 12 + (d2.month - d1.month)
                        if gap_mois > 1:
                            pauses.append({
                                'de': d1.strftime('%Y-%m'),
                                'a': d2.strftime('%Y-%m'),
                                'duree_mois': gap_mois,
                                'commentaire': 'Trou détecté'
                            })

    cv_data['EXPERIENCES'] = experiences
    cv_data['ACCOMPLISSEMENTS_GLOBAUX'] = accomplissements
    cv_data['PAUSES'] = pauses

    cumul_formate = {}
    for role, total_mois in cumul_experiences.items():
        ans = total_mois // 12
        mois = total_mois % 12
        cumul_formate[role] = f"{ans} ans et {mois} mois"
    cv_data['EXPERIENCES_CUMULEES'] = cumul_formate

    # 7. COMPETENCES
    skills_match = re.search(r'Skills:\n(.*?)(?=\n\nLanguages:)', texte, re.DOTALL)
    if skills_match:
        cv_data['COMPETENCES'] = [s.strip() for s in skills_match.group(1).strip().split('\n') if s.strip()]

    # 8. LANGUES
    lang_match = re.search(r'Languages:\n(.*?)(?=\n\nCertifications:)', texte, re.DOTALL)
    if lang_match:
        cv_data['LANGUES'] = [l.strip() for l in lang_match.group(1).strip().split('\n') if l.strip()]

    # 9. CERTIFICATIONS
    cert_match = re.search(r'Certifications:\n(.*)', texte, re.DOTALL)
    if cert_match:
        cv_data['CERTIFICATIONS'] = [c.strip() for c in cert_match.group(1).strip().split('\n') if c.strip()]

    return cv_data


def generer_json_depuis_dossier():
    chemin_script = Path(__file__).resolve()
    dossier_racine = chemin_script.parent.parent

    dossier_raw = dossier_racine / 'data' / 'raw'
    dossier_extracted = dossier_racine / 'data' / 'extracted'

    dossier_extracted.mkdir(parents=True, exist_ok=True)

    fichiers_cv = list(dossier_raw.glob('cv*.txt'))
    print(f"Début de l'extraction de {len(fichiers_cv)} CV(s) depuis {dossier_raw.name}...")

    for fichier in fichiers_cv:
        nom_candidat = fichier.stem
        print(f"Traitement de {fichier.name}...")

        donnees_cv = extraire_donnees_cv(fichier)

        chemin_sortie = dossier_extracted / f"{nom_candidat}.json"

        with open(chemin_sortie, 'w', encoding='utf-8') as f_json:
            json.dump(donnees_cv, f_json, indent=4, ensure_ascii=False)

    print(f"\nExtraction terminée ! Les JSON sont générés dans le dossier: {dossier_extracted}")


if __name__ == "__main__":
    generer_json_depuis_dossier()