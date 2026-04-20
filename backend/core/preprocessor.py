import re
import time
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

geolocator = Nominatim(user_agent="cv_distance_calculator_luxembourg")
LUXEMBOURG_COORDS = (49.6116, 6.1319)

# ---------------------------------------------------------------------------
# Constantes & patterns compilés une seule fois au chargement du module
# ---------------------------------------------------------------------------

_SECTION_HEADERS = [
    "Name", "Gender", "Date of Birth", "Address", "Email", "Phone", "Target Role",
    "Professional Summary", "Education", "Experience",
    "Skills", "Languages", "Certifications",
]

_SECTION_SPLIT_RE = re.compile(
    r'^(' + '|'.join(re.escape(h) for h in _SECTION_HEADERS) + r'):',
    re.MULTILINE,
)

_LLM_SECTION_PATTERNS = {
    section: re.compile(
        rf"{re.escape(section)}:\s*(.*?)(?=\n(?:{'|'.join(re.escape(s) for s in _SECTION_HEADERS if s != section)}):|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    for section in ("Education", "Experience")
}

_LANG_RE        = re.compile(r'^(.+?)\s*[—\-]\s*(.+)$', re.MULTILINE)
_CERT_YEAR_RE   = re.compile(r'^(.+?)\s*[—\-]\s*(\d{4})$')
_YEAR_RE        = re.compile(r'\b(20\d{2}|19\d{2})\b')
_EMAIL_RE       = re.compile(r'[\w.+-]+@[\w-]+\.[a-z]{2,}', re.IGNORECASE)
_SKILL_LABEL_RE = re.compile(r'^[^:]+:')
_US_FORMAT_RE   = re.compile(r'\b[A-Z]{2}\b|\b\d{5}\b')

LANGUAGE_LEVEL_SCORE: dict[str, int] = {
    "A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6,
}

_DEGREE_KW_SCORE: dict[str, int] = {
    "phd": 5, "doctorat": 5, "doctor": 5, "d.sc": 5,
    "master": 4, "msc": 4, "mba": 4, "meng": 4, "ma ": 4,
    "m.s": 4, "m.a": 4, "magistère": 4,
    "bachelor": 3, "licence": 3, "licencié": 3,
    "b.s": 3, "b.a": 3, "bsc": 3, "ba ": 3,
    "bts": 2, "dut": 2, "associate": 2, "hnd": 2,
    "hnc": 2, "brevet de technicien": 2,
    "baccalauréat": 1, "baccalaureat": 1, "high school": 1, "lycée": 1,
}

_DATE_FORMATS = ['%Y-%m', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%m/%Y']


def _split_sections(cv_text: str) -> dict[str, str]:
    """Découpe le CV en sections en utilisant la regex compilée."""
    sections: dict[str, str] = {}
    matches = list(_SECTION_SPLIT_RE.finditer(cv_text))
    for i, m in enumerate(matches):
        key = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cv_text)
        sections[key] = cv_text[start:end].strip()
    return sections


def parse_date(date_str: str) -> datetime | None:
    """Parse une date en testant séquentiellement plusieurs formats standards."""
    if date_str.strip().lower() == "present":
        return datetime.now()

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def extract_email(cv_text: str) -> str | None:
    """Extrait la première adresse email trouvée."""
    m = _EMAIL_RE.search(cv_text)
    return m.group(0) if m else None


def compute_age(dob: datetime) -> int:
    """Calcule l'âge exact à partir de la date de naissance."""
    now = datetime.now()
    return now.year - dob.year - ((now.month, now.day) < (dob.month, dob.day))


def _geocode_with_fallback(address: str):
    """
    Tente de géocoder une adresse avec 4 niveaux de simplification successifs
    pour maximiser les chances de correspondances API.
    """
    address_lower = address.lower()

    # 1. Adresse exacte
    location = geolocator.geocode(address, timeout=10)
    if location:
        return location

    # 2. Ajout du pays si format US détecté
    if _US_FORMAT_RE.search(address) and "usa" not in address_lower and "united states" not in address_lower:
        time.sleep(1)
        location = geolocator.geocode(f"{address}, USA", timeout=10)
        if location:
            return location

    parts = [p.strip() for p in address.split(',')]

    # 3. Simplification en supprimant la rue
    if len(parts) > 1:
        simplified = ", ".join(parts[1:])
        if _US_FORMAT_RE.search(simplified) and "usa" not in simplified.lower():
            simplified += ", USA"
        time.sleep(1)
        location = geolocator.geocode(simplified, timeout=10)
        if location:
            return location

    # 4. Ville + Pays (Nettoyage des codes postaux)
    if len(parts) >= 2:
        # On cible l'avant-dernier élément (ex: "38274 Lisbon")
        city_part = parts[-2]

        # On supprime les chiffres (code postal) pour ne garder que la ville
        city_clean = re.sub(r'\d+', '', city_part).strip()
        # On supprime aussi les préfixes pays éventuels (ex: "PT-Lisbon")
        city_clean = re.sub(r'^[A-Z]{1,2}-', '', city_clean).strip()

        if city_clean:
            time.sleep(1)
            return geolocator.geocode(f"{city_clean}, {parts[-1]}", timeout=10)
    return None


def compute_distance_km(address: str) -> float | None:
    """Retourne la distance en km entre l'adresse fournie et le centre de Luxembourg."""
    try:
        location = _geocode_with_fallback(address)
        if location:
            coords = (location.latitude, location.longitude)
            return round(geodesic(coords, LUXEMBOURG_COORDS).kilometers, 2)
    except Exception as e:
        print(f"Géolocalisation échouée pour '{address}': {e}")
    return None


# Extractions & Métriques
def score_language_level(level: str) -> int | None:
    return LANGUAGE_LEVEL_SCORE.get(level.strip().upper()) if level else None


def score_education(degree: str) -> int | None:
    if not degree:
        return None
    d = degree.lower()
    for kw, score in _DEGREE_KW_SCORE.items():
        if kw in d:
            return score
    return None


def extract_skills(section_text: str) -> list[str]:
    """Nettoie et extrait les compétences sous forme de liste plate."""
    skills = []
    for line in filter(None, (l.strip() for l in section_text.split('\n'))):
        content = _SKILL_LABEL_RE.sub('', line).strip()
        for skill in content.split(','):
            s = skill.strip()
            if s:
                skills.append(s)
    return skills


def extract_languages(section_text: str) -> list[dict]:
    """Extrait les langues parlées et attribue un score selon le standard CECRL."""
    return [{
        "language": m.group(1).strip(),
        "level": m.group(2).strip(),
        "score": score_language_level(m.group(2).strip())
    } for m in _LANG_RE.finditer(section_text)]


def extract_certifications(section_text: str) -> list[dict]:
    """Sépare les noms de certifications de leur année d'obtention si présente."""
    certifications = []
    for line in filter(None, (l.strip() for l in section_text.split('\n'))):
        m = _CERT_YEAR_RE.match(line)
        certifications.append(
            {"name": m.group(1).strip(), "year": int(m.group(2))} if m
            else {"name": line, "year": None}
        )
    return certifications


def extract_graduation_year(section_text: str) -> int | None:
    """Trouve la dernière année mentionnée dans la section éducation."""
    years = _YEAR_RE.findall(section_text)
    return int(years[-1]) if years else None


def clean_cv_text_for_llm(cv_text: str) -> str:
    """Filtre le texte pour ne conserver que les sections pertinentes pour le LLM."""
    return "\n\n".join(
        f"{section}:\n{m.group(1).strip()}"
        for section, pattern in _LLM_SECTION_PATTERNS.items()
        if (m := pattern.search(cv_text))
    )


def compute_experience_metrics(experiences: list) -> dict:
    """Calcule l'expérience globale, la durée par poste et identifie les 'trous' dans le CV."""
    if not experiences:
        return {"experiences": [], "total_experience_years": 0.0, "experience_gaps_months": []}

    total_months = 0
    gaps, enriched_exps, parsed_exps = [], [], []

    for exp in experiences:
        start_date = parse_date(exp.get("start", ""))
        end_date = parse_date(exp.get("end", "").strip()) if exp.get("end", "").strip() else None

        if start_date:
            parsed_exps.append({"raw": exp, "start": start_date, "end": end_date or start_date})
        else:
            enriched_exps.append({**exp, "duration_months": None})

    parsed_exps.sort(key=lambda x: x["start"])

    for i, exp in enumerate(parsed_exps):
        duration_months = round((exp["end"] - exp["start"]).days / 30.44)
        total_months += max(0, duration_months)  # Précaution contre les dates inversées
        enriched_exps.append({**exp["raw"], "duration_months": duration_months})

        if i < len(parsed_exps) - 1:
            gap_months = round((parsed_exps[i + 1]["start"] - exp["end"]).days / 30.44)
            if gap_months > 1:
                gaps.append({
                    "from": exp["end"].strftime("%Y-%m"),
                    "to": parsed_exps[i + 1]["start"].strftime("%Y-%m"),
                    "duration_months": gap_months,
                })

    return {
        "experiences": enriched_exps,
        "total_experience_years": round(total_months / 12.0, 1),
        "experience_gaps_months": gaps,
    }


def pre_process_cv(cv_text: str) -> dict:
    """Point d'entrée principal pour pré-calculer les attributs structurés du CV."""
    sections = _split_sections(cv_text)
    graduation_year = extract_graduation_year(sections.get("Education", ""))

    pre_data = {
        "target_role": sections.get("Target Role", "").strip() or None,
        "age": None,
        "distance_ville_haute_km": None,
        "graduation_year": graduation_year,
        "years_since_graduation": (datetime.now().year - graduation_year) if graduation_year else None,
        "skills": extract_skills(sections.get("Skills", "")),
        "languages": extract_languages(sections.get("Languages", "")),
        "certifications": extract_certifications(sections.get("Certifications", "")),
    }

    dob = parse_date(sections.get("Date of Birth", "").strip())
    if dob:
        pre_data["age"] = compute_age(dob)

    address = sections.get("Address", "").strip()
    if address:
        pre_data["distance_ville_haute_km"] = compute_distance_km(address)
        time.sleep(1)  # Respect stricte du rate limit Nominatim

    return pre_data