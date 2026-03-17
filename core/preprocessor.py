import re
import time
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import json

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

# Regex compilé pour découper toutes les sections en une seule passe
_SECTION_SPLIT_RE = re.compile(
    r'^(' + '|'.join(re.escape(h) for h in _SECTION_HEADERS) + r'):',
    re.MULTILINE,
)

# Regex compilés pour clean_cv_text_for_llm
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

LANGUAGE_LEVEL_SCORE: dict[str, int] = {
    "A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6,
}

# Dict plat keyword→score (évite la double boucle à chaque appel)
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

# Formats de date du plus au moins fréquent dans les CVs
_DATE_FORMATS = ['%Y-%m', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%m/%Y']


# ---------------------------------------------------------------------------
# Parsing du texte brut
# ---------------------------------------------------------------------------

def _split_sections(cv_text: str) -> dict[str, str]:
    """
    Découpe le CV en sections en une seule passe regex.
    Retourne un dict {header: contenu}.
    Remplace tous les appels séparés à extract_section / extract_field.
    """
    sections: dict[str, str] = {}
    matches = list(_SECTION_SPLIT_RE.finditer(cv_text))
    for i, m in enumerate(matches):
        key   = m.group(1)
        start = m.end()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(cv_text)
        sections[key] = cv_text[start:end].strip()
    return sections


def parse_date(date_str: str) -> datetime | None:
    """Parse une date en essayant les formats du plus au moins fréquent."""
    s = date_str.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def extract_email(cv_text: str) -> str | None:
    """Extrait l'email même entouré de markdown."""
    m = _EMAIL_RE.search(cv_text)
    return m.group(0) if m else None


def compute_age(dob: datetime) -> int:
    now = datetime.now()
    return now.year - dob.year - ((now.month, now.day) < (dob.month, dob.day))


def compute_distance_km(address: str) -> float | None:
    """Géocode une adresse et retourne la distance en km depuis Luxembourg Ville."""
    search_address = address
    if "USA" in address.upper():
        parts = [p.strip() for p in address.split(',')]
        if len(parts) >= 2:
            search_address = f"{parts[-2]}, USA"
    try:
        location = geolocator.geocode(search_address, timeout=10)
        if location:
            return round(geodesic((location.latitude, location.longitude), LUXEMBOURG_COORDS).kilometers, 2)
    except Exception as e:
        print(f"Géolocalisation échouée pour '{search_address}': {e}")
    return None


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def score_language_level(level: str) -> int | None:
    """Convertit un niveau CECRL (A1–C2) en score entier de 1 à 6."""
    return LANGUAGE_LEVEL_SCORE.get(level.strip().upper()) if level else None


def score_education(degree: str) -> int | None:
    """
    Retourne un score ordinal (1–5) selon le niveau du diplôme.
    Utilise un dict plat précompilé pour éviter la double boucle à chaque appel.
    """
    if not degree:
        return None
    d = degree.lower()
    for kw, score in _DEGREE_KW_SCORE.items():
        if kw in d:
            return score
    return None


# ---------------------------------------------------------------------------
# Extraction des sections (acceptent le contenu brut, pas le CV entier)
# ---------------------------------------------------------------------------

def extract_skills(section_text: str) -> list[str]:
    skills = []
    for line in section_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        content = _SKILL_LABEL_RE.sub('', line).strip()
        for skill in content.split(','):
            s = skill.strip()
            if s:
                skills.append(s)
    return skills


def extract_languages(section_text: str) -> list[dict]:
    languages = []
    for m in _LANG_RE.finditer(section_text):
        level = m.group(2).strip()
        languages.append({
            "language": m.group(1).strip(),
            "level":    level,
            "score":    score_language_level(level),
        })
    return languages


def extract_certifications(section_text: str) -> list[dict]:
    certifications = []
    for line in section_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        m = _CERT_YEAR_RE.match(line)
        certifications.append(
            {"name": m.group(1).strip(), "year": int(m.group(2))} if m
            else {"name": line, "year": None}
        )
    return certifications


def extract_graduation_year(section_text: str) -> int | None:
    years = _YEAR_RE.findall(section_text)
    return int(years[-1]) if years else None


def clean_cv_text_for_llm(cv_text: str) -> str:
    """
    N'envoie au LLM que les sections Education et Experience.
    Utilise des patterns précompilés au niveau module.
    """
    parts = []
    for section, pattern in _LLM_SECTION_PATTERNS.items():
        m = pattern.search(cv_text)
        if m:
            parts.append(f"{section}:\n{m.group(1).strip()}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Calculs sur les expériences
# ---------------------------------------------------------------------------

def compute_experience_metrics(experiences: list) -> dict:
    """
    Prend la liste brute des expériences du LLM et calcule :
    - La durée de chaque poste (en mois)
    - Les trous de plus de 1 mois entre deux postes
    - L'expérience totale en années
    """
    if not experiences:
        return {"experiences": [], "total_experience_years": 0.0, "experience_gaps_months": []}

    now = datetime.now()
    total_months = 0
    gaps: list[dict] = []
    enriched_exps: list[dict] = []
    parsed_exps: list[dict] = []

    for exp in experiences:
        start_date = parse_date(exp.get("start", ""))
        end_str    = exp.get("end", "").lower()
        end_date   = now if end_str == "present" else parse_date(end_str)

        if start_date:
            parsed_exps.append({"raw": exp, "start": start_date, "end": end_date or start_date})
        else:
            enriched_exps.append({**exp, "duration_months": None})

    parsed_exps.sort(key=lambda x: x["start"])

    for i, exp in enumerate(parsed_exps):
        duration_months = round((exp["end"] - exp["start"]).days / 30.44)
        total_months   += duration_months
        enriched_exps.append({**exp["raw"], "duration_months": duration_months})

        if i < len(parsed_exps) - 1:
            gap_months = round((parsed_exps[i + 1]["start"] - exp["end"]).days / 30.44)
            if gap_months > 1:
                gaps.append({
                    "from":            exp["end"].strftime("%Y-%m"),
                    "to":              parsed_exps[i + 1]["start"].strftime("%Y-%m"),
                    "duration_months": gap_months,
                })

    return {
        "experiences":            enriched_exps,
        "total_experience_years": round(total_months / 12, 1),
        "experience_gaps_months": gaps,
    }


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def pre_process_cv(cv_text: str) -> dict:
    """
    Extrait toutes les données du CV en une seule passe de parsing,
    puis calcule les champs dérivés (âge, distance, années depuis diplôme).
    """
    # Une seule passe pour découper toutes les sections
    sections = _split_sections(cv_text)

    pre_data: dict = {
        "target_role":             sections.get("Target Role", "").strip() or None,
        "age":                     None,
        "distance_ville_haute_km": None,
        "graduation_year":         extract_graduation_year(sections.get("Education", "")),
        "years_since_graduation":  None,
        "skills":                  extract_skills(sections.get("Skills", "")),
        "languages":               extract_languages(sections.get("Languages", "")),
        "certifications":          extract_certifications(sections.get("Certifications", "")),
    }

    dob_str = sections.get("Date of Birth", "").strip()
    if dob_str:
        dob = parse_date(dob_str)
        if dob:
            pre_data["age"] = compute_age(dob)

    gy = pre_data["graduation_year"]
    pre_data["years_since_graduation"] = (datetime.now().year - gy) if gy else None

    address = sections.get("Address", "").strip() or None
    if address:
        pre_data["distance_ville_haute_km"] = compute_distance_km(address)
        time.sleep(1)  # Respect du rate limit Nominatim

    return pre_data