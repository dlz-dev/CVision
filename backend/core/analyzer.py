import json
import os
import re
import time
from pathlib import Path

from groq import Groq

_MD_FENCE_RE = re.compile(r"^```(?:json)?", re.MULTILINE)

# Paramètres retry
MAX_RETRIES = 99999      # Nombre maximum de tentatives
RETRY_DELAY_SEC = 3      # Délai (secondes) entre chaque tentative
SKIP_ON_FAILURE = False  # False → le fichier est ignoré (mis en quarantaine) après MAX_RETRIES échecs
                         # True → l'exception est propagée et arrête le pipeline

# Chargement du prompt au démarrage du module
_PROMPT_TEMPLATE = (Path(__file__).parent.parent / "config" / "prompt.txt").read_text(encoding="utf-8")

def _get_groq_config() -> dict:
    """Lit la configuration Groq depuis les variables d'environnement."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "La variable d'environnement GROQ_API_KEY est absente ou vide. "
            "Vérifie ton fichier .env ou les variables du conteneur."
        )
    return {
        "api_key":    api_key,
        "model":      os.getenv("GROQ_MODEL", "llama3-70b-8192"),
        "temperature": float(os.getenv("GROQ_TEMPERATURE", "0.0")),
    }


def extract_cv(cv_text: str) -> dict:
    """Envoie le texte du CV optimisé au LLM (Groq) et retourne un dictionnaire formaté."""
    groq_config = _get_groq_config()
    client = Groq(api_key=groq_config["api_key"])
    prompt_content = _PROMPT_TEMPLATE.replace("{cv_text}", cv_text)

    last_exception = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=groq_config["model"],
                messages=[{"role": "user", "content": prompt_content}],
                temperature=groq_config["temperature"],
            )
            raw_response = _MD_FENCE_RE.sub("", response.choices[0].message.content.strip()).strip()
            return json.loads(raw_response)

        except json.JSONDecodeError as e:
            print(f"  [Tentative {attempt}/{MAX_RETRIES}] Réponse non sérialisable (JSONDecodeError) : {e}")
            last_exception = e
        except Exception as e:
            print(f"  [Tentative {attempt}/{MAX_RETRIES}] Erreur réseau ou API (Groq) : {e}")
            last_exception = e

        if attempt < MAX_RETRIES:
            print(f"  Nouvel essai dans {RETRY_DELAY_SEC}s...")
            time.sleep(RETRY_DELAY_SEC)

    print(f"  Échec complet après {MAX_RETRIES} tentatives.")

    if SKIP_ON_FAILURE:
        raise RuntimeError(
            f"Extraction JSON impossible après {MAX_RETRIES} tentatives. "
            f"Dernière erreur : {last_exception}"
        ) from last_exception

    raise last_exception
