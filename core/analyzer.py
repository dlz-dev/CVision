import json
import re
import time
from pathlib import Path

from groq import Groq

_MD_FENCE_RE = re.compile(r"^```(?:json)?", re.MULTILINE)

# Paramètres retry
MAX_RETRIES = 99999      # Nombre maximum de tentatives
RETRY_DELAY_SEC = 3      # Délai (secondes) entre chaque tentative
SKIP_ON_FAILURE = False  # True → le fichier est ignoré (mis en quarantaine) après MAX_RETRIES échecs
                         # False → l'exception est propagée et arrête le pipeline

# Chargement du prompt au démarrage du module
_PROMPT_TEMPLATE = Path("config/prompt.txt").read_text(encoding="utf-8")


def extract_cv(cv_text: str, config: dict) -> dict:
    """Envoie le texte du CV optimisé au LLM (Groq) et retourne un dictionnaire formaté."""
    client = Groq(api_key=config["api"]["api_key"])
    prompt_content = _PROMPT_TEMPLATE.replace("{cv_text}", cv_text)

    last_exception = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=config["api"]["model"],
                messages=[{"role": "user", "content": prompt_content}],
                temperature=config["api"]["temperature"],
            )

            # Nettoyage et conversion de la réponse Markdown en JSON
            raw_response = _MD_FENCE_RE.sub("", response.choices[0].message.content.strip()).strip()
            return json.loads(raw_response)

        except json.JSONDecodeError as e:
            print(f"  [Tentative {attempt}/{MAX_RETRIES}] Réponse non sérialisable (JSONDecodeError) : {e}")
            last_exception = e
        except Exception as e:
            print(f"  [Tentative {attempt}/{MAX_RETRIES}] Erreur réseau ou API (Groq) : {e}")
            last_exception = e

        if attempt < MAX_RETRIES:
            print(f"  Nouvel essai programmé dans {RETRY_DELAY_SEC}s...")
            time.sleep(RETRY_DELAY_SEC)

    print(f"  Échec complet après {MAX_RETRIES} tentatives.")

    if SKIP_ON_FAILURE:
        raise RuntimeError(
            f"Extraction JSON impossible après {MAX_RETRIES} tentatives. "
            f"Dernière erreur remontée : {last_exception}"
        ) from last_exception

    raise last_exception