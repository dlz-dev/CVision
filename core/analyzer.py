import json
import re
import time
from groq import Groq

_MD_FENCE_RE = re.compile(r"^```(?:json)?", re.MULTILINE)

# Paramètres retry
MAX_RETRIES = 99999      # Nombre maximum de tentatives
RETRY_DELAY_SEC = 3      # Délai (secondes) entre chaque tentative
SKIP_ON_FAILURE = False  # True → le fichier est ignoré (mis en quarantaine) après MAX_RETRIES échecs
                         # False → l'exception est propagée et arrête le pipeline

_PROMPT_TEMPLATE = Path("config/prompt.txt").read_text(encoding="utf-8")


def extract_cv(cv_text: str, config: dict) -> dict:
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

            raw = _MD_FENCE_RE.sub("", response.choices[0].message.content.strip()).strip()
            return json.loads(raw)

        except json.JSONDecodeError as e:
            print(f"  [Tentative {attempt}/{MAX_RETRIES}] Réponse Groq non valide (JSONDecodeError) : {e}")
            last_exception = e

        except Exception as e:
            print(f"  [Tentative {attempt}/{MAX_RETRIES}] Erreur Groq : {e}")
            last_exception = e

        if attempt < MAX_RETRIES:
            print(f"  Nouvel essai dans {RETRY_DELAY_SEC}s...")
            time.sleep(RETRY_DELAY_SEC)

    # Toutes les tentatives ont échoué
    print(f"  Échec après {MAX_RETRIES} tentative(s).")

    if SKIP_ON_FAILURE:
        # On lève une exception générique que main.py attrapera pour mettre le fichier en quarantaine
        raise RuntimeError(
            f"Groq n'a pas retourné de JSON valide après {MAX_RETRIES} tentatives. "
            f"Dernière erreur : {last_exception}"
        )
    else:
        # On propage l'exception originale pour arrêter le pipeline
        raise last_exception