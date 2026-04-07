import json
import re
from pathlib import Path
from groq import Groq

_MD_FENCE_RE = re.compile(r"^```(?:json)?", re.MULTILINE)


def load_prompt() -> str:
    return Path("config/prompt.txt").read_text(encoding="utf-8")


def extract_cv(cv_text: str, config: dict) -> dict:
    client = Groq(api_key=config["api"]["api_key"])

    prompt_content = load_prompt().replace("{cv_text}", cv_text)

    response = client.chat.completions.create(
        model=config["api"]["model"],
        messages=[{"role": "user", "content": prompt_content}],
        temperature=config["api"]["temperature"],
    )

    raw = _MD_FENCE_RE.sub("", response.choices[0].message.content.strip()).strip()
    return json.loads(raw)