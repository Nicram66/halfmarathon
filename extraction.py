"""Wyłuskiwanie danych biegacza z tekstu za pomocą OpenAI + śledzenie w Langfuse."""
import json
import os
import re

from langfuse import get_client, observe
from langfuse.openai import OpenAI  # drop-in replacement: automatycznie loguje wywołania do Langfuse

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """Jesteś asystentem wyłuskującym dane biegacza z tekstu po polsku.
Zwróć WYŁĄCZNIE obiekt JSON o polach:
- "gender": "M" (mężczyzna) lub "K" (kobieta); null jeśli brak informacji,
- "age": wiek w latach (liczba całkowita); null jeśli brak informacji,
- "time_5km": czas na 5 km w formacie "MM:SS" lub "H:MM:SS"; null jeśli brak informacji.

Zasady:
- NIE zgaduj ani nie wymyślaj brakujących danych – wpisz null.
- Płeć możesz wywnioskować z jednoznacznych form gramatycznych (np. "jestem biegaczką" -> "K", "mam 30 lat, biegacz" -> "M").
- Jeśli podano rok urodzenia zamiast wieku, wylicz wiek względem roku 2026.
- Czas typu "25 minut 30 sekund" lub "pod 25 minut" zapisz jako "25:30"/"25:00".
- Tempo na km (np. "5:00/km") NIE jest czasem na 5 km – przelicz je (x5) tylko jeśli jasne, że to tempo.
"""

REQUIRED_FIELDS = {
    "gender": "płeć (M/K)",
    "age": "wiek",
    "time_5km": "czas na 5 km",
}


def parse_time_to_seconds(value):
    """'24:30' -> 1470, '1:02:10' -> 3730. Zwraca None gdy format niepoprawny."""
    if value is None:
        return None
    parts = re.fullmatch(r"\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*", str(value))
    if not parts:
        return None
    a, b, c = parts.groups()
    if c is None:  # MM:SS
        return int(a) * 60 + int(b)
    return int(a) * 3600 + int(b) * 60 + int(c)


def normalize_and_validate(raw: dict) -> dict:
    """Zamienia surowy JSON z LLM na dane gotowe dla modelu i zwraca listę braków/błędów."""
    gender = raw.get("gender")
    gender = gender.upper() if isinstance(gender, str) else None
    gender = gender if gender in ("M", "K") else None

    age = raw.get("age")
    try:
        age = int(age) if age is not None else None
    except (TypeError, ValueError):
        age = None

    time_s = parse_time_to_seconds(raw.get("time_5km"))

    missing, problems = [], []
    if gender is None:
        missing.append(REQUIRED_FIELDS["gender"])
    if age is None:
        missing.append(REQUIRED_FIELDS["age"])
    elif not 14 <= age <= 90:
        problems.append(f"Wiek {age} jest poza zakresem 14–90 lat.")
    if time_s is None:
        missing.append(REQUIRED_FIELDS["time_5km"])
    elif not 12 * 60 <= time_s <= 60 * 60:
        problems.append("Czas na 5 km powinien mieścić się między 12:00 a 60:00 minut.")

    return {
        "gender": gender,
        "age": age,
        "time_5km_s": time_s,
        "missing": missing,
        "problems": problems,
        "ok": not missing and not problems,
    }


@observe(name="extract_runner_data")
def extract_runner_data(user_text: str) -> dict:
    """Wywołuje LLM, waliduje wynik i zapisuje score'y w Langfuse. Zwraca dane + trace_id."""
    langfuse = get_client()
    client = OpenAI()  # klucz z OPENAI_API_KEY

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        name="llm_extraction",
    )

    try:
        raw = json.loads(response.choices[0].message.content)
        json_valid = 1
    except (json.JSONDecodeError, TypeError):
        raw, json_valid = {}, 0

    result = normalize_and_validate(raw)
    result["raw"] = raw
    result["trace_id"] = langfuse.get_current_trace_id()

    # metryki skuteczności LLM'a w Langfuse
    langfuse.update_current_trace(input=user_text, output=raw, tags=["half-marathon-app"])
    langfuse.score_current_trace(name="json_valid", value=json_valid)
    langfuse.score_current_trace(name="extraction_complete", value=1 if not result["missing"] else 0)
    langfuse.score_current_trace(name="values_in_range", value=1 if not result["problems"] else 0)
    langfuse.score_current_trace(name="fields_extracted", value=3 - len(result["missing"]))
    return result


def send_user_feedback(trace_id: str, value: int, comment: str | None = None):
    """Ocena użytkownika (1 = poprawnie wyłuskane dane, 0 = błędnie)."""
    if not trace_id:
        return
    langfuse = get_client()
    langfuse.create_score(trace_id=trace_id, name="user_feedback", value=value,
                          data_type="NUMERIC", comment=comment)
    langfuse.flush()
