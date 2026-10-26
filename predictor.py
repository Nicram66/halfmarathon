"""Ładowanie modelu (Digital Ocean Spaces -> fallback lokalny) i predykcja."""
import io
import os

import boto3
import joblib
import pandas as pd

MODEL_KEY = "models/latest.joblib"
LOCAL_MODEL_PATH = os.path.join("model", "latest.joblib")


def _load_from_spaces():
    key = os.getenv("DO_SPACES_KEY")
    secret = os.getenv("DO_SPACES_SECRET")
    bucket = os.getenv("DO_SPACES_BUCKET")
    region = os.getenv("DO_SPACES_REGION", "sfo3")
    if not all([key, secret, bucket]):
        return None
    s3 = boto3.session.Session().client(
        "s3",
        region_name=region,
        endpoint_url=f"https://{region}.digitaloceanspaces.com",
        aws_access_key_id=key,
        aws_secret_access_key=secret,
    )
    obj = s3.get_object(Bucket=bucket, Key=MODEL_KEY)
    return joblib.load(io.BytesIO(obj["Body"].read()))


def load_model():
    """Zwraca (artifact, source). artifact = {'model': Pipeline, 'metadata': dict}."""
    try:
        artifact = _load_from_spaces()
        if artifact is not None:
            return artifact, "Digital Ocean Spaces"
    except Exception as exc:  # np. brak sieci / złe klucze
        print(f"Nie udało się pobrać modelu ze Spaces: {exc}")
    if os.path.exists(LOCAL_MODEL_PATH):
        return joblib.load(LOCAL_MODEL_PATH), "plik lokalny"
    raise FileNotFoundError("Brak modelu – uruchom notebook train_model.ipynb.")


def predict_seconds(artifact, gender: str, age: int, time_5km_s: int) -> int:
    x = pd.DataFrame([{"czas_5km_s": time_5km_s, "plec": gender, "wiek": age}])
    x = x[artifact["metadata"]["features"]]
    return int(artifact["model"].predict(x)[0])


def format_hms(seconds: int) -> str:
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"
