# Estymator czasu półmaratonu (trenowany na podstawie danych z półmaratonu wrocławskiego 2023)

Aplikacja Streamlit: użytkownik opisuje się tekstem (płeć, wiek, czas na 5 km) → OpenAI wyłuskuje dane do JSON → model ML szacuje czas półmaratonu. Wywołania LLM są monitorowane w Langfuse.

## Struktura repo
```
app.py              # UI Streamlit
extraction.py       # OpenAI + Langfuse (ekstrakcja danych, walidacja, score'y)
predictor.py        # ładowanie modelu ze Spaces (fallback: model/latest.joblib)
train_model.ipynb   # pipeline: Spaces -> czyszczenie -> trening -> zapis lokalnie i do Spaces
data/               # CSV (lokalna kopia; źródłem jest Spaces)
model/latest.joblib # lokalna kopia modelu (fallback)
requirements.txt, Procfile, .env.example, .gitignore
```

## Krok 1 – Digital Ocean Spaces
1. Zaloguj się w cloud.digitalocean.com → **Create → Spaces Object Storage**.
2. Wybierz region (np. **Frankfurt – fra1**), nadaj nazwę bucketu (np. `halfmarathon-ml`), *Restrict File Listing* zostaw włączone → **Create**.
3. W bucketcie **Upload** → utwórz folder `data/` i wgraj `halfmarathon_wroclaw_2023__final.csv` (docelowa ścieżka: `data/halfmarathon_wroclaw_2023__final.csv`).
4. Klucze dostępu: menu **API → Spaces Keys → Generate New Key** → nazwa dowolna → skopiuj **Access Key** i **Secret Key** (secret widać tylko raz!).

## Krok 2 – Lokalna konfiguracja
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt jupyter
cp .env.example .env               # uzupełnij wszystkie wartości
```

## Krok 3 – Trening modelu
Uruchom `jupyter notebook train_model.ipynb` → **Run All**. Notebook pobiera CSV ze Spaces, czyści dane, porównuje cechy i modele (MAE, RMSE, MedAE, MAPE, R²) i zapisuje:
* lokalnie: `model/latest.joblib`, `model/half_marathon_model_<timestamp>.joblib`, `model/latest_metadata.json`
* do Spaces: te same pliki w folderze `models/`

Wersja scikit-learn w `requirements.txt` musi być taka sama jak przy treningu (pickle jest wrażliwy na wersje).

## Krok 4 – OpenAI i Langfuse
* **OpenAI:** platform.openai.com → API keys → *Create new secret key* → `OPENAI_API_KEY` (konto musi mieć doładowany budżet).
* **Langfuse:** cloud.langfuse.com → załóż konto → *New project* → *Settings → API Keys → Create* → `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`. `LANGFUSE_HOST` = `https://cloud.langfuse.com` (EU) lub `https://us.cloud.langfuse.com` (US) – zgodnie z regionem projektu.

Co trafia do Langfuse: trace `extract_runner_data` (input, output JSON, wywołanie OpenAI, koszt, latency) oraz score'y: `json_valid`, `extraction_complete`, `values_in_range`, `fields_extracted` i `user_feedback` (👍/👎 w aplikacji).

## Krok 5 – Test lokalny
```bash
streamlit run app.py
```
Przykład: *„Cześć, jestem Ania, mam 32 lata i 5 km biegam w 27 minut i 30 sekund."*

## Krok 6 – Deploy na Digital Ocean App Platform
1. Wrzuć projekt na GitHub (plik `.env` jest w `.gitignore` – **nie commituj kluczy**):
   ```bash
   git init && git add . && git commit -m "init"
   git branch -M main
   git remote add origin https://github.com/<user>/<repo>.git
   git push -u origin main
   ```
2. DigitalOcean → **Create → App Platform** → źródło **GitHub** → wybierz repo i branch `main` (przy pierwszym razie autoryzuj dostęp DO do GitHuba) → *Autodeploy* włączone.
3. Platforma wykryje Pythona (z `requirements.txt`) jako **Web Service**. Edytuj ustawienia komponentu:
   * **Run Command:** `streamlit run app.py --server.port=8080 --server.address=0.0.0.0 --server.headless=true`
   * **HTTP Port:** `8080`
   * **Instance size:** wystarczy najmniejszy (Basic, 512 MB–1 GB RAM).
4. **Environment Variables** (zaznacz *Encrypt* dla sekretów): `DO_SPACES_KEY`, `DO_SPACES_SECRET`, `DO_SPACES_REGION`, `DO_SPACES_BUCKET`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`.
5. **Create Resources** → poczekaj na build (kilka minut) → otwórz adres `https://<nazwa>.ondigitalocean.app`.
6. Jeśli build zgłosi problem z wersją Pythona (scikit-learn 1.8 wymaga Pythona ≥ 3.11), dodaj do repo plik `runtime.txt` z wpisem `python-3.12.x` (wersja dostępna w buildpacku – sprawdź logi builda).

## Aktualizacja modelu
Uruchom ponownie notebook → nowy `models/latest.joblib` trafia do Spaces → w App Platform zrób **Actions → Restart** (model jest cache'owany przy starcie aplikacji).
