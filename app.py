import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from extraction import extract_runner_data, send_user_feedback  # noqa: E402
from predictor import format_hms, load_model, predict_seconds  # noqa: E402

st.set_page_config(page_title="Estymator czasu półmaratonu", page_icon="🏃", layout="centered")


@st.cache_resource(show_spinner="Ładowanie modelu…")
def get_model():
    return load_model()


st.title("🏃 Estymator czasu półmaratonu")
st.write(
    "Przedstaw się własnymi słowami i podaj **płeć, wiek oraz czas na 5 km** – "
    "oszacuję, w jakim czasie przebiegniesz półmaraton."
)

artifact, source = get_model()
meta = artifact["metadata"]

user_text = st.text_area(
    "Opowiedz o sobie",
    placeholder="Cześć, jestem Ania, mam 32 lata i swoje 5 km biegam w 27 minut i 30 sekund.",
    height=140,
)

if st.button("Oszacuj czas", type="primary"):
    if not user_text.strip():
        st.warning("Wpisz kilka zdań o sobie.")
    else:
        with st.spinner("Analizuję Twoją wypowiedź…"):
            try:
                st.session_state["result"] = extract_runner_data(user_text)
                st.session_state["feedback_sent"] = False
            except Exception as exc:
                st.session_state.pop("result", None)
                st.error(f"Nie udało się przeanalizować tekstu (OpenAI/Langfuse): {exc}")

result = st.session_state.get("result")
if result:
    if result["missing"]:
        st.warning("Brakuje danych: **" + ", ".join(result["missing"]) + "**. Uzupełnij je w opisie i spróbuj ponownie.")
    if result["problems"]:
        for p in result["problems"]:
            st.error(p)

    if result["ok"]:
        seconds = predict_seconds(artifact, result["gender"], result["age"], result["time_5km_s"])
        st.success(f"Szacowany czas półmaratonu: **{format_hms(seconds)}**")
        c1, c2 = st.columns(2)
        c1.metric("Średnie tempo", f"{seconds / 21.0975 / 60:.2f} min/km".replace(".", ","))
        c2.metric("Typowy błąd modelu", f"± {meta['test_metrics']['MAE_s'] / 60:.1f} min")

    with st.expander("Dane wyłuskane z Twojego opisu"):
        st.json({
            "płeć": result["gender"],
            "wiek": result["age"],
            "czas 5 km [s]": result["time_5km_s"],
        })

    if result.get("trace_id"):
        st.caption("Czy dane zostały wyłuskane poprawnie?")
        fb1, fb2, _ = st.columns([1, 1, 6])
        if not st.session_state.get("feedback_sent"):
            if fb1.button("👍"):
                send_user_feedback(result["trace_id"], 1)
                st.session_state["feedback_sent"] = True
                st.rerun()
            if fb2.button("👎"):
                send_user_feedback(result["trace_id"], 0)
                st.session_state["feedback_sent"] = True
                st.rerun()
        else:
            st.caption("Dziękuję za opinię! ✅")

st.divider()
st.caption(
    f"Model: {meta['model_name']} (wersja {meta['version']}, źródło: {source}) · "
    f"trenowany na {meta['n_samples']} wynikach z Wrocław Półmaraton 2023."
)
