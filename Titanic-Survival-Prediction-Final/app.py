"""Streamlit dashboard for Titanic survival prediction."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "titanic_best_model.pkl"
RESULTS_PATH = ROOT / "results" / "model_comparison.csv"

st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢", layout="centered")

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #07111f 0%, #12304a 55%, #1f5f78 100%); color: white; }
.block-container { max-width: 900px; padding-top: 2rem; }
.card { background: rgba(255,255,255,.10); padding: 1.4rem; border-radius: 18px; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

st.title("🚢 Titanic Survival Prediction")
st.write("Machine Learning dashboard for predicting passenger survival probability.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.form("prediction_form"):
    st.subheader("Passenger Information")
    c1, c2 = st.columns(2)
    with c1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
        sibsp = st.number_input("Siblings / Spouses Aboard", min_value=0, max_value=10, value=0, step=1)
    with c2:
        parch = st.number_input("Parents / Children Aboard", min_value=0, max_value=10, value=0, step=1)
        fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0, step=0.5)
        embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"], format_func=lambda x: {"S":"Southampton (S)","C":"Cherbourg (C)","Q":"Queenstown (Q)"}[x])
    submitted = st.form_submit_button("Predict Survival", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([{
        "Pclass": pclass, "Sex": sex, "Age": age, "SibSp": sibsp,
        "Parch": parch, "Fare": fare, "Embarked": embarked
    }])
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0, 1])

    st.divider()
    if prediction == 1:
        st.success(f"Prediction: Survived\n\nEstimated survival probability: {probability:.1%}")
    else:
        st.warning(f"Prediction: Did not survive\n\nEstimated survival probability: {probability:.1%}")
    st.progress(probability, text=f"Survival probability: {probability:.1%}")

if RESULTS_PATH.exists():
    with st.expander("View model comparison"):
        st.dataframe(pd.read_csv(RESULTS_PATH), use_container_width=True, hide_index=True)

st.caption("Titanic Survival Prediction • Streamlit Machine Learning Project")
