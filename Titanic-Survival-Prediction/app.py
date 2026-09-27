import streamlit as st
import pandas as pd
import joblib

# Page settings
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# Load trained model
model = joblib.load("models/titanic_best_model.pkl")


# Title
st.title("🚢 Titanic Survival Predictor")

st.subheader("Machine Learning Based Survival Prediction")

st.write(
    "Enter the passenger information below "
    "to predict the probability of survival."
)

st.divider()


# Passenger Information
st.header("Passenger Information")

col1, col2 = st.columns(2)


with col1:

    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

    sex = st.selectbox(
        "Sex",
        ["male", "female"]
    )

    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=100.0,
        value=30.0
    )

    sibsp = st.number_input(
        "Siblings / Spouses",
        min_value=0,
        max_value=10,
        value=0
    )


with col2:

    parch = st.number_input(
        "Parents / Children",
        min_value=0,
        max_value=10,
        value=0
    )

    fare = st.number_input(
        "Fare",
        min_value=0.0,
        max_value=600.0,
        value=30.0
    )

    embarked = st.selectbox(
        "Port of Embarkation",
        ["S", "C", "Q"]
    )


st.divider()


# Prediction
if st.button(
    "🔮 Predict Survival",
    use_container_width=True
):

    input_data = pd.DataFrame({

        "Pclass": [pclass],

        "Sex": [sex],

        "Age": [age],

        "SibSp": [sibsp],

        "Parch": [parch],

        "Fare": [fare],

        "Embarked": [embarked]
    })


    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    survival_probability = probability[1]

    death_probability = probability[0]


    # Display result
    st.divider()

    st.header("Prediction Result")


    if prediction == 1:

        st.success(
            "✅ Passenger is predicted to SURVIVE."
        )

    else:

        st.error(
            "❌ Passenger is predicted NOT TO SURVIVE."
        )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Survival Probability",
            f"{survival_probability:.2%}"
        )


    with col2:

        st.metric(
            "Not Survival Probability",
            f"{death_probability:.2%}"
        )


    st.write("Survival Probability")

    st.progress(
        float(survival_probability)
    )


st.divider()

st.caption(
    "Titanic Survival Prediction | "
    "Machine Learning + Streamlit"
)