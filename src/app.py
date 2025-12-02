import streamlit as st
import pickle
import numpy as np

st.title("AISERPANTS")

# Note: This grid is not going to space itself depending on how much 
# text wrap there is. So it's the developers job to make sure 
# columns come out evenly spaced out
def create_grid(n, m, container):
    grid = []

    for _ in range(n):
        tmp = []
        for c in container.columns(m):
            tmp.append(c)
        grid.append(tmp)

    return grid


col1, col2 = st.columns(2, vertical_alignment="center")
col1.subheader("Please input information to predict if you have diabetes")
col1_grid = create_grid(4, 2, col1)

# Getting the user's data
pregnancies = col1_grid[0][0].number_input(
    "How Many Pregnancies have you had?", min_value=0, step=1
)
glucose = col1_grid[0][1].number_input(
    "What is your glucose level(mg/Dl)? ", min_value=0, step=1
)
blood_pressure = col1_grid[1][0].number_input(
    "What is your diastolic blood pressure (mm Hg)?", min_value=0
)
skin_thickness = col1_grid[3][0].number_input(
    "What is your triceps skin fold thickness in millimeters( default value is average, in case you don't know)?",
    min_value=0.0,
    value=17.0,
)
insulin = col1_grid[1][1].number_input("What are your insulin levels? (mu U/ml)")
diabetes_pedigree_function = col1_grid[3][1].number_input(
    "What is your diabetes pedigree function. (The likelihood you have diabetes based on your family history)",
    min_value=0.0,
    step=0.1,
)
age = col1_grid[2][0].number_input("How old are you?", min_value=0, step=1)
# BMI input placed in the unused cell
bmi = col1_grid[2][1].number_input(
    "What is your BMI (Body Mass Index)?",
    min_value=0.0,
    value=25.0,
)

# Prediction button
predict_btn = col2.button("Predict Diabetes")

if predict_btn:
    try:
        # Load scaler and model
        with open("./models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        with open("./models/xgboostClassifier.pkl", "rb") as f:
            model = pickle.load(f)

        features = np.array([
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree_function,
            age,
        ]).reshape(1, -1)

        scaled_features = scaler.transform(features)
        # Scale then predict
        pred = model.predict(scaled_features)
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(scaled_features)[0, 1]

        if int(pred[0]) == 1:
            col2.success(f"Prediction: Positive for diabetes. Probability: {proba:.2f}" if proba is not None else "Prediction: Positive for diabetes.")
        else:
            col2.success(f"Prediction: Negative for diabetes. Probability: {proba:.2f}" if proba is not None else "Prediction: Negative for diabetes.")

    except FileNotFoundError as e:
        col2.error(f"Model/scaler file not found: {e}")
    except Exception as e:
        col2.error(f"An error occurred while predicting: {e}")
