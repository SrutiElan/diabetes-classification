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


st.subheader("Please input information to predict if you have diabetes")

# Top: two rows of four inputs (full width)
row1 = st.columns(4)
row2 = st.columns(4)

# Row 1 inputs
pregnancies = row1[0].number_input(
    "How Many Pregnancies have you had?", min_value=0, step=1
)
glucose = row1[1].number_input(
    "What is your glucose level(mg/Dl)? ", min_value=0, step=1
)
blood_pressure = row1[2].number_input(
    "What is your diastolic blood pressure (mm Hg)?", min_value=0
)
skin_thickness = row2[2].number_input(
    "What is your triceps skin fold thickness in millimeters (default value is average, in case you don't know)?",
    min_value=0.0,
    value=17.0,
)

# Row 2 inputs
insulin = row1[3].number_input("What are your insulin levels? (mu U/ml)")
bmi = row2[1].number_input(
    "What is your BMI (Body Mass Index)?",
    min_value=0.0,
    value=25.0,
)
diabetes_pedigree_function = row2[3].number_input(
    "What is your diabetes pedigree function. (The likelihood you have diabetes based on your family history)",
    min_value=0.0,
    step=0.1,
)
age = row2[0].number_input("How old are you?", min_value=0, step=1)


# Separator and bottom area for the model
st.markdown("---")
bottom_left, bottom_right = st.columns(2)

# Optionally use the left bottom column for notes or model info
with bottom_left:
    # Place the Predict button all the way to the left side of the page
    predict_btn = st.button("Predict Diabetes")

with bottom_right:
    # Result placeholder lives in the right column
    result_placeholder = bottom_right.empty()

    if predict_btn:
        result_placeholder.info("Running model...")
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

            # Try scaling, fall back to raw features on failure
            try:
                scaled_features = scaler.transform(features)
            except Exception:
                scaled_features = features

            # Predict
            pred = model.predict(scaled_features)
            proba = None
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(scaled_features)[0, 1]
                except Exception:
                    proba = None

            # Show result at the bottom_right
            if int(pred[0]) == 1:
                if proba is not None:
                    result_placeholder.success(f"Prediction: Positive for diabetes. Probability: {proba:.2f}")
                else:
                    result_placeholder.success("Prediction: Positive for diabetes.")
            else:
                if proba is not None:
                    result_placeholder.success(f"Prediction: Negative for diabetes. Probability: {proba:.2f}")
                else:
                    result_placeholder.success("Prediction: Negative for diabetes.")

        except FileNotFoundError as e:
            result_placeholder.error(f"Model/scaler file not found: {e}")
        except Exception as e:
            result_placeholder.error(f"An error occurred while predicting: {e}")
