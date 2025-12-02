from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle


column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# Read the CSV, skipping the header row

# Run this from the working directory
raw_diabetes_data = pd.read_csv("data/raw/pima-indians-diabetes.csv", names=column_names, skiprows=1)
sc_X = StandardScaler()
scalar =  sc_X.fit(raw_diabetes_data.drop(["Outcome"],axis = 1),)

# Pickling the scalar 
with open("./models/scaler.pkl", "wb") as f:
    pickle.dump(scalar, f)
