"""
Shared utility functions for diabetes classification project
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
def load_processed_data():
    """
    Load pre-processed data that Sruti created in the EDA notebook
    This is the MAIN function everyone uses!
    No need to clean data yourself - it's already done.
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    try:
        train_df = pd.read_csv('../data/processed/train.csv')
        test_df = pd.read_csv('../data/processed/test.csv')
    except FileNotFoundError:
        print("❌ Error: Processed data not found!")
        print("   Ask Sruti to run the EDA notebook first (notebooks/00_EDA.ipynb)")
        print("   Or check that you're running from the notebooks/ directory")
        raise
    
    # Split features and target
    X_train = train_df.drop('Outcome', axis=1).values
    y_train = train_df['Outcome'].values
    
    X_test = test_df.drop('Outcome', axis=1).values
    y_test = test_df['Outcome'].values
    
    feature_names = train_df.drop('Outcome', axis=1).columns.tolist()
    
    print(f"✅ Data loaded successfully!")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Features: {len(feature_names)}")
    
    return X_train, X_test, y_train, y_test, feature_names

def load_and_clean_data(filepath='../data/pima-indians-diabetes.csv'):
    # Load and clean the Pima diabetes dataset
    # Returns:  X_train, X_test, y_train, y_test, scaler
    
    # Load data
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv(filepath, names=column_names)
    
    # Handle missing values (zeros)
    zero_not_allowed = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_not_allowed:
        df[col] = df[col].replace(0, np.nan)
        df[col].fillna(df[col].median(), inplace=True)
    
    # Split features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """
    Evaluate a model and print metrics
    
    Returns:
        dict with metrics
    """
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    # Get probabilities for ROC-AUC
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    else:
        roc_auc = None
    
    print(f"\n{'='*50}")
    print(f"{model_name} Performance")
    print(f"{'='*50}")
    print(f"Training Accuracy:   {train_acc:.3f}")
    print(f"Test Accuracy:       {test_acc:.3f}")
    if roc_auc:
        print(f"ROC-AUC Score:       {roc_auc:.3f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred_test))
    
    return {
        'model_name': model_name,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'roc_auc': roc_auc
    }