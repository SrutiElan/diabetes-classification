import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd

# So that imports work
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import load_processed_data
from src.evaluation import evaluate_model, plot_roc_curve, plot_confusion_matrix

def load_processed_data():
    """
    Load pre-processed from EDA notebook
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    try:
        train_df = pd.read_csv('../data/processed/train.csv')
        test_df = pd.read_csv('../data/processed/test.csv')
    except FileNotFoundError:
        print("Error, make sure that you're running from the notebooks/ directory")
        raise
    
    # Split features and target
    X_train = train_df.drop('Outcome', axis=1).values
    y_train = train_df['Outcome'].values
    
    X_test = test_df.drop('Outcome', axis=1).values
    y_test = test_df['Outcome'].values
    
    feature_names = train_df.drop('Outcome', axis=1).columns.tolist()
    
    print(f"Data loaded successfully!")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {len(feature_names)}")
    
    return X_train, X_test, y_train, y_test, feature_names

# Load data
X_train, X_test, y_train, y_test, feature_names = load_processed_data()

rfc = RandomForestClassifier()


rf_param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [None, 5, 10],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

rfc_cv = GridSearchCV(rfc, rf_param_grid, cv=5)
rfc_cv.fit(X_train, y_train)

print(f"Best Score: {rfc_cv.best_score_}")
print(f"Best Parameters: {rfc_cv.best_params_}")

best_rfc = rfc_cv.best_estimator_
y_pred = best_rfc.predict(X_test)

# Evaluation functions
evaluate_model(best_rfc, X_train, X_test, y_train, y_test, "Random Forest")
plot_confusion_matrix(y_pred, y_test, "Random Forest")
plot_roc_curve(y_pred, y_test, "Random Forest")