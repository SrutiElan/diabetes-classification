import os
import sys
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
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


gbc = GradientBoostingClassifier()

param_grid = {
    "n_estimators": [50, 100, 150],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [2, 3, 4],
    "min_samples_split": [2, 5, 10]
}
gb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 2, 4],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2],
    'subsample': [0.8, 1.0, 1.5]
}

gbc_cv = GridSearchCV(gbc, gb_param_grid, cv=5)
gbc_cv.fit(X_train, y_train)

print(f"Best Score: {gbc_cv.best_score_}")
print(f"Best Parameters: {gbc_cv.best_params_}")

best_gbc = gbc_cv.best_estimator_
y_pred = best_gbc.predict(X_test)

evaluate_model(best_gbc, X_train, X_test, y_train, y_test, "Gradient Boosting")
plot_confusion_matrix(y_pred, y_test, "Gradient Boosting")
plot_roc_curve(y_pred, y_test, "Gradient Boosting")
