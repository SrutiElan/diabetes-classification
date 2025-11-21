import os
import sys

# So that imports work
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from xgboost import XGBClassifier
from src.evaluation import evaluate_model, plot_roc_curve, plot_confusion_matrix
from src.utils import load_processed_data
from sklearn.model_selection import GridSearchCV
import numpy as np

X_train, X_test, y_train, y_test, feature_names = load_processed_data()

xgb_clf = XGBClassifier(objective='binary:logistic')
param_grid = {
    "max_depth": np.arange(3,15,2),
    "n_estimators": np.arange(0, 400, 100),
    "learning_rate":  [0.01, 0.05, 0.1]
}

xgb_clf_cv = GridSearchCV(xgb_clf, param_grid, cv=3)
xgb_clf_cv.fit(X_train, y_train)

print("Best parameters found: ", xgb_clf_cv.best_params_)
print("Best ROC AUC score: ", xgb_clf_cv.best_score_)

best_xgb_clf = xgb_clf_cv.best_estimator_

y_pred = best_xgb_clf.predict(X_test)

evaluate_model(best_xgb_clf, X_train, X_test, y_train, y_test, "XGBoost Classifier")

plot_confusion_matrix(y_pred, y_test, "XGBoost Classifier")

plot_roc_curve(y_pred, y_test, "XGBoost Classifier")