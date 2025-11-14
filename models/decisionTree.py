import pandas as pd
import os
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# So that imports work
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import load_processed_data
from src.evaluation import evaluate_model

X_train, X_test, y_train, y_test, feature_names = load_processed_data()

dtc = DecisionTreeClassifier()
# gini and entropy are both measures of impurity
# Entropy measures randomness
# gini is probability of misclassification
# according to a stack overflow post, entropy seems to be interchangable with log_loss
# so I will not include it here
param_grid = {"criterion": ["gini", "entropy"],
              "max_depth": np.arange(3,15,2), # The deeper you go, the more likely you are to overfit
              "min_samples_split": np.arange(2,5) # the minimum number of samples required to allow a "split" in the tree
              }
dtc_cv = GridSearchCV(dtc, param_grid, cv=5)
dtc_cv.fit(X_train, y_train)

print(f"Best Score: {dtc_cv.best_score_}")
print(f"Best Parameters: {dtc_cv.best_params_}")

best_dtc = dtc_cv.best_estimator_

evaluate_model(best_dtc, X_train, X_test, y_train, y_test, "Decision Tree")
