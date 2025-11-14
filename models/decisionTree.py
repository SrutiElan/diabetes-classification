import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ..src.utils import *

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

# At last we find the metrics
metrics_dir_path = os.path.join(cwd_path, "models", "decision_tree_metrics")
os.makedirs(metrics_dir_path, exist_ok=True)

# Now lets create a confusion matrix
y_pred = best_dtc.predict(X_test)
dtc_matrix = metrics.confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(dtc_matrix), annot=True, cmap="YlGnBu", fmt="g")
plt.title("Confusion Matrix")
plt.ylabel("Actual Case of Diabetes")
plt.xlabel("Predicted Case of Diabetes")
plt.savefig(os.path.join(metrics_dir_path, "confusion_matrix.png"))

# Now lets create our ROC curve 
plt.clf()
y_pred_proba = best_dtc.predict_proba(X_test)[:,1]

# fpr = False Positive Rate
# tpr = True Positive Rate
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_proba)

plt.plot([0,1], [0,1], "k--")
plt.plot(fpr, tpr, label="dtc")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Decision Tree ROC Curve")
plt.legend()
plt.savefig(os.path.join(metrics_dir_path, "roc_curve.png"))

# last lets get a file with the other metrics
with open(os.path.join(metrics_dir_path,"metrics.txt"), "w") as f:
    f.write(metrics.classification_report(y_test, y_pred))
    f.write(f"Area Under The Curve Score: {metrics.roc_auc_score(y_test, y_pred_proba)}")
