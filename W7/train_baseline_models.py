import pandas as pd
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

lr = LogisticRegression(max_iter=1000)
lr_params = {"C": [0.1, 1, 10]}

lr_grid = GridSearchCV(lr, lr_params, cv=3, scoring="f1_macro")
lr_grid.fit(X_train, y_train)

best_lr = lr_grid.best_estimator_
joblib.dump(best_lr, "models/logistic_regression.pkl")

svm = SVC()
svm_params = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}

svm_grid = GridSearchCV(svm, svm_params, cv=3, scoring="f1_macro")
svm_grid.fit(X_train, y_train)

best_svm = svm_grid.best_estimator_
joblib.dump(best_svm, "models/svm.pkl")

rf = RandomForestClassifier()
rf_params = {"n_estimators": [100, 200], "max_depth": [None, 10, 20]}

rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring="f1_macro")
rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_
joblib.dump(best_rf, "models/random_forest.pkl")

print("All models trained and saved!")

y_pred = best_rf.predict(X_test)

report = classification_report(y_test, y_pred)

with open("results/classification_report.txt", "w") as f:
    f.write(report)

print("\nClassification Report:\n")
print(report)