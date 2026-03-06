import pandas as pd
import joblib
import os

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

os.makedirs("advanced_models", exist_ok=True)
os.makedirs("advanced_results", exist_ok=True)
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

#Gradient Boosting
gb = GradientBoostingClassifier()
gb_params = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1]
}

gb_grid = GridSearchCV(gb, gb_params, cv=3, scoring="f1_macro")
gb_grid.fit(X_train, y_train)

best_gb = gb_grid.best_estimator_
joblib.dump(best_gb, "advanced_models/gradient_boosting.pkl")

y_pred_gb = best_gb.predict(X_test)
report_gb = classification_report(y_test, y_pred_gb)

with open("advanced_results/gb_report.txt", "w") as f:
    f.write(report_gb)

print("Gradient Boosting Report:\n", report_gb)

#Neural Network
mlp = MLPClassifier(max_iter=500)
mlp_params = {
    "hidden_layer_sizes": [(100,), (100,50)],
    "alpha": [0.0001, 0.001]
}

mlp_grid = GridSearchCV(mlp, mlp_params, cv=3, scoring="f1_macro")
mlp_grid.fit(X_train, y_train)

best_mlp = mlp_grid.best_estimator_
joblib.dump(best_mlp, "advanced_models/mlp_classifier.pkl")

y_pred_mlp = best_mlp.predict(X_test)
report_mlp = classification_report(y_test, y_pred_mlp)

with open("advanced_results/mlp_report.txt", "w") as f:
    f.write(report_mlp)

print("MLP Report:\n", report_mlp)

print("Advanced models trained successfully!")