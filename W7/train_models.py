import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

X_train = joblib.load(os.path.join(BASE_DIR, 'X_train.pkl'))
y_train = joblib.load(os.path.join(BASE_DIR, 'y_train.pkl'))

MODELS_PATH = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_PATH, exist_ok=True)

models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "svm": SVC()
}

for name, model in models.items():
    print(f"Training {name}...")
    
    model.fit(X_train, y_train)
    
    joblib.dump(model, os.path.join(MODELS_PATH, f"{name}.pkl"))

print("Week 7 Done: Models Trained & Saved")