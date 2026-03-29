import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
W7_PATH = os.path.join(BASE_DIR, '../W7')

X_train = joblib.load(os.path.join(W7_PATH, 'X_train.pkl'))
y_train = joblib.load(os.path.join(W7_PATH, 'y_train.pkl'))

ADV_MODELS_PATH = os.path.join(BASE_DIR, 'advanced_models')
os.makedirs(ADV_MODELS_PATH, exist_ok=True)

models = {
    "gradient_boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05),
    "mlp_classifier": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
}

for name, model in models.items():
    print(f"Training {name}...")
    
    model.fit(X_train, y_train)
    
    joblib.dump(model, os.path.join(ADV_MODELS_PATH, f"{name}.pkl"))

print("Week 8 Done: Advanced Models Trained")