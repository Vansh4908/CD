import joblib
import os
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
W7_PATH = os.path.join(BASE_DIR, '../W7')
W8_PATH = os.path.join(BASE_DIR, '../W8')
MODEL_PATH = os.path.join(BASE_DIR, '../model')

X_test = joblib.load(os.path.join(W7_PATH, 'X_test.pkl'))
y_test = joblib.load(os.path.join(W7_PATH, 'y_test.pkl'))

models = {
    "Logistic Regression": joblib.load(os.path.join(W7_PATH, 'models/logistic_regression.pkl')),
    "Random Forest": joblib.load(os.path.join(W7_PATH, 'models/random_forest.pkl')),
    "SVM": joblib.load(os.path.join(W7_PATH, 'models/svm.pkl')),
    "Gradient Boosting": joblib.load(os.path.join(W8_PATH, 'advanced_models/gradient_boosting.pkl')),
    "MLP": joblib.load(os.path.join(W8_PATH, 'advanced_models/mlp_classifier.pkl'))
}

best_model = None
best_accuracy = 0
best_name = ""

print("\nMODEL COMPARISON:\n")

for name, model in models.items():
    print(f"--- {name} ---")
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")
    print(classification_report(y_test, y_pred))
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_name = name

os.makedirs(MODEL_PATH, exist_ok=True)

joblib.dump(best_model, os.path.join(MODEL_PATH, 'model.pkl'))

print(f"\nBest Model: {best_name}")
print("Saved as model/model.pkl")