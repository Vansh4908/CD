import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
W7_PATH = os.path.join(BASE_DIR, '../W7')
W8_PATH = os.path.join(BASE_DIR, '../W8')

X_test = joblib.load(os.path.join(W7_PATH, 'X_test.pkl'))
y_test = joblib.load(os.path.join(W7_PATH, 'y_test.pkl'))

models = {
    "Logistic Regression": joblib.load(os.path.join(W7_PATH, 'models/logistic_regression.pkl')),
    "Random Forest": joblib.load(os.path.join(W7_PATH, 'models/random_forest.pkl')),
    "SVM": joblib.load(os.path.join(W7_PATH, 'models/svm.pkl')),
    "Gradient Boosting": joblib.load(os.path.join(W8_PATH, 'advanced_models/gradient_boosting.pkl')),
    "MLP": joblib.load(os.path.join(W8_PATH, 'advanced_models/mlp_classifier.pkl'))
}

names = []
accuracies = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    names.append(name)
    accuracies.append(acc)

plt.figure()
plt.bar(names, accuracies)
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Comparison")

plt.xticks(rotation=30)
plt.savefig("model_comparison.png")

print("Graph saved as model_comparison.png")

plt.show()