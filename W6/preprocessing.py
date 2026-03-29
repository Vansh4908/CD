import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('dataset.csv')
X = df['error_message']
y = df['category']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
vectorizer = TfidfVectorizer(max_features=300)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

W7_PATH = os.path.join(BASE_DIR, '../W7')
MODEL_PATH = os.path.join(BASE_DIR, '../model')

os.makedirs(W7_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

joblib.dump(X_train_vec, os.path.join(W7_PATH, 'X_train.pkl'))
joblib.dump(X_test_vec, os.path.join(W7_PATH, 'X_test.pkl'))
joblib.dump(y_train, os.path.join(W7_PATH, 'y_train.pkl'))
joblib.dump(y_test, os.path.join(W7_PATH, 'y_test.pkl'))

joblib.dump(vectorizer, os.path.join(MODEL_PATH, 'vectorizer.pkl'))