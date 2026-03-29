from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load model + vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, 'model/model.pkl'))
vectorizer = joblib.load(os.path.join(BASE_DIR, 'model/vectorizer.pkl'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    error_msg = request.form['error']

    # Transform using SAME vectorizer
    data = vectorizer.transform([error_msg])

    prediction = model.predict(data)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)