# Machine Learning-Based Compiler Error Classification

## Overview

This project is a Machine Learning-based system that classifies compiler error messages into three categories:

* Lexical Errors
* Syntactic Errors
* Semantic Errors

The system helps developers quickly understand errors and improves debugging efficiency. It also highlights potential security risks associated with semantic errors.

---

## Features

* Automatic classification of compiler errors
* Multiple ML models implemented and compared
* Flask-based web interface for real-time prediction
* Modern and clean user interface
* Performance benchmarking with graphs

---

## Tech Stack

* Python
* Scikit-learn
* Flask
* HTML & CSS
* Matplotlib
* Joblib

---

## Project Structure

```
NCD/
│
├── data/
│   └── dataset.csv
│
├── W6/   # Preprocessing & Feature Extraction
├── W7/   # Model Training
├── W8/   # Advanced Models
├── W9/   # Evaluation
├── W10/  # Security Analysis
├── W11/  # Flask Integration
├── W12/  # Benchmarking
├── W13/  # Documentation
├── W14/  # Presentation
│
├── model/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── app.py
└── README.md
```

---

## Run the Application

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:5000/
```

---

## Example Inputs

| Input                | Output    |
| -------------------- | --------- |
| undefined variable x | Semantic  |
| missing semicolon    | Syntactic |
| invalid token @      | Lexical   |

---

## Model Performance

Multiple machine learning models were trained and evaluated:

* Logistic Regression (Best Model)
* Random Forest
* Support Vector Machine (SVM)
* Gradient Boosting
* Multi-Layer Perceptron (MLP)

The best model was selected based on accuracy and performance metrics.

---

## Security Relevance

Semantic errors can indicate potential vulnerabilities such as:

* Null pointer dereferencing
* Memory misuse
* Type mismatches

This system helps identify such issues early, contributing to secure software development.

---

## Future Improvements

* Use larger and more diverse datasets
* Implement advanced models like BERT
* Deploy application on cloud platforms
* Add real-time compiler integration

---

## Author

Vansh Rohit

---

## Acknowledgment

This project was developed as part of academic coursework and demonstrates the practical application of Machine Learning in compiler error analysis and software development.
