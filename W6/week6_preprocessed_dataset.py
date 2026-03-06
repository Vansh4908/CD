import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("compiler_error_dataset_1200_balanced.csv")

print("Dataset Loaded")
print(df.head())

# Text Preprocessing

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df["clean_error"] = df["error_message"].apply(clean_text)

print("Text Cleaning Done")

# TF-IDF
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=2000
)

X = vectorizer.fit_transform(df["clean_error"])

print("TF-IDF Feature Extraction Done")

# Encode Labels

label_map = {
    "Lexical":0,
    "Syntactic":1,
    "Semantic":2
}

y = df["category"].map(label_map)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Train Test Split Done")

pd.DataFrame(X_train.toarray()).to_csv("X_train.csv", index=False)
pd.DataFrame(X_test.toarray()).to_csv("X_test.csv", index=False)

pd.DataFrame(y_train).to_csv("y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("y_test.csv", index=False)

df.to_csv("cleaned_dataset.csv", index=False)

print("All Preprocessed Files Saved ")
print("Total samples:", len(df))
print("Unique error messages:", df["clean_error"].nunique())