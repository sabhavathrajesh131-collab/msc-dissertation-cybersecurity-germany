"""
MSc Dissertation – Cybersecurity Incident Response & Recovery
German Financial Sector (2019–2024)

This script:
1. Extracts cybersecurity-related events from GDELT
2. Preprocesses text using NLP
3. Classifies events into Incident / Response / Recovery
4. Compares Rule-based, Logistic Regression, and SVM models
5. Exports results for dissertation reporting

Author: Rajesh
Programme: MSc Business Analytics
Year: 2025
"""

# =========================
# 1. Imports
# =========================
import requests
import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

# =========================
# 2. NLTK Setup
# =========================
nltk.download("stopwords")
nltk.download("wordnet")

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# =========================
# 3. GDELT Data Extraction
# =========================
def fetch_gdelt_data():
    """
    Fetch cybersecurity-related events affecting the
    German financial sector using GDELT Document API.
    """
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": "Germany financial cyber attack",
        "mode": "ArtList",
        "format": "JSON",
        "maxrecords": 250,
        "sourcelang": "English"
    }

    response = requests.get(url, params=params)
    data = response.json()
    articles = data.get("articles", [])

    df = pd.DataFrame(articles)
    df = df[["title", "seendate", "source"]]
    df.columns = ["text", "date", "source"]

    return df

# =========================
# 4. Text Preprocessing
# =========================
def preprocess_text(text):
    """
    Clean and normalise text for NLP analysis.
    """
    if pd.isna(text):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)

# =========================
# 5. Rule-Based Classifier
# =========================
def rule_based_classifier(text):
    """
    Baseline rule-based classification.
    """
    if any(word in text for word in ["restore", "resumed", "recovered"]):
        return "Recovery"
    elif any(word in text for word in ["investigation", "response", "mitigation"]):
        return "Response"
    else:
        return "Incident"

# =========================
# 6. Machine Learning Models
# =========================
def train_ml_models(X, y):
    """
    Train Logistic Regression and SVM models.
    """
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42
    )

    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    # Support Vector Machine
    svm_model = LinearSVC()
    svm_model.fit(X_train, y_train)
    svm_preds = svm_model.predict(X_test)

    results = {
        "Logistic Regression": {
            "accuracy": accuracy_score(y_test, lr_preds),
            "f1": f1_score(y_test, lr_preds, average="weighted"),
            "report": classification_report(y_test, lr_preds)
        },
        "SVM": {
            "accuracy": accuracy_score(y_test, svm_preds),
            "f1": f1_score(y_test, svm_preds, average="weighted"),
            "report": classification_report(y_test, svm_preds)
        }
    }

    return results

# =========================
# 7. Main Pipeline
# =========================
def main():
    print("Starting cybersecurity analysis pipeline...")

    # Step 1: Data Extraction
    df = fetch_gdelt_data()
    print(f"Records fetched: {len(df)}")

    # Step 2: Text Preprocessing
    df["processed_text"] = df["text"].apply(preprocess_text)

    # Step 3: Rule-Based Classification
    df["rule_label"] = df["processed_text"].apply(rule_based_classifier)

    # For ML demonstration, use rule-based labels as proxy ground truth
    X = df["processed_text"]
    y = df["rule_label"]

    # Step 4: Train and Evaluate ML Models
    model_results = train_ml_models(X, y)

    # Step 5: Output Results
    summary = []
    for model, metrics in model_results.items():
        summary.append({
            "model": model,
            "accuracy": round(metrics["accuracy"], 2),
            "f1_score": round(metrics["f1"], 2)
        })
        print(f"\n{model} Classification Report:\n")
        print(metrics["report"])

    results_df = pd.DataFrame(summary)
    results_df.to_csv("../results/tables/model_performance.csv", index=False)

    print("\nModel comparison saved to results/tables/model_performance.csv")
    print("Pipeline completed successfully.")

# =========================
# 8. Entry Point
# =========================
if __name__ == "__main__":
    main()
