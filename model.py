import re

import nltk
from nltk.corpus import stopwords
import joblib
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import json
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

cache_dir = Path("./.cache")

memory = joblib.Memory(location=cache_dir, verbose=0)


def load_data(file: Path) -> pd.DataFrame:
    """
    Load a JSON file and flatten its structure into a pandas DataFrame.

    Args:
        file (Path): Path to the JSON file.
    Returns:
        pd.DataFrame: Flattened DataFrame.
    """
    nltk.download("stopwords")
    data = json.load(open(file, "r"))
    df = pd.DataFrame(data).T
    df.reset_index(inplace=True)
    df.rename(columns={"index": "course_id"}, inplace=True)
    reviews_df = df[["course_id", "reviews"]].explode("reviews").reset_index(drop=True)
    reviews_df = reviews_df[reviews_df["reviews"].notnull()].copy()
    reviews_df["text"] = reviews_df["reviews"].apply(
        lambda x: x.get("review_text") if isinstance(x, dict) else None
    )
    reviews_df["label"] = reviews_df["reviews"].apply(
        lambda x: x.get("course_rating") if isinstance(x, dict) else None
    )
    reviews_df = reviews_df.drop(columns=["reviews"])
    reviews_df = reviews_df.drop(columns=["course_id"])
    return reviews_df


def text_preproccessing(text: str) -> str:
    """
    Preprocess the input text by converting it to lowercase and removing punctuation.

    Args:
        text (str): The input text to preprocess.
    Returns:
        str: The preprocessed text.
    """
    text = text.casefold()
    alphanum = r"[^a-zA-Z0-9\s]+$"
    text = re.sub(alphanum, "", text)

    # remove stopwords
    stop_words = set(stopwords.words("english"))
    text = " ".join([word for word in text.split() if word not in stop_words])
    text = text.strip()
    return text


def preprocess_reviews(reviews_df: pd.DataFrame) -> pd.DataFrame:
    reviews_df["label"] = reviews_df["label"].apply(
        lambda x: 1 if x == "liked course" else 0
    )
    reviews_df["text"] = reviews_df["text"].apply(text_preproccessing)
    return reviews_df


def embed_reviews(
    reviews: pd.Series, model_name: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """
    Embed the input reviews using a pre-trained SentenceTransformer model.

    Args:
        reviews (pd.Series): Series of review texts to embed.
        model_name (str): Name of the pre-trained SentenceTransformer model.
    Returns:
        np.ndarray: Array of embedded review vectors.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(reviews.tolist(), show_progress_bar=True)
    return embeddings


@memory.cache
def run(reviews_df: pd.DataFrame):
    X = embed_reviews(reviews_df["text"])
    y = reviews_df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Support Vector Machine": SVC(),
    }
    scores = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary"
        )
        cm = confusion_matrix(y_test, y_pred)
        scores[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "confusion_matrix": cm,
        }
    return scores


def compare_scores(scores: dict) -> None:
    # Function to compare scores of different classifiers and print the best one in each metric
    metrics = ["accuracy", "precision", "recall", "f1-score"]
    best_scores = {metric: {"classifier": None, "score": 0} for metric in metrics}
    for name, score in scores.items():
        for metric in metrics:
            if score[metric] > best_scores[metric]["score"]:
                best_scores[metric]["score"] = score[metric]
                best_scores[metric]["classifier"] = name
    for metric in metrics:
        print(
            f"Best {metric}: {best_scores[metric]['score']:.4f} by {best_scores[metric]['classifier']}"
        )


if __name__ == "__main__":
    data_file = Path("./course_data.json")
    reviews_df = load_data(data_file)
    reviews_df = preprocess_reviews(reviews_df)
    scores = run(reviews_df)
    for name, score in scores.items():
        print(f"Classifier: {name}")
        print(f"Accuracy: {score['accuracy']:.4f}")
        print(f"Precision: {score['precision']:.4f}")
        print(f"Recall: {score['recall']:.4f}")
        print(f"F1-Score: {score['f1-score']:.4f}")
        print(f"Confusion Matrix:\n{score['confusion_matrix']}\n")
    print("Comparison of Classifier Scores:")
    compare_scores(scores)
