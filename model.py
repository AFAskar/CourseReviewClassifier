import re

import nltk
from nltk.corpus import stopwords
import joblib
from pathlib import Path
from transformers import AutoTokenizer
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


def load_and_flatten(file: Path) -> pd.DataFrame:
    """
    Load a JSON file and flatten its structure into a pandas DataFrame.

    Args:
        file (Path): Path to the JSON file.
    Returns:
        pd.DataFrame: Flattened DataFrame.
    """
    data = json.load(open(file, "r"))
    df = pd.DataFrame(data).T
    df.reset_index(inplace=True)
    df.rename(columns={"index": "course_id"}, inplace=True)
    return df


def text_preproccessing(text: str) -> str:
    """
    Preprocess the input text by converting it to lowercase and removing punctuation.

    Args:
        text (str): The input text to preprocess.
    Returns:
        str: The preprocessed text.
    """
    text = text.casefold()
    punctuation_regex = r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]'
    text = re.sub(punctuation_regex, "", text)
    # remove emojis
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags (iOS)
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "]+",
        flags=re.UNICODE,
    )
    text = re.sub(emoji_pattern, "", text)
    # remove stopwords
    stop_words = set(stopwords.words("english"))
    text = " ".join([word for word in text.split() if word not in stop_words])
    text = text.strip()
    return text


if __name__ == "__main__":
    json_file = Path("./course_data.json")
    df = load_and_flatten(json_file)
