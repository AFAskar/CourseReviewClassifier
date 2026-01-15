import re

import nltk
from nltk.corpus import stopwords
import joblib
from pathlib import Path
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json

cache_dir = Path("./.cache")

memory = joblib.Memory(location=cache_dir, verbose=0)


class JsonDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# load json and return torch dataset
def load_data(file: Path) -> JsonDataset:
    """
    Load Data From json file
    Args:
        file (Path): Path to the json file.
    Returns:
        Dataset: Loaded dataset.
    """

    with open(file, "r") as f:
        data = json.load(f)
    # print data variable datatype
    print(type(data))
    return JsonDataset(data)


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
    dataset = load_data(json_file)
    print(f"Loaded {len(dataset)} samples from {json_file}")
    print("First sample:", dataset[0])
