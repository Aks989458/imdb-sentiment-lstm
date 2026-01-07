import os
import tarfile
import urllib.request
import re
from collections import Counter
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATA_DIR = "data"
ARCHIVE_PATH = os.path.join(DATA_DIR, "aclImdb_v1.tar.gz")
EXTRACT_PATH = os.path.join(DATA_DIR, "aclImdb")

def download_and_extract_imdb():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(ARCHIVE_PATH):
        urllib.request.urlretrieve(IMDB_URL, ARCHIVE_PATH)
    if not os.path.exists(EXTRACT_PATH):
        with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
            tar.extractall(DATA_DIR)
    return EXTRACT_PATH

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def load_imdb_data(path):
    texts, labels = [], []
    for name in ["pos", "neg"]:
        label = 1 if name == "pos" else 0
        folder = os.path.join(path, name)
        for f in os.listdir(folder):
            with open(os.path.join(folder, f), encoding="utf-8") as file:
                texts.append(clean_text(file.read()))
                labels.append(label)
    return texts, labels

def build_vocab(texts, max_vocab=20000):
    counter = Counter()
    for t in texts:
        counter.update(t.split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, _ in counter.most_common(max_vocab):
        vocab[w] = len(vocab)
    return vocab

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=200):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = [self.vocab.get(w, 1) for w in self.texts[idx].split()][:self.max_len]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

def collate_fn(batch):
    x, y = zip(*batch)
    x = pad_sequence(x, batch_first=True, padding_value=0)
    return x, torch.stack(y)

def prepare_imdb():
    base = download_and_extract_imdb()
    texts, labels = load_imdb_data(os.path.join(base, "train"))
    train_x, val_x, train_y, val_y = train_test_split(
        texts, labels, test_size=0.1, stratify=labels, random_state=42
    )
    test_x, test_y = load_imdb_data(os.path.join(base, "test"))
    vocab = build_vocab(train_x)
    return train_x, train_y, val_x, val_y, test_x, test_y, vocab
