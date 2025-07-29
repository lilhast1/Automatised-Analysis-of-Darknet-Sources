# data_loader.py
import os, re
from bs4 import BeautifulSoup
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer

# Load a tokenizer (can be BERT-based or simple)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def clean_html(file_path):
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "noscript"]):
        element.extract()
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def tokenize_text(text, max_len=512):
    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt"
    )
    return tokens["input_ids"].squeeze(0)

class HTMLDataset(Dataset):
    def __init__(self, file_paths, labels=None, max_len=512):
        self.file_paths = file_paths
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        text = clean_html(path)
        input_ids = tokenize_text(text, self.max_len)

        if self.labels is not None:
            label = self.labels[idx]
            return input_ids, torch.tensor(label, dtype=torch.long)
        else:
            return input_ids, torch.tensor(-1, dtype=torch.long)
