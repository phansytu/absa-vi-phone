import json
import torch
from torch.utils.data import Dataset
from src.data.preprocess import normalize


class ABSADataset(Dataset):
    def __init__(self, path, norm_kwargs=None, segmenter=None):
        with open(path, encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f]
        self.norm_kwargs = norm_kwargs or {}
        self.segmenter = segmenter        

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        text = normalize(r["text"], **self.norm_kwargs)
        if self.segmenter:
            text = self.segmenter(text)
        return {"id": r["id"], "text": text,
                "presence": r["presence"], "sentiment": r["sentiment"]}


def make_collate(tokenizer, max_length=128):
    def collate(batch):
        enc = tokenizer([b["text"] for b in batch], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        enc["presence"] = torch.tensor([b["presence"] for b in batch], dtype=torch.float)
        enc["sentiment"] = torch.tensor([b["sentiment"] for b in batch], dtype=torch.long)
        enc["ids"] = [b["id"] for b in batch]
        return enc
    return collate