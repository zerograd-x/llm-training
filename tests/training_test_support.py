from types import SimpleNamespace

import torch
from torch import nn


class ToyLM(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(8, 16)
        self.seen = []

    def forward(self, input_ids):
        self.seen.extend(input_ids.detach().cpu().tolist())
        return SimpleNamespace(
            logits=self.head(self.dropout(self.embedding(input_ids)))
        )


def rows():
    return [
        {
            "input_ids": torch.tensor([1, index + 3, 2]),
            "labels": torch.tensor(
                [1, index + 3 if index % 2 else -100, 2]
            ),
        }
        for index in range(5)
    ]


def collate(batch):
    return {
        key: torch.stack([row[key] for row in batch])
        for key in batch[0]
    }
