from __future__ import annotations

import json
from pathlib import Path

import torch

from examples.train_qwen35_category_sft import CategoryRows


class TinyTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=True):
        ids = [1] if add_special_tokens else []
        ids += [3 + (ord(ch) % 50) for ch in text]
        return {"input_ids": ids}


def test_category_sft_masks_prompt_and_supervises_answer(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps({
            "system": "Classify the product.",
            "user": "Product: wireless earbuds",
            "assistant": "ELECTRONICS",
        }) + "\n",
        encoding="utf-8",
    )

    rows = CategoryRows(path, TinyTokenizer(), max_length=128)
    item = rows[0]
    labels = item["labels"]
    attention = item["attention_mask"]

    assert len(item["input_ids"]) == 128
    assert (labels == -100).any()
    assert (labels != -100).any()
    assert torch.equal(labels[attention == 0], torch.full_like(labels[attention == 0], -100))
