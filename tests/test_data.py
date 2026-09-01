import pytest

from llm_training import adapt_column_batch_collator, rows_to_column_batch


def test_rows_to_column_batch() -> None:
    rows = [
        {"prompt": "a", "completion": "x"},
        {"prompt": "b", "completion": "y"},
    ]
    assert rows_to_column_batch(rows) == {
        "prompt": ["a", "b"],
        "completion": ["x", "y"],
    }


def test_adapter_calls_column_collator() -> None:
    seen = {}

    def column_collator(batch):
        seen.update(batch)
        return {"ok": True}

    collate = adapt_column_batch_collator(column_collator)
    result = collate([{"prompt": "a"}, {"prompt": "b"}])

    assert result == {"ok": True}
    assert seen == {"prompt": ["a", "b"]}


def test_rows_must_have_same_keys() -> None:
    with pytest.raises(ValueError, match="expected"):
        rows_to_column_batch([{"a": 1}, {"b": 2}])
