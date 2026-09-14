from __future__ import annotations

import pytest

from llm_training.cli import JsonlDataset, run_job
from llm_training.config_file import training_job_from_dict


@pytest.mark.integration
def test_real_dpo_trains_one_step(tmp_path) -> None:
    job = training_job_from_dict(
        {
            "model": {"name_or_path": "sshleifer/tiny-gpt2"},
            "data": {
                "train_file": "unused.jsonl",
                "format": "dpo",
                "max_sequence_length": 64,
            },
            "training": {
                "output_dir": str(tmp_path / "output"),
                "max_steps": 1,
                "per_device_batch_size": 2,
                "log_every_steps": 1,
            },
            "optimizer": {"learning_rate": 1e-5, "weight_decay": 0.0},
            "scheduler": {"name": "linear", "warmup_ratio": 0.0},
            "checkpoint": {"save_every_steps": None, "keep_last": 1},
            "dpo": {
                "beta": 0.1,
                "max_length": 64,
                "max_prompt_length": 32,
            },
        }
    )
    dataset = JsonlDataset(
        [
            {
                "prompt": "The opposite of hot is",
                "chosen": " cold.",
                "rejected": " loud.",
            },
            {
                "prompt": "Two plus two is",
                "chosen": " four.",
                "rejected": " seven.",
            },
        ]
    )

    result = run_job(job, train_dataset=dataset)

    assert result.global_step == 1
    assert (tmp_path / "output" / "config.json").is_file()
