from __future__ import annotations

import pytest

from llm_training.cli import JsonlDataset, run_job
from llm_training.config_file import training_job_from_dict


@pytest.mark.integration
def test_real_grpo_trains_one_step(tmp_path) -> None:
    job = training_job_from_dict(
        {
            "model": {"name_or_path": "sshleifer/tiny-gpt2"},
            "data": {
                "train_file": "unused.jsonl",
                "format": "grpo",
            },
            "training": {
                "output_dir": str(tmp_path / "output"),
                "max_steps": 1,
                "per_device_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "log_every_steps": 1,
            },
            "optimizer": {"learning_rate": 1e-6, "weight_decay": 0.0},
            "scheduler": {"name": "linear", "warmup_ratio": 0.0},
            "checkpoint": {"save_every_steps": None, "keep_last": 1},
            "grpo": {
                "num_generations": 2,
                "max_completion_length": 4,
                "temperature": 1.0,
                "rewards": [
                    {
                        "module_path": "llm_training.rewards.exact_match_reward",
                        "weight": 1.0,
                    }
                ],
            },
        }
    )
    dataset = JsonlDataset(
        [
            {
                "prompt": "Two plus two is",
                "answer": " four.",
            },
            {
                "prompt": "The opposite of hot is",
                "answer": " cold.",
            },
        ]
    )

    result = run_job(job, train_dataset=dataset)

    assert result.global_step == 1
    assert (tmp_path / "output" / "config.json").is_file()
    assert (tmp_path / "output" / "grpo_config.json").is_file()
