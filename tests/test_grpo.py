from __future__ import annotations

import sys
from types import ModuleType

import pytest

from llm_training.config import GRPORewardConfig
from llm_training.config_file import training_job_from_dict
from llm_training.grpo import (
    grpo_training_arguments,
    load_reward,
    prepare_prompt_rows,
)


def grpo_job(**overrides):
    payload = {
        "model": {"name_or_path": "fake/model", "dtype": "bfloat16"},
        "data": {
            "train_file": "prompts.jsonl",
            "validation_file": "validation.jsonl",
            "format": "grpo",
        },
        "training": {
            "output_dir": "outputs/grpo",
            "max_steps": 3,
            "per_device_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "mixed_precision": "bf16",
        },
        "optimizer": {"learning_rate": 1e-6},
        "scheduler": {"name": "cosine", "warmup_steps": 2},
        "checkpoint": {"save_every_steps": None, "keep_last": 1},
        "grpo": {
            "num_generations": 4,
            "max_completion_length": 128,
            "temperature": 0.8,
            "loss_type": "dapo",
            "rewards": [
                {
                    "module_path": "example_rewards.correctness",
                    "weight": 1.0,
                }
            ],
        },
    }
    payload.update(overrides)
    return training_job_from_dict(payload)


def test_grpo_config_and_training_arguments():
    job = grpo_job()
    args = grpo_training_arguments(job)

    assert job.data.format == "grpo"
    assert job.grpo.num_generations == 4
    assert args["num_generations"] == 4
    assert args["max_completion_length"] == 128
    assert args["learning_rate"] == 1e-6
    assert args["warmup_steps"] == 2
    assert args["save_strategy"] == "no"
    assert args["eval_strategy"] == "epoch"
    assert args["reward_weights"] == [1.0]
    assert args["bf16"] is True
    assert "use_vllm" not in args


def test_grpo_preserves_reward_metadata_and_maps_prompt_column():
    job = training_job_from_dict(
        {
            "model": {"name_or_path": "fake/model"},
            "data": {
                "train_file": "prompts.jsonl",
                "format": "grpo",
                "prompt_field": "question",
            },
            "training": {
                "output_dir": "outputs/grpo",
                "per_device_batch_size": 2,
            },
            "grpo": {
                "num_generations": 2,
                "rewards": [
                    {"module_path": "example_rewards.correctness"}
                ],
            },
        }
    )

    rows = prepare_prompt_rows(
        [{"question": "2+2?", "answer": "4", "row_id": 7}],
        job.data,
    )

    assert rows == [
        {
            "question": "2+2?",
            "prompt": "2+2?",
            "answer": "4",
            "row_id": 7,
        }
    ]


def test_grpo_requires_generation_batch_divisibility():
    with pytest.raises(ValueError, match="must be divisible"):
        grpo_job(
            training={
                "output_dir": "outputs/grpo",
                "per_device_batch_size": 1,
                "gradient_accumulation_steps": 2,
            },
            grpo={
                "num_generations": 3,
                "rewards": [
                    {"module_path": "example_rewards.correctness"}
                ],
            },
        )


def test_grpo_rejects_custom_loss_and_trainable_tokens():
    with pytest.raises(ValueError, match="owns its loss"):
        grpo_job(loss={"module_path": "example.CustomLoss"})
    with pytest.raises(ValueError, match="does not support trainable_tokens"):
        grpo_job(trainable_tokens={"n_tokens_added": 2})


def test_reward_loader_binds_kwargs_and_preserves_name(monkeypatch):
    module = ModuleType("example_rewards")

    def correctness(*, completions, answer, bonus=0.0, **kwargs):
        return [
            float(completion == expected) + bonus
            for completion, expected in zip(completions, answer)
        ]

    module.correctness = correctness
    monkeypatch.setitem(sys.modules, "example_rewards", module)

    reward = load_reward(
        GRPORewardConfig(
            module_path="example_rewards.correctness",
            kwargs={"bonus": 0.25},
        )
    )

    assert reward.__name__ == "correctness"
    assert reward(completions=["4", "5"], answer=["4", "4"]) == [1.25, 0.25]


def test_reward_abstention_is_explicit(monkeypatch):
    module = ModuleType("example_rewards")

    def sometimes_abstains(**kwargs):
        return [1.0, None]

    module.sometimes_abstains = sometimes_abstains
    monkeypatch.setitem(sys.modules, "example_rewards", module)

    strict = load_reward(
        GRPORewardConfig(
            module_path="example_rewards.sometimes_abstains",
            abstain="error",
        )
    )
    with pytest.raises(ValueError, match="abstained"):
        strict()

    zero = load_reward(
        GRPORewardConfig(
            module_path="example_rewards.sometimes_abstains",
            abstain="zero",
        )
    )
    assert zero() == [1.0, 0.0]


def test_grpo_section_is_exclusive_to_grpo_data():
    with pytest.raises(ValueError, match="requires data.format='grpo'"):
        grpo_job(
            data={
                "train_file": "train.jsonl",
                "format": "sft",
            }
        )
