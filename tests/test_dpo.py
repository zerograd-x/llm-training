from __future__ import annotations

import sys
from types import ModuleType

import pytest

from llm_training.config_file import training_job_from_dict
from llm_training.dpo import (
    _build_trl_dpo_config,
    _load_trl_dpo_classes,
    dpo_deepspeed_config,
    dpo_training_arguments,
    prepare_preference_rows,
)


def dpo_job(**overrides):
    payload = {
        "model": {"name_or_path": "fake/model", "dtype": "bfloat16"},
        "data": {
            "train_file": "preferences.jsonl",
            "validation_file": "validation.jsonl",
            "format": "dpo",
            "max_sequence_length": 999,
        },
        "training": {
            "output_dir": "outputs/dpo",
            "max_steps": 3,
            "per_device_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "mixed_precision": "bf16",
        },
        "optimizer": {"learning_rate": 5e-6},
        "scheduler": {"name": "cosine", "warmup_steps": 2},
        "checkpoint": {"save_every_steps": None, "keep_last": 1},
        "dpo": {
            "beta": 0.2,
            "loss_type": "sigmoid",
            "max_length": 256,
            "max_prompt_length": 128,
        },
    }
    payload.update(overrides)
    return training_job_from_dict(payload)


def test_preference_rows_map_custom_columns():
    job = training_job_from_dict(
        {
            "model": {"name_or_path": "fake/model"},
            "data": {
                "train_file": "preferences.jsonl",
                "format": "dpo",
                "prompt_field": "question",
                "chosen_field": "winner",
                "rejected_field": "loser",
            },
            "training": {"output_dir": "outputs/dpo"},
        }
    )
    rows = prepare_preference_rows(
        [{"question": "2+2?", "winner": "4", "loser": "5"}],
        job.data,
    )
    assert rows == [{"prompt": "2+2?", "chosen": "4", "rejected": "5"}]


def test_system_prompt_uses_chat_template():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages[0] == {"role": "system", "content": "Be precise."}
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            return "<system>Be precise.</system><user>2+2?</user><assistant>"

    job = dpo_job()
    rows = prepare_preference_rows(
        [{"prompt": "2+2?", "chosen": "4", "rejected": "5"}],
        job.data,
        tokenizer=Tokenizer(),
        system_prompt="Be precise.",
    )
    assert rows[0]["prompt"].endswith("<assistant>")


def test_dpo_training_arguments_preserve_final_only_checkpoint():
    job = dpo_job()
    args = dpo_training_arguments(job)
    assert args["beta"] == 0.2
    assert args["max_length"] == 256
    assert args["max_prompt_length"] == 128
    assert args["learning_rate"] == 5e-6
    assert args["warmup_steps"] == 2
    assert args["save_strategy"] == "no"
    assert args["eval_strategy"] == "epoch"
    assert args["bf16"] is True


def test_dpo_maps_warmup_ratio_to_transformers_5_warmup_steps():
    job = dpo_job(scheduler={"name": "cosine", "warmup_ratio": 0.1})

    assert dpo_training_arguments(job)["warmup_steps"] == 0.1
    assert "warmup_ratio" not in dpo_training_arguments(job)


def test_dpo_deepspeed_translation_leaves_optimizer_to_hf():
    job = dpo_job(
        deepspeed={
            "zero_stage": 3,
            "offload_optimizer_device": "cpu",
            "offload_param_device": "cpu",
        }
    )
    config = dpo_deepspeed_config(job)
    assert config["zero_optimization"]["stage"] == 3
    assert config["gradient_accumulation_steps"] == "auto"
    assert "optimizer" not in config
    assert "scheduler" not in config


def test_dpo_lengths_are_validated():
    with pytest.raises(ValueError, match="max_prompt_length must be <="):
        dpo_job(dpo={"max_length": 64, "max_prompt_length": 65})


def test_trl_023_import_isolates_unused_callback_dependencies(monkeypatch):
    fake_trl = ModuleType("trl")

    class FakeDPOConfig:
        pass

    class FakeDPOTrainer:
        pass

    fake_trl.DPOConfig = FakeDPOConfig

    def module_getattr(name):
        if name != "DPOTrainer":
            raise AttributeError(name)
        mergekit_stub = sys.modules["trl.mergekit_utils"]
        judges_stub = sys.modules["trl.trainer.judges"]
        assert hasattr(mergekit_stub, "MergeConfig")
        assert hasattr(mergekit_stub, "merge_models")
        assert hasattr(judges_stub, "BasePairwiseJudge")
        return FakeDPOTrainer

    fake_trl.__getattr__ = module_getattr
    old_mergekit = ModuleType("trl.mergekit_utils")
    old_judges = ModuleType("trl.trainer.judges")
    monkeypatch.setitem(sys.modules, "trl", fake_trl)
    monkeypatch.setitem(sys.modules, "trl.mergekit_utils", old_mergekit)
    monkeypatch.setitem(sys.modules, "trl.trainer.judges", old_judges)
    monkeypatch.setattr("llm_training.dpo.package_version", lambda name: "0.23.0")

    config_cls, trainer_cls = _load_trl_dpo_classes()

    assert config_cls is FakeDPOConfig
    assert trainer_cls is FakeDPOTrainer
    assert sys.modules["trl.mergekit_utils"] is old_mergekit
    assert sys.modules["trl.trainer.judges"] is old_judges


def test_new_trl_dpo_config_compatibility():
    class CurrentDPOConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    config = _build_trl_dpo_config(
        CurrentDPOConfig,
        {
            "beta": 0.1,
            "max_length": 128,
            "max_prompt_length": 64,
            "reference_free": False,
        },
    )

    assert config.kwargs == {"beta": 0.1, "max_length": 128}

    with pytest.raises(ValueError, match="reference-free DPO"):
        _build_trl_dpo_config(
            CurrentDPOConfig,
            {"max_length": 128, "reference_free": True},
        )

