from __future__ import annotations

import pytest

from llm_training.config_file import load_training_job, training_job_from_dict
from llm_training.cli import main


def base_config():
    return {
        "model": {"name_or_path": "Qwen/Qwen3.5-4B", "dtype": "bfloat16"},
        "data": {
            "train_file": "train.jsonl",
            "format": "sft",
            "max_sequence_length": 128,
        },
        "training": {"output_dir": "outputs/run", "max_steps": 2},
        "lora": {"rank": 4, "alpha": 8, "target_modules": ["q_proj"]},
        "checkpoint": {"save_every_steps": None, "keep_last": 1},
    }


def test_yaml_builds_nested_training_config(tmp_path):
    path = tmp_path / "train.yaml"
    path.write_text(
        """model:\n  name_or_path: Qwen/Qwen3.5-4B\ndata:\n  train_file: train.jsonl\n  format: causal_lm\ntraining:\n  output_dir: outputs/run\n  max_steps: 2\n""",
        encoding="utf-8",
    )
    job = load_training_job(path)
    assert job.training.max_steps == 2
    assert job.data.format == "causal_lm"


def test_sequences_are_normalized_to_tuples():
    job = training_job_from_dict(base_config())
    assert job.training.lora.target_modules == ("q_proj",)


def test_unknown_nested_field_fails():
    payload = base_config()
    payload["optimizer"] = {"lr": 1e-5}
    with pytest.raises(ValueError, match="Unknown optimizer fields: lr"):
        training_job_from_dict(payload)


def test_adapter_output_requires_adapter_training():
    payload = base_config()
    del payload["lora"]
    payload["artifacts"] = {"save": "adapter"}
    with pytest.raises(ValueError, match="requires lora or trainable_tokens"):
        training_job_from_dict(payload)


def test_cli_dry_run_validates_data_without_loading_model(tmp_path, capsys):
    data = tmp_path / "train.jsonl"
    data.write_text('{"text":"hello world"}\n', encoding="utf-8")
    config = tmp_path / "train.yaml"
    config.write_text(
        f"""model:\n  name_or_path: fake/model\ndata:\n  train_file: {data}\n  format: causal_lm\ntraining:\n  output_dir: {tmp_path / 'output'}\n  max_steps: 1\n""",
        encoding="utf-8",
    )

    assert main(["--config", str(config), "--dry-run"]) == 0
    assert "TRAINING PLAN" in capsys.readouterr().out
    assert not (tmp_path / "output").exists()


def test_dpo_config_builds_and_validates():
    payload = base_config()
    payload["data"].update(
        format="dpo",
        chosen_field="winner",
        rejected_field="loser",
    )
    payload["dpo"] = {
        "beta": 0.2,
        "loss_type": "hinge",
        "max_length": 256,
        "max_prompt_length": 128,
    }
    job = training_job_from_dict(payload)
    assert job.dpo is not None
    assert job.dpo.beta == 0.2
    assert job.data.chosen_field == "winner"


def test_dpo_section_requires_dpo_data():
    payload = base_config()
    payload["dpo"] = {"beta": 0.2}
    with pytest.raises(ValueError, match="requires data.format='dpo'"):
        training_job_from_dict(payload)


def test_dpo_rejects_degenerate_preference():
    payload = base_config()
    payload["data"]["format"] = "dpo"
    job = training_job_from_dict(payload)
    from llm_training.cli import JsonlDataset, validate_dataset

    dataset = JsonlDataset(
        [{"prompt": "question", "chosen": "same", "rejected": "same"}]
    )
    with pytest.raises(ValueError, match="must differ"):
        validate_dataset(dataset, job.data)


def test_cli_dpo_dry_run_needs_no_optional_dependency(tmp_path, capsys):
    data = tmp_path / "preferences.jsonl"
    data.write_text(
        '{"prompt":"2+2?","chosen":"4","rejected":"5"}\n',
        encoding="utf-8",
    )
    config = tmp_path / "dpo.yaml"
    config.write_text(
        f"""model:
  name_or_path: fake/model
data:
  train_file: {data}
  format: dpo
training:
  output_dir: {tmp_path / 'output'}
  max_steps: 1
dpo:
  beta: 0.1
  max_length: 128
  max_prompt_length: 64
""",
        encoding="utf-8",
    )

    assert main(["--config", str(config), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "EFFECTIVE TRAINING PLAN" in output
    assert "max sequence length: 128" in output
    assert not (tmp_path / "output").exists()
