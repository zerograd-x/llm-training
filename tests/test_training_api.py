import json
from pathlib import Path

import pytest

pytest.importorskip("llm_training_data")
pytest.importorskip("peft")

from llm_training import (
    ArtifactManifest,
    EvaluationConfig,
    LoRAConfig,
    OptimizerConfig,
    TrainableTokensConfig,
    TrainingConfig,
    run_training,
)
from test_vocabulary_manifest import make_base


@pytest.mark.parametrize("mode", ["full", "lora", "tokens"])
def test_training_lifecycle(tmp_path, mode):
    make_base(tmp_path / "base")
    train = [
        {"prompt": "question one", "completion": "answer red"},
        {"prompt": "question two", "completion": "answer blue"},
        {"prompt": "question three", "completion": "answer <new_a>"},
    ]
    validation = [{"prompt": "question four", "completion": "answer green"}]
    test = [{"prompt": "question green", "completion": "answer <new_b>"}]
    config = TrainingConfig(
        output_dir=str(tmp_path / "run"),
        gradient_accumulation_steps=2,
        optimizer=OptimizerConfig(learning_rate=0.01),
        evaluation=EvaluationConfig(every_steps=1),
        lora=LoRAConfig(target_modules=("c_attn",)) if mode == "lora" else None,
        trainable_tokens=(
            TrainableTokensConfig(n_tokens_added=2) if mode == "tokens" else None
        ),
    )

    result = run_training(
        base_path=tmp_path / "base",
        train_data=train,
        validation_data=validation,
        test_data=test,
        config=config,
        new_tokens=["<new_a>", "<new_b>"],
        inference=True,
        max_new_tokens=2,
        max_sequence_length=32,
        run_id="tiny-lifecycle",
        artifact_root_uri="s3://bucket/runs",
    )

    manifest = ArtifactManifest.load(result.model_path).validate(
        result.model_path, base_path=tmp_path / "run/expanded-base"
    )
    assert manifest.kind == "model"
    assert manifest.new_tokens == {"<new_a>": 12, "<new_b>": 13}
    assert (result.adapter_path is not None) == (mode != "full")
    assert result.run_id == "tiny-lifecycle"
    assert result.artifact_uri == "s3://bucket/runs/tiny-lifecycle/"

    predictions = [
        json.loads(line)
        for line in Path(result.predictions_path).read_text().splitlines()
    ]
    assert len(predictions) == 1
    assert predictions[0]["prompt"] == test[0]["prompt"]

    run_dir = Path(config.output_dir)
    run_metadata = json.loads((run_dir / "run.json").read_text())
    plan = json.loads((run_dir / "training-plan.json").read_text())
    assert run_metadata["run_id"] == "tiny-lifecycle"
    assert run_metadata["artifact_uri"] == "s3://bucket/runs/tiny-lifecycle/"
    assert plan["run_id"] == "tiny-lifecycle"
    assert plan["optimizer_steps"] == 2
    assert plan["optimizer_owner"] == "torch"

    stage_dir = run_dir / "stages"
    records = [json.loads(path.read_text()) for path in stage_dir.glob("*-result.json")]
    assert records
    assert all(record["ok"] for record in records)
    assert len({record["pid"] for record in records}) == len(records)
    assert result.evaluation["validation"][-1]["eval_examples"] == 1
    assert (run_dir / "training-result.json").is_file()


def test_training_rejects_invalid_splits_before_model_loading(tmp_path):
    config = TrainingConfig(output_dir=str(tmp_path / "run"))
    row = {"prompt": "question", "completion": "answer"}

    with pytest.raises(ValueError, match="overlap"):
        run_training(
            base_path=tmp_path / "missing",
            train_data=[row],
            validation_data=[row],
            config=config,
        )

    with pytest.raises(ValueError, match="requires validation or test"):
        run_training(
            base_path=tmp_path / "missing",
            train_data=[row],
            config=config,
            inference=True,
        )

    assert not Path(config.output_dir).exists()
