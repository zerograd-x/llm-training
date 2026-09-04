import json
from pathlib import Path

import pytest

pytest.importorskip("llm_training_data")
pytest.importorskip("peft")

from llm_training import (
    ArtifactManifest,
    EvaluationConfig,
    LoRAConfig,
    TrainableTokensConfig,
    OptimizerConfig,
    TrainingConfig,
    run_training_workflow,
)
from test_vocabulary_manifest import make_base


@pytest.mark.parametrize("mode", ["full", "lora", "tokens"])
def test_real_tokenizer_training_export_fresh_generation(tmp_path, mode):
    make_base(tmp_path / "base")
    train = [
        {"prompt": "question one", "completion": "answer red"},
        {"prompt": "question two", "completion": "answer blue"},
        {"prompt": "question three", "completion": "answer <new_a>"},
    ]
    validation = [{"prompt": "question four", "completion": "answer green"}]
    test = [{"prompt": "question green", "completion": "answer <new_b>"}]
    cfg = TrainingConfig(
        output_dir=str(tmp_path / "run"),
        gradient_accumulation_steps=2,
        optimizer=OptimizerConfig(learning_rate=0.01),
        evaluation=EvaluationConfig(every_steps=1),
        lora=LoRAConfig(target_modules=("c_attn",)) if mode == "lora" else None,
        trainable_tokens=(
            TrainableTokensConfig(n_tokens_added=2) if mode == "tokens" else None
        ),
    )
    result = run_training_workflow(
        base_path=tmp_path / "base",
        train_data=train,
        validation_data=validation,
        test_data=test,
        config=cfg,
        new_tokens=["<new_a>", "<new_b>"],
        inference=True,
        max_new_tokens=2,
        max_sequence_length=32,
    )
    manifest = ArtifactManifest.load(result.model_path).validate(
        result.model_path, base_path=tmp_path / "run/expanded-base"
    )
    assert manifest.kind == "model" and manifest.new_tokens == {
        "<new_a>": 12,
        "<new_b>": 13,
    }
    assert (
        result.adapter_path is not None
        if mode != "full"
        else result.adapter_path is None
    )
    predictions = [
        json.loads(x) for x in Path(result.predictions_path).read_text().splitlines()
    ]
    assert len(predictions) == 1 and predictions[0]["prompt"] == test[0]["prompt"]
    stages = Path(cfg.output_dir) / "workflow"
    records = [json.loads(p.read_text()) for p in stages.glob("*-result.json")]
    assert all(row["ok"] for row in records) and len(
        {row["pid"] for row in records}
    ) == len(records)
    assert result.evaluation["validation"][-1]["eval_examples"] == 1


def test_workflow_invalid_splits_fail_before_model_loading(tmp_path):
    cfg = TrainingConfig(output_dir=str(tmp_path / "run"))
    row = {"prompt": "question", "completion": "answer"}
    with pytest.raises(ValueError, match="overlap"):
        run_training_workflow(
            base_path=tmp_path / "missing",
            train_data=[row],
            validation_data=[row],
            config=cfg,
        )
    with pytest.raises(ValueError, match="requires validation or test"):
        run_training_workflow(
            base_path=tmp_path / "missing", train_data=[row], config=cfg, inference=True
        )
    assert not Path(cfg.output_dir).exists()
