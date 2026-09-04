"""Single-GPU BF16/ZeRO-3 workflow with a real tokenizer and fresh generation."""

import json
import os
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_vocabulary_manifest import make_base
from llm_training import (
    ArtifactManifest,
    DeepSpeedConfig,
    EvaluationConfig,
    LoRAConfig,
    OptimizerConfig,
    TrainingConfig,
    TrainableTokensConfig,
    run_training_workflow,
)


def main(root, pretrained=False):
    root.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if pretrained:
        name = "HuggingFaceTB/SmolLM2-135M-Instruct"
        AutoModelForCausalLM.from_pretrained(name).save_pretrained(root / "base")
        AutoTokenizer.from_pretrained(name).save_pretrained(root / "base")
    else:
        make_base(root / "base")
    results = []
    for mode in (("lora",) if pretrained else ("full", "lora", "tokens")):
        cfg = TrainingConfig(
            output_dir=str(root / mode),
            mixed_precision="bf16",
            num_epochs=2,
            gradient_accumulation_steps=2,
            optimizer=OptimizerConfig(learning_rate=0.001),
            evaluation=EvaluationConfig(every_steps=1),
            deepspeed=DeepSpeedConfig(zero_stage=3),
            lora=(
                LoRAConfig(
                    target_modules=("q_proj", "v_proj") if pretrained else ("c_attn",)
                )
                if mode == "lora"
                else None
            ),
            trainable_tokens=(
                TrainableTokensConfig(n_tokens_added=2) if mode == "tokens" else None
            ),
        )
        trained = run_training_workflow(
            base_path=root / "base",
            config=cfg,
            train_data=[
                {"prompt": "question one", "completion": "answer red"},
                {"prompt": "question two", "completion": "answer blue"},
                {"prompt": "question three", "completion": "answer <new_a>"},
            ],
            validation_data=[{"prompt": "question four", "completion": "answer green"}],
            test_data=[{"prompt": "question green", "completion": "answer <new_b>"}],
            new_tokens=["<new_a>", "<new_b>"],
            inference=True,
            inference_device="cuda",
            max_sequence_length=64,
            max_new_tokens=4,
            timeout=600,
        )
        stages = root / mode / "workflow"
        records = [json.loads(p.read_text()) for p in stages.glob("*-result.json")]
        train = json.loads((stages / "train-result.json").read_text())
        assert train["device"].startswith("cuda") and train["mixed_precision"] == "bf16"
        assert len({record["pid"] for record in records}) == len(records)
        for path in [trained.checkpoint_path, trained.model_path] + (
            [trained.adapter_path] if trained.adapter_path else []
        ):
            ArtifactManifest.load(path).validate(
                path, base_path=root / mode / "expanded-base"
            )
        result = dict(
            mode=mode,
            ok=True,
            pretrained=pretrained,
            evaluation=trained.evaluation,
            stage_pids=[record["pid"] for record in records],
        )
        results.append(result)
        (root / "matrix.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), "--pretrained" in sys.argv)
