"""Plain-Python workflow; each model lifecycle stage runs in a fresh process."""

from dataclasses import asdict, dataclass
from importlib.util import find_spec
import json
import os
from pathlib import Path
import subprocess
import signal
import sys

from . import config as configuration
from .distributed import require_single_process
from .manifest import ArtifactManifest
from .vocabulary import expand_vocabulary


@dataclass(frozen=True)
class WorkflowResult:
    model_path: str
    checkpoint_path: str
    adapter_path: str | None
    predictions_path: str | None
    evaluation: dict


def training_config_from_dict(value):
    value = dict(value)
    classes = dict(
        freeze=configuration.FreezeConfig,
        lora=configuration.LoRAConfig,
        trainable_tokens=configuration.TrainableTokensConfig,
        optimizer=configuration.OptimizerConfig,
        scheduler=configuration.SchedulerConfig,
        loss=configuration.LossConfig,
        gradient_checkpointing=configuration.GradientCheckpointingConfig,
        checkpoint=configuration.CheckpointConfig,
        evaluation=configuration.EvaluationConfig,
    )
    for name, cls in classes.items():
        if value.get(name) is not None:
            value[name] = cls(**value[name])
    if value.get("deepspeed") is not None:
        ds = dict(value["deepspeed"])
        for key, cls in [
            ("optimizer", configuration.DeepSpeedOptimizerConfig),
            ("scheduler", configuration.DeepSpeedSchedulerConfig),
        ]:
            if ds.get(key) is not None:
                ds[key] = cls(**ds[key])
        value["deepspeed"] = configuration.DeepSpeedConfig(**ds)
    config = configuration.TrainingConfig(**value)
    config.validate()
    return config


def read_examples(value):
    if isinstance(value, (str, Path)):
        value = [
            json.loads(line)
            for line in Path(value).read_text().splitlines()
            if line.strip()
        ]
    result = list(value)
    if not result:
        raise ValueError("Dataset is empty")
    for row in result:
        if not isinstance(row, dict) or not all(
            isinstance(row.get(k), str) for k in ("prompt", "completion")
        ):
            raise ValueError(
                "Each example requires string prompt and completion fields"
            )
    return result


def _stage(name, request, directory, timeout):
    path = directory / f"{name}-request.json"
    path.write_text(json.dumps(request) + "\n")
    result_path = directory / f"{name}-result.json"
    env = dict(os.environ)
    for key in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "ACCELERATE_USE_DEEPSPEED",
    ):
        env.pop(key, None)
    with (directory / f"{name}.log").open("w") as log:
        result = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "llm_training.workflow_worker",
                name,
                str(path),
                str(result_path),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            result.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(result.pid, signal.SIGKILL)
            result.wait()
            raise TimeoutError(
                f"Workflow {name} timed out; see {directory/name}.log"
            ) from None
    if result.returncode != 0 or not result_path.is_file():
        raise RuntimeError(f"Workflow {name} failed; see {directory/name}.log")
    payload = json.loads(result_path.read_text())
    if payload.get("ok") is not True:
        raise RuntimeError(f"Workflow {name} did not report success")
    return payload


def run_training_workflow(
    *,
    base_path,
    train_data,
    config,
    validation_data=None,
    test_data=None,
    new_tokens=None,
    initialization="mean",
    inference=False,
    max_sequence_length=128,
    max_new_tokens=16,
    system_prompt=None,
    inference_device="cpu",
    timeout=900,
):
    """Train/export locally; optional fresh-process generation with exact match.

    Datasets are JSONL paths or lists of prompt/completion dictionaries. Split
    overlap is rejected by explicit id when present and exact example content.
    This initial driver launches one training process; use the Trainer/runtime
    APIs and distributed workers for multi-rank jobs.
    """
    require_single_process("Training workflow driver")
    config.validate()
    if max_sequence_length <= 0 or max_new_tokens <= 0:
        raise ValueError("Sequence lengths must be positive")
    splits = dict(
        train=read_examples(train_data),
        validation=(
            read_examples(validation_data) if validation_data is not None else None
        ),
        test=read_examples(test_data) if test_data is not None else None,
    )
    for field in ("id", "content"):
        seen = set()
        for name, examples in splits.items():
            if examples is None:
                continue
            keys = [
                (
                    (row["prompt"], row["completion"])
                    if field == "content"
                    else str(row["id"])
                )
                for row in examples
                if field == "content" or "id" in row
            ]
            if seen.intersection(keys):
                raise ValueError(f"Dataset split overlap in {name} ({field})")
            if field == "id" and len(keys) != len(set(keys)):
                raise ValueError(f"Duplicate sample id in {name}")
            seen.update(keys)
    scoring = splits["test"] if splits["test"] is not None else splits["validation"]
    if inference and scoring is None:
        raise ValueError("Inference requires validation or test data")
    if find_spec("llm_training_data") is None:
        raise ImportError(
            "The portable workflow requires llm-training-data; install the companion package first"
        )
    root = Path(config.output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Workflow requires a new output directory")
    root.mkdir(parents=True, exist_ok=True)
    stage_dir = root / "workflow"
    stage_dir.mkdir()
    actual_base = str(Path(base_path).resolve())
    tokens = {}
    if (Path(actual_base) / "artifact.json").is_file():
        base_manifest = ArtifactManifest.load(actual_base)
        base_manifest.validate(actual_base)
        tokens = dict(base_manifest.new_tokens)
    if new_tokens is not None:
        expanded = expand_vocabulary(
            actual_base,
            root / "expanded-base",
            new_tokens,
            initialization=initialization,
            seed=config.seed,
        )
        actual_base, tokens = expanded.model_path, expanded.token_ids
        if (
            config.trainable_tokens is not None
            and config.trainable_tokens.n_tokens_added != expanded.n_tokens_added
        ):
            raise ValueError(
                "trainable_tokens.n_tokens_added differs from actual vocabulary additions"
            )
    elif config.trainable_tokens is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(actual_base, local_files_only=True)
        count = config.trainable_tokens.n_tokens_added
        if tokens and len(tokens) != count:
            raise ValueError(
                "trainable_tokens.n_tokens_added differs from the expanded base manifest"
            )
        if not tokens:
            tokens = {
                token: index
                for token, index in tokenizer.get_vocab().items()
                if index >= len(tokenizer) - count
            }
    if config.checkpoint.resume_from is not None:
        from .checkpoint import resolve_resume_checkpoint

        checkpoint = resolve_resume_checkpoint(root, config.checkpoint.resume_from)
        ArtifactManifest.load(checkpoint).validate(checkpoint, base_path=actual_base)
    request = dict(
        base_path=actual_base,
        config=config.to_dict(),
        splits=splits,
        new_tokens=tokens,
        max_sequence_length=max_sequence_length,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        inference_device=inference_device,
    )
    request["config"]["output_dir"] = str(root)
    trained = _stage("train", request, stage_dir, timeout)
    artifact = trained["adapter_path"] or trained["model_path"]
    ArtifactManifest.load(artifact).validate(artifact, base_path=actual_base)
    if trained["adapter_path"]:
        merged = _stage(
            "merge",
            dict(request, adapter_path=trained["adapter_path"]),
            stage_dir,
            timeout,
        )
        model_path = merged["model_path"]
    else:
        model_path = trained["model_path"]
    ArtifactManifest.load(model_path).validate(model_path, base_path=actual_base)
    evaluation = dict(validation=trained["evaluation"])
    predictions_path = None
    if inference:
        generated = _stage(
            "infer",
            dict(request, model_path=model_path, score_data=scoring),
            stage_dir,
            timeout,
        )
        predictions_path = generated["predictions_path"]
        evaluation["generation"] = generated["metrics"]
    result = WorkflowResult(
        model_path,
        trained["checkpoint_path"],
        trained["adapter_path"],
        predictions_path,
        evaluation,
    )
    (root / "workflow-result.json").write_text(
        json.dumps(asdict(result), indent=2) + "\n"
    )
    return result
