from __future__ import annotations

from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import shutil
from typing import Any

from accelerate import Accelerator
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from .adapters import ModelAdaptationSummary
from .config import TrainingConfig
from .distributed import require_single_process


_BASE_TRAINABLE_STATE_FILENAME = "base_trainable_state.safetensors"


def _require_peft() -> None:
    if importlib.util.find_spec("peft") is None:
        raise ImportError(
            "PEFT is required for adapter artifacts. "
            "Install llm-training with the 'peft' extra."
        )


def _adapter_output_dir(root: Path, adapter_name: str) -> Path:
    return root if adapter_name == "default" else root / adapter_name


def _unique_parameters(parameters: list[nn.Parameter]) -> list[nn.Parameter]:
    unique: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in parameters:
        parameter_id = id(parameter)
        if parameter_id in seen:
            continue
        seen.add(parameter_id)
        unique.append(parameter)
    return unique


def _zero3_gather_context(
    parameters: list[nn.Parameter],
    config: TrainingConfig,
):
    if config.deepspeed is None or config.deepspeed.zero_stage != 3:
        return nullcontext()

    import deepspeed

    return deepspeed.zero.GatheredParameters(
        _unique_parameters(parameters),
        modifier_rank=0,
    )


def _write_adapter_metadata(
    *,
    artifact_dir: Path,
    config: TrainingConfig,
    tokenizer: PreTrainedTokenizerBase | None,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "training_config.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    if tokenizer is not None:
        tokenizer.save_pretrained(artifact_dir)


def _resolve_lora_artifact_parameters(
    model: nn.Module,
    adaptation_summary: ModelAdaptationSummary,
) -> tuple[list[nn.Parameter], list[tuple[str, nn.Parameter]]]:
    """Split live trainables into LoRA tensors and canonical base overrides."""

    canonical_names = adaptation_summary.full_train_base_parameter_names
    wrapped_names = adaptation_summary.full_train_base_wrapped_parameter_names
    if len(canonical_names) != len(wrapped_names):
        raise RuntimeError(
            "Hybrid LoRA artifact metadata is inconsistent: canonical and wrapped "
            "base-parameter name counts differ"
        )
    if adaptation_summary.full_train_base_parameters and not canonical_names:
        raise RuntimeError(
            "Hybrid LoRA has full-trained base parameters but no canonical names "
            "were recorded during model adaptation"
        )

    named_parameters = dict(model.named_parameters())
    base_entries: list[tuple[str, nn.Parameter]] = []
    base_parameter_ids: set[int] = set()
    for canonical_name, wrapped_name in zip(canonical_names, wrapped_names):
        parameter = named_parameters.get(wrapped_name)
        if parameter is None:
            raise RuntimeError(
                "Cannot find a full-trained base parameter after backend preparation: "
                f"{wrapped_name}"
            )
        if not parameter.requires_grad:
            raise RuntimeError(
                "A base parameter recorded for hybrid LoRA is no longer trainable: "
                f"{wrapped_name}"
            )
        base_entries.append((canonical_name, parameter))
        base_parameter_ids.add(id(parameter))

    adapter_parameters: list[nn.Parameter] = []
    unexpected_trainables: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in base_parameter_ids:
            continue
        if "lora_" in name:
            adapter_parameters.append(parameter)
        else:
            unexpected_trainables.append(name)

    if unexpected_trainables:
        preview = ", ".join(unexpected_trainables[:5])
        raise NotImplementedError(
            "LoRA artifact export found trainable parameters that are neither LoRA "
            "parameters nor FreezeConfig-managed base parameters: "
            f"{preview}"
        )
    if not adapter_parameters:
        raise RuntimeError("LoRA model has no trainable adapter parameters")

    return _unique_parameters(adapter_parameters), base_entries


def save_lora_adapter(
    model: nn.Module,
    *,
    accelerator: Accelerator,
    output_dir: str | Path,
    config: TrainingConfig,
    adaptation_summary: ModelAdaptationSummary,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> Path:
    """Save a PEFT LoRA artifact without consolidating the full base model."""

    _require_peft()
    if config.lora is None or adaptation_summary.method != "lora":
        raise ValueError("save_lora_adapter requires a LoRA-adapted model")
    if config.lora.bias != "none":
        raise NotImplementedError(
            "LoRA artifact export currently requires lora.bias='none'. "
            "PEFT bias-training artifact semantics are not validated yet."
        )

    from peft import PeftModel, get_peft_model_state_dict

    unwrapped = accelerator.unwrap_model(model)
    if not isinstance(unwrapped, PeftModel):
        raise TypeError("Expected accelerator.unwrap_model(model) to be a PeftModel")

    adapter_name = config.lora.adapter_name
    adapter_parameters, base_entries = _resolve_lora_artifact_parameters(
        unwrapped,
        adaptation_summary,
    )
    parameters_to_gather = _unique_parameters(
        adapter_parameters + [parameter for _, parameter in base_entries]
    )

    adapter_state: dict[str, Any] | None = None
    base_state: dict[str, torch.Tensor] | None = None
    with _zero3_gather_context(parameters_to_gather, config):
        if accelerator.is_main_process:
            state = get_peft_model_state_dict(
                unwrapped,
                adapter_name=adapter_name,
            )
            adapter_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in state.items()
            }
            base_state = {
                canonical_name: parameter.detach().cpu().clone().contiguous()
                for canonical_name, parameter in base_entries
            }

    root = Path(output_dir)
    artifact_dir = _adapter_output_dir(root, adapter_name)
    if accelerator.is_main_process:
        root.mkdir(parents=True, exist_ok=True)
        assert adapter_state is not None
        assert base_state is not None
        unwrapped.save_pretrained(
            root,
            selected_adapters=[adapter_name],
            state_dict=adapter_state,
            safe_serialization=True,
            is_main_process=True,
        )
        _write_adapter_metadata(
            artifact_dir=artifact_dir,
            config=config,
            tokenizer=tokenizer,
        )

        base_state_path = artifact_dir / _BASE_TRAINABLE_STATE_FILENAME
        if base_state:
            from safetensors.torch import save_file

            save_file(
                base_state,
                str(base_state_path),
                metadata={"format": "llm-training-hybrid-lora-base-state-v1"},
            )
        elif base_state_path.exists():
            base_state_path.unlink()

    accelerator.wait_for_everyone()
    return artifact_dir


def _trainable_tokens_parameters(model: nn.Module) -> list[nn.Parameter]:
    trainable: list[nn.Parameter] = []
    unexpected: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "trainable_tokens_delta" in name:
            trainable.append(parameter)
        else:
            unexpected.append(name)

    if unexpected:
        preview = ", ".join(unexpected[:5])
        raise RuntimeError(
            "Standalone TrainableTokens artifact export expected only token-delta "
            f"parameters to be trainable, found: {preview}"
        )
    if not trainable:
        raise RuntimeError("TrainableTokens model has no trainable token-delta parameters")
    return _unique_parameters(trainable)


def save_trainable_tokens_adapter(
    model: nn.Module,
    *,
    accelerator: Accelerator,
    output_dir: str | Path,
    config: TrainingConfig,
    adaptation_summary: ModelAdaptationSummary,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> Path:
    """Save a compact standalone PEFT TrainableTokens artifact.

    The resized base embedding matrix is deliberately *not* serialized. PEFT's
    default ``save_embedding_layers='auto'`` can save the full embedding matrix
    after a vocabulary resize, which defeats partial-token training. This path
    forces ``save_embedding_layers=False`` and saves only TrainableTokens state.

    Under ZeRO-3 only the trainable token-delta parameters are gathered for
    serialization. Real multi-rank ZeRO-3 behavior remains a separate runtime
    validation item.
    """

    _require_peft()
    if (
        config.trainable_tokens is None
        or adaptation_summary.method != "trainable_tokens"
    ):
        raise ValueError(
            "save_trainable_tokens_adapter requires a TrainableTokens-adapted model"
        )

    from peft import PeftModel, get_peft_model_state_dict

    unwrapped = accelerator.unwrap_model(model)
    if not isinstance(unwrapped, PeftModel):
        raise TypeError("Expected accelerator.unwrap_model(model) to be a PeftModel")

    adapter_name = config.trainable_tokens.adapter_name
    adapter_parameters = _trainable_tokens_parameters(unwrapped)

    adapter_state: dict[str, Any] | None = None
    with _zero3_gather_context(adapter_parameters, config):
        if accelerator.is_main_process:
            state = get_peft_model_state_dict(
                unwrapped,
                adapter_name=adapter_name,
                save_embedding_layers=False,
            )
            adapter_state = {
                name: tensor.detach().cpu().clone().contiguous()
                for name, tensor in state.items()
            }

    root = Path(output_dir)
    artifact_dir = _adapter_output_dir(root, adapter_name)
    if accelerator.is_main_process:
        root.mkdir(parents=True, exist_ok=True)
        assert adapter_state is not None
        unwrapped.save_pretrained(
            root,
            selected_adapters=[adapter_name],
            state_dict=adapter_state,
            safe_serialization=True,
            save_embedding_layers=False,
            is_main_process=True,
        )
        _write_adapter_metadata(
            artifact_dir=artifact_dir,
            config=config,
            tokenizer=tokenizer,
        )

    accelerator.wait_for_everyone()
    return artifact_dir


def _tokenizer_source(
    base_model_name_or_path: str | Path,
    adapter_path: Path,
    explicit_source: str | Path | None,
) -> str | Path | None:
    if explicit_source is not None:
        return explicit_source

    if (adapter_path / "tokenizer_config.json").is_file():
        return adapter_path

    base_path = Path(base_model_name_or_path)
    if base_path.is_dir():
        if (
            (base_path / "tokenizer_config.json").is_file()
            or (base_path / "tokenizer.json").is_file()
        ):
            return base_path
        return None

    return base_model_name_or_path


def _apply_base_trainable_state(model: nn.Module, adapter_path: Path) -> None:
    state_path = adapter_path / _BASE_TRAINABLE_STATE_FILENAME
    if not state_path.is_file():
        return

    from safetensors.torch import load_file

    state = load_file(str(state_path), device="cpu")
    if not state:
        raise ValueError(f"Hybrid LoRA base-state artifact is empty: {state_path}")

    base_parameters = dict(model.named_parameters())
    missing_names = [name for name in state if name not in base_parameters]
    if missing_names:
        preview = ", ".join(missing_names[:5])
        raise KeyError(
            "Hybrid LoRA base-state parameters do not exist in the supplied base "
            f"model: {preview}"
        )

    shape_mismatches = [
        f"{name}: artifact={tuple(tensor.shape)} base={tuple(base_parameters[name].shape)}"
        for name, tensor in state.items()
        if tensor.shape != base_parameters[name].shape
    ]
    if shape_mismatches:
        raise ValueError(
            "Hybrid LoRA base-state shape mismatch; the supplied base model is not "
            "compatible with the training artifact: " + "; ".join(shape_mismatches[:5])
        )

    with torch.no_grad():
        for name, tensor in state.items():
            parameter = base_parameters[name]
            if parameter.is_meta:
                raise ValueError(
                    "Cannot restore hybrid LoRA base state into a meta parameter: "
                    f"{name}"
                )
            parameter.copy_(
                tensor.to(device=parameter.device, dtype=parameter.dtype)
            )


def _save_merged_model(
    merged_model: nn.Module,
    *,
    destination: Path,
    base_model_name_or_path: str | Path,
    adapter_path: Path,
    tokenizer_name_or_path: str | Path | None,
    tokenizer_kwargs: dict[str, Any] | None,
    safe_serialization: bool,
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(
        destination,
        safe_serialization=safe_serialization,
    )

    tokenizer_source = _tokenizer_source(
        base_model_name_or_path,
        adapter_path,
        tokenizer_name_or_path,
    )
    if tokenizer_source is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            **(tokenizer_kwargs or {}),
        )
        tokenizer.save_pretrained(destination)

    training_config = adapter_path / "training_config.json"
    if training_config.is_file():
        shutil.copy2(training_config, destination / training_config.name)

    return destination


def merge_lora(
    base_model_name_or_path: str | Path,
    adapter_path: str | Path,
    output_dir: str | Path,
    *,
    model_kwargs: dict[str, Any] | None = None,
    tokenizer_name_or_path: str | Path | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
    safe_serialization: bool = True,
) -> Path:
    """Offline-merge a LoRA artifact into its exact base model."""

    require_single_process("merge_lora")
    _require_peft()
    from peft import PeftModel

    adapter_path = Path(adapter_path)
    if not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"Missing PEFT adapter config: {adapter_path / 'adapter_config.json'}"
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        **(model_kwargs or {}),
    )
    _apply_base_trainable_state(base_model, adapter_path)

    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = peft_model.merge_and_unload()
    return _save_merged_model(
        merged_model,
        destination=Path(output_dir),
        base_model_name_or_path=base_model_name_or_path,
        adapter_path=adapter_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        tokenizer_kwargs=tokenizer_kwargs,
        safe_serialization=safe_serialization,
    )


def merge_trainable_tokens(
    base_model_name_or_path: str | Path,
    adapter_path: str | Path,
    output_dir: str | Path,
    *,
    model_kwargs: dict[str, Any] | None = None,
    tokenizer_name_or_path: str | Path | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
    safe_serialization: bool = True,
) -> Path:
    """Offline-merge TrainableTokens into the exact vocabulary-expanded base.

    ``base_model_name_or_path`` must identify the same resized base checkpoint
    used for training. The small TrainableTokens adapter replaces only the
    selected token rows during ``merge_and_unload()`` and the result is saved as
    an ordinary Hugging Face causal-LM checkpoint.
    """

    require_single_process("merge_trainable_tokens")
    _require_peft()
    from peft import PeftConfig, PeftModel

    adapter_path = Path(adapter_path)
    if not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"Missing PEFT adapter config: {adapter_path / 'adapter_config.json'}"
        )

    peft_config = PeftConfig.from_pretrained(adapter_path)
    if str(peft_config.peft_type).split(".")[-1] != "TRAINABLE_TOKENS":
        raise ValueError(
            "merge_trainable_tokens requires a PEFT TRAINABLE_TOKENS adapter, "
            f"got {peft_config.peft_type}"
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        **(model_kwargs or {}),
    )
    embedding = base_model.get_input_embeddings()
    vocab_size = int(embedding.weight.shape[0])
    token_indices = tuple(int(index) for index in peft_config.token_indices)
    if not token_indices:
        raise ValueError("TrainableTokens adapter contains no token indices")
    if min(token_indices) < 0 or max(token_indices) >= vocab_size:
        raise ValueError(
            "The supplied base model is not the vocabulary-expanded base used "
            f"for training: adapter token ids reach {max(token_indices)}, but "
            f"base vocabulary size is {vocab_size}"
        )

    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = peft_model.merge_and_unload()
    return _save_merged_model(
        merged_model,
        destination=Path(output_dir),
        base_model_name_or_path=base_model_name_or_path,
        adapter_path=adapter_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        tokenizer_kwargs=tokenizer_kwargs,
        safe_serialization=safe_serialization,
    )
