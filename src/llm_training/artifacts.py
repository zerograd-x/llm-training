from __future__ import annotations

from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import shutil
from typing import Any

from accelerate import Accelerator
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from .adapters import ModelAdaptationSummary
from .config import TrainingConfig


def _require_peft() -> None:
    if importlib.util.find_spec("peft") is None:
        raise ImportError(
            "PEFT is required for LoRA artifacts. "
            "Install llm-training with the 'peft' extra."
        )


def _adapter_output_dir(root: Path, adapter_name: str) -> Path:
    return root if adapter_name == "default" else root / adapter_name


def _pure_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("LoRA model has no trainable parameters")

    non_lora = [name for name, _ in trainable if "lora_" not in name]
    if non_lora:
        preview = ", ".join(non_lora[:5])
        raise NotImplementedError(
            "Adapter-only export currently supports pure LoRA only. "
            "The model also has fully trainable non-LoRA parameters: "
            f"{preview}"
        )
    return [parameter for _, parameter in trainable]


def save_lora_adapter(
    model: nn.Module,
    *,
    accelerator: Accelerator,
    output_dir: str | Path,
    config: TrainingConfig,
    adaptation_summary: ModelAdaptationSummary,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> Path:
    """Save a standard PEFT LoRA artifact without consolidating the base model.

    This function is safe to call from all distributed ranks. Under ZeRO-3 it
    gathers only the trainable LoRA parameters for serialization; it never asks
    Accelerate/DeepSpeed to materialize a full base-model state dict.

    Hybrid LoRA, where non-LoRA base parameters are also fully trained, is
    intentionally rejected until the additional base-state artifact format is
    implemented.
    """

    _require_peft()
    if config.lora is None or adaptation_summary.method != "lora":
        raise ValueError("save_lora_adapter requires a LoRA-adapted model")
    if adaptation_summary.full_train_base_parameters:
        raise NotImplementedError(
            "Adapter-only export for hybrid LoRA is not implemented yet; "
            "fully trained base parameters require a separate base-state artifact."
        )
    if config.lora.bias != "none":
        raise NotImplementedError(
            "Adapter-only export currently requires lora.bias='none'. "
            "Bias-training artifact semantics will be handled with hybrid LoRA state."
        )

    from peft import PeftModel, get_peft_model_state_dict

    unwrapped = accelerator.unwrap_model(model)
    if not isinstance(unwrapped, PeftModel):
        raise TypeError("Expected accelerator.unwrap_model(model) to be a PeftModel")

    adapter_name = config.lora.adapter_name
    adapter_parameters = _pure_lora_parameters(unwrapped)

    gather_context = nullcontext()
    if config.deepspeed is not None and config.deepspeed.zero_stage == 3:
        import deepspeed

        gather_context = deepspeed.zero.GatheredParameters(
            adapter_parameters,
            modifier_rank=0,
        )

    adapter_state: dict[str, Any] | None = None
    with gather_context:
        if accelerator.is_main_process:
            state = get_peft_model_state_dict(
                unwrapped,
                adapter_name=adapter_name,
            )
            adapter_state = {
                name: tensor.detach().cpu().clone()
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
            is_main_process=True,
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "training_config.json").write_text(
            json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n"
        )
        if tokenizer is not None:
            tokenizer.save_pretrained(artifact_dir)

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

    # A non-local string is assumed to be a Hugging Face model/tokenizer id.
    return base_model_name_or_path


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
    """Offline-merge a LoRA adapter into its exact base model.

    The merge intentionally runs outside the distributed training lifecycle:
    load a complete base model, attach the saved PEFT adapter, fold the LoRA
    delta into the base weights with ``merge_and_unload()``, and write a plain
    Hugging Face checkpoint.
    """

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
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = peft_model.merge_and_unload()

    destination = Path(output_dir)
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

    # Preserve generic training metadata without making it part of the HF model
    # loading contract. Storage- or application-specific sidecars stay outside
    # llm-training.
    training_config = adapter_path / "training_config.json"
    if training_config.is_file():
        shutil.copy2(training_config, destination / training_config.name)

    return destination
