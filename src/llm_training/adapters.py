from __future__ import annotations

from dataclasses import dataclass
import importlib.util

from torch import nn

from .config import FreezeConfig, LoRAConfig
from .freeze import apply_freeze_policy, summarize_trainable_parameters
from .model_introspection import find_transformer_layers, iter_layer_module_names


_DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "c_attn",
    "c_proj",
    "c_fc",
    "query_key_value",
    "dense",
    "dense_h_to_4h",
    "dense_4h_to_h",
)


@dataclass(frozen=True)
class ModelAdaptationSummary:
    method: str | None
    target_modules: tuple[str, ...]
    trainable_parameters: int
    total_parameters: int
    full_train_base_parameters: int

    @property
    def trainable_fraction(self) -> float:
        if self.total_parameters == 0:
            return 0.0
        return self.trainable_parameters / self.total_parameters


def _module_has_matrix_weight(module: nn.Module) -> bool:
    weight = getattr(module, "weight", None)
    return isinstance(weight, nn.Parameter) and weight.ndim == 2


def _matches_target(full_name: str, target: str) -> bool:
    return full_name == target or full_name.endswith("." + target)


def discover_lora_target_modules(
    model: nn.Module,
    config: LoRAConfig,
) -> tuple[str, ...]:
    """Resolve LoRA targets to fully-qualified transformer-block module names."""

    _, layers = find_transformer_layers(model)
    selected_layers = list(layers)
    if config.target_last_n_layers is not None:
        if config.target_last_n_layers > len(selected_layers):
            raise ValueError(
                f"lora.target_last_n_layers={config.target_last_n_layers} exceeds "
                f"the model's {len(selected_layers)} transformer layers"
            )
        selected_layers = selected_layers[-config.target_last_n_layers :]

    patterns = config.target_modules or _DEFAULT_LORA_TARGETS
    matches: list[str] = []
    seen_module_ids: set[int] = set()

    for layer_name, layer in iter_layer_module_names(model, selected_layers):
        for relative_name, module in layer.named_modules():
            if not relative_name:
                continue
            full_name = f"{layer_name}.{relative_name}"
            if not any(_matches_target(full_name, pattern) for pattern in patterns):
                continue
            if not _module_has_matrix_weight(module):
                continue
            if id(module) in seen_module_ids:
                continue
            seen_module_ids.add(id(module))
            matches.append(full_name)

    if not matches:
        raise ValueError(
            "No LoRA target modules were discovered inside transformer blocks. "
            f"Tried target patterns: {', '.join(patterns)}"
        )

    return tuple(matches)


def _restore_full_train_base_parameters(
    model: nn.Module,
    parameter_ids: set[int],
) -> None:
    for parameter in model.parameters():
        if id(parameter) in parameter_ids:
            parameter.requires_grad = True


def adapt_model_for_training(
    model: nn.Module,
    *,
    freeze: FreezeConfig | None,
    lora: LoRAConfig | None,
) -> tuple[nn.Module, ModelAdaptationSummary]:
    """Apply base freeze policy and optional LoRA before optimizer creation.

    ``freeze=None`` preserves the ordinary full-fine-tuning path. With LoRA,
    ``freeze=None`` means standard adapter-only training because PEFT freezes
    the base model. Supplying ``FreezeConfig`` explicitly identifies base
    parameters that should remain fully trainable after PEFT wraps the model.
    """

    intended_full_train_ids: set[int] = set()
    if freeze is not None:
        apply_freeze_policy(model, freeze)
        intended_full_train_ids = {
            id(parameter) for parameter in model.parameters() if parameter.requires_grad
        }

    full_train_base_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if id(parameter) in intended_full_train_ids
    )

    if lora is None:
        summary = summarize_trainable_parameters(model)
        return model, ModelAdaptationSummary(
            method=None,
            target_modules=(),
            trainable_parameters=summary.trainable_parameters,
            total_parameters=summary.total_parameters,
            full_train_base_parameters=full_train_base_parameters,
        )

    if importlib.util.find_spec("peft") is None:
        raise ImportError(
            "LoRA was requested but PEFT is not installed. "
            "Install llm-training with the 'peft' extra."
        )

    from peft import LoraConfig as PeftLoraConfig
    from peft import TaskType, get_peft_model

    target_modules = discover_lora_target_modules(model, lora)
    peft_config = PeftLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora.rank,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        bias=lora.bias,
        target_modules=list(target_modules),
    )
    adapted_model = get_peft_model(
        model,
        peft_config,
        adapter_name=lora.adapter_name,
    )

    _restore_full_train_base_parameters(adapted_model, intended_full_train_ids)

    trainable_names = [
        name for name, parameter in adapted_model.named_parameters() if parameter.requires_grad
    ]
    if not any("lora_" in name for name in trainable_names):
        raise RuntimeError("PEFT wrapping produced no trainable LoRA parameters")

    summary = summarize_trainable_parameters(adapted_model)
    return adapted_model, ModelAdaptationSummary(
        method="lora",
        target_modules=target_modules,
        trainable_parameters=summary.trainable_parameters,
        total_parameters=summary.total_parameters,
        full_train_base_parameters=full_train_base_parameters,
    )
