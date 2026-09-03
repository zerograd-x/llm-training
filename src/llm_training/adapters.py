from __future__ import annotations

from dataclasses import dataclass
import importlib.util

from torch import nn

from .config import FreezeConfig, LoRAConfig, TrainableTokensConfig
from .freeze import apply_freeze_policy, summarize_trainable_parameters
from .model_introspection import (
    find_embedding,
    find_lm_head,
    find_transformer_layers,
    iter_layer_module_names,
    weights_are_tied,
)


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
    full_train_base_parameter_names: tuple[str, ...] = ()
    full_train_base_wrapped_parameter_names: tuple[str, ...] = ()
    trainable_token_ids: tuple[int, ...] = ()

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


def discover_trainable_token_target_modules(model: nn.Module) -> tuple[str, ...]:
    """Resolve embedding targets for standalone PEFT TrainableTokens.

    The input embedding is always targeted. For untied models the output
    embedding / LM head is targeted separately so the selected token rows can
    learn on both sides of the language-model objective. For tied models only
    the input embedding is named explicitly; PEFT discovers the tied output
    weight and attaches its tied TrainableTokens wrapper itself.

    Fully-qualified names are used instead of broad leaf-name suffixes so a
    multimodal model with multiple embedding modules is not adapted by
    accident.
    """

    embedding_name, embedding = find_embedding(model)
    if not _module_has_matrix_weight(embedding):
        raise TypeError(
            f"Input embedding module {embedding_name!r} does not expose a 2D weight"
        )

    targets = [embedding_name]
    lm_head = find_lm_head(model)
    if lm_head is not None and not weights_are_tied(model):
        lm_head_name, lm_head_module = lm_head
        if not _module_has_matrix_weight(lm_head_module):
            raise TypeError(
                f"Output embedding module {lm_head_name!r} does not expose a 2D weight"
            )
        targets.append(lm_head_name)

    return tuple(targets)


def resolve_trainable_token_ids(
    model: nn.Module,
    config: TrainableTokensConfig,
) -> tuple[int, ...]:
    """Return the final N vocabulary rows selected by TrainableTokensConfig."""

    embedding_name, embedding = find_embedding(model)
    weight = getattr(embedding, "weight", None)
    if not isinstance(weight, nn.Parameter) or weight.ndim != 2:
        raise TypeError(
            f"Input embedding module {embedding_name!r} does not expose a 2D parameter weight"
        )

    vocab_size = int(weight.shape[0])
    if config.n_tokens_added > vocab_size:
        raise ValueError(
            f"trainable_tokens.n_tokens_added={config.n_tokens_added} exceeds "
            f"the resized embedding vocabulary size {vocab_size}"
        )

    token_ids = tuple(range(vocab_size - config.n_tokens_added, vocab_size))

    lm_head = find_lm_head(model)
    if lm_head is not None and not weights_are_tied(model):
        lm_head_name, lm_head_module = lm_head
        output_weight = getattr(lm_head_module, "weight", None)
        if not isinstance(output_weight, nn.Parameter) or output_weight.ndim != 2:
            raise TypeError(
                f"Output embedding module {lm_head_name!r} does not expose a 2D parameter weight"
            )
        if int(output_weight.shape[0]) != vocab_size:
            raise ValueError(
                "TrainableTokens requires untied input/output vocabulary dimensions "
                f"to match, got input={vocab_size} and output={int(output_weight.shape[0])}"
            )

    return token_ids


def _restore_full_train_base_parameters(
    model: nn.Module,
    parameter_ids: set[int],
) -> None:
    for parameter in model.parameters():
        if id(parameter) in parameter_ids:
            parameter.requires_grad = True


def _require_peft(method: str) -> None:
    if importlib.util.find_spec("peft") is None:
        raise ImportError(
            f"{method} was requested but PEFT is not installed. "
            "Install llm-training with the 'peft' extra."
        )


def _adapt_with_trainable_tokens(
    model: nn.Module,
    config: TrainableTokensConfig,
) -> tuple[nn.Module, ModelAdaptationSummary]:
    _require_peft("TrainableTokens")

    from peft import TaskType, TrainableTokensConfig as PeftTrainableTokensConfig
    from peft import get_peft_model

    target_modules = discover_trainable_token_target_modules(model)
    token_ids = resolve_trainable_token_ids(model, config)
    peft_config = PeftTrainableTokensConfig(
        task_type=TaskType.CAUSAL_LM,
        token_indices=list(token_ids),
        target_modules=list(target_modules),
        init_weights=True,
    )
    adapted_model = get_peft_model(
        model,
        peft_config,
        adapter_name=config.adapter_name,
    )

    trainable_names = [
        name for name, parameter in adapted_model.named_parameters() if parameter.requires_grad
    ]
    if not trainable_names or not all(
        "trainable_tokens_delta" in name for name in trainable_names
    ):
        raise RuntimeError(
            "PEFT TrainableTokens produced unexpected trainable parameters: "
            + ", ".join(trainable_names[:5])
        )

    summary = summarize_trainable_parameters(adapted_model)
    return adapted_model, ModelAdaptationSummary(
        method="trainable_tokens",
        target_modules=target_modules,
        trainable_parameters=summary.trainable_parameters,
        total_parameters=summary.total_parameters,
        full_train_base_parameters=0,
        trainable_token_ids=token_ids,
    )


def adapt_model_for_training(
    model: nn.Module,
    *,
    freeze: FreezeConfig | None,
    lora: LoRAConfig | None,
    trainable_tokens: TrainableTokensConfig | None = None,
) -> tuple[nn.Module, ModelAdaptationSummary]:
    """Apply base freeze policy and optional PEFT adaptation before optimizer creation.

    ``freeze=None`` preserves the ordinary full-fine-tuning path. With LoRA,
    ``freeze=None`` means standard adapter-only training because PEFT freezes
    the base model. Supplying ``FreezeConfig`` explicitly identifies base
    parameters that should remain fully trainable after PEFT wraps the model.

    Standalone TrainableTokens has a deliberately narrower contract: only the
    selected final vocabulary rows are trainable. Combining it with FreezeConfig
    or LoRA is rejected rather than silently overriding one adaptation with
    another.
    """

    if trainable_tokens is not None:
        if lora is not None:
            raise ValueError("LoRA and TrainableTokens cannot be adapted together yet")
        if freeze is not None:
            raise ValueError("FreezeConfig cannot be combined with standalone TrainableTokens yet")
        return _adapt_with_trainable_tokens(model, trainable_tokens)

    intended_full_train_ids: set[int] = set()
    intended_full_train_records: list[tuple[int, str]] = []
    if freeze is not None:
        apply_freeze_policy(model, freeze)
        intended_full_train_records = [
            (id(parameter), name)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        intended_full_train_ids = {
            parameter_id for parameter_id, _ in intended_full_train_records
        }

    if lora is None:
        summary = summarize_trainable_parameters(model)
        base_trainable_names = tuple(
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        return model, ModelAdaptationSummary(
            method=None,
            target_modules=(),
            trainable_parameters=summary.trainable_parameters,
            total_parameters=summary.total_parameters,
            full_train_base_parameters=summary.trainable_parameters,
            full_train_base_parameter_names=base_trainable_names,
            full_train_base_wrapped_parameter_names=base_trainable_names,
        )

    base_parameter_by_id = {
        id(parameter): parameter for parameter in model.parameters()
    }
    full_train_base_parameters = sum(
        base_parameter_by_id[parameter_id].numel()
        for parameter_id, _ in intended_full_train_records
    )

    _require_peft("LoRA")

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

    wrapped_name_by_id = {
        id(parameter): name
        for name, parameter in adapted_model.named_parameters()
    }
    missing_wrapped_names = [
        original_name
        for parameter_id, original_name in intended_full_train_records
        if parameter_id not in wrapped_name_by_id
    ]
    if missing_wrapped_names:
        preview = ", ".join(missing_wrapped_names[:5])
        raise RuntimeError(
            "PEFT wrapping replaced intentionally full-trained base parameters; "
            "cannot preserve hybrid LoRA artifact identity for: "
            f"{preview}"
        )

    full_train_base_parameter_names = tuple(
        original_name
        for _, original_name in intended_full_train_records
    )
    full_train_base_wrapped_parameter_names = tuple(
        wrapped_name_by_id[parameter_id]
        for parameter_id, _ in intended_full_train_records
    )

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
        full_train_base_parameter_names=full_train_base_parameter_names,
        full_train_base_wrapped_parameter_names=full_train_base_wrapped_parameter_names,
    )
