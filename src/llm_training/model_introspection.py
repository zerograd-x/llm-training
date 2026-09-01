from __future__ import annotations

from collections.abc import Iterable

from torch import nn

_LAYER_PATHS = (
    "model.layers",
    "language_model.layers",
    "model.language_model.layers",
    "transformer.h",
    "model.decoder.layers",
)

_EMBED_PATHS = (
    "model.embed_tokens",
    "language_model.model.embed_tokens",
    "model.language_model.embed_tokens",
    "transformer.wte",
    "transformer.word_embeddings",
)


def resolve_path(root: object, dotted_path: str) -> object | None:
    current: object = root
    for part in dotted_path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def _module_name(root: nn.Module, target: nn.Module) -> str | None:
    for name, module in root.named_modules():
        if module is target:
            return name
    return None


def find_transformer_layers(model: nn.Module) -> tuple[str, nn.ModuleList]:
    for path in _LAYER_PATHS:
        value = resolve_path(model, path)
        if isinstance(value, nn.ModuleList):
            return path, value
    raise ValueError(
        "Could not locate the transformer layer list. "
        f"Tried: {', '.join(_LAYER_PATHS)}"
    )


def find_embedding(model: nn.Module) -> tuple[str, nn.Module]:
    getter = getattr(model, "get_input_embeddings", None)
    if callable(getter):
        module = getter()
        if isinstance(module, nn.Module):
            name = _module_name(model, module)
            if name is not None:
                return name, module

    for path in _EMBED_PATHS:
        value = resolve_path(model, path)
        if isinstance(value, nn.Module):
            return path, value
    raise ValueError(
        "Could not locate the input embedding module. "
        f"Tried: {', '.join(_EMBED_PATHS)} and get_input_embeddings()."
    )


def find_lm_head(model: nn.Module) -> tuple[str, nn.Module] | None:
    getter = getattr(model, "get_output_embeddings", None)
    if callable(getter):
        module = getter()
        if isinstance(module, nn.Module):
            name = _module_name(model, module)
            if name is not None:
                return name, module

    value = getattr(model, "lm_head", None)
    if isinstance(value, nn.Module):
        return "lm_head", value
    return None


def weights_are_tied(model: nn.Module) -> bool:
    _, embedding = find_embedding(model)
    lm_head = find_lm_head(model)
    if lm_head is None:
        return False
    _, output = lm_head
    input_weight = getattr(embedding, "weight", None)
    output_weight = getattr(output, "weight", None)
    return input_weight is not None and input_weight is output_weight


def iter_layer_module_names(
    model: nn.Module,
    layers: Iterable[nn.Module],
) -> Iterable[tuple[str, nn.Module]]:
    names_by_id = {id(module): name for name, module in model.named_modules()}
    for layer in layers:
        layer_name = names_by_id.get(id(layer))
        if layer_name is None:
            continue
        yield layer_name, layer
