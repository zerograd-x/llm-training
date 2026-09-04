"""Offline, deterministic vocabulary expansion with explicit token identities."""

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .distributed import require_single_process
from .evaluation import preserve_rng
from .manifest import tokenizer_fingerprint, write_artifact_manifest


@dataclass(frozen=True)
class VocabularyResult:
    model_path: str
    n_tokens_added: int
    token_ids: dict[str, int]
    tokenizer_fingerprint: str


def expand_vocabulary(base_path, output_dir, tokens, *, initialization="mean", seed=42):
    """Append unique missing tokens; keep all existing embedding rows exact.

    ``mean`` uses the mean old row; ``random`` uses seeded normal initialization
    at the old matrix's standard deviation. Input/output tables are handled
    separately when untied. Inputs must have matching tokenizer/embedding sizes.
    """
    require_single_process("Vocabulary expansion")
    if initialization not in {"mean", "random"}:
        raise ValueError("initialization must be mean or random")
    if isinstance(tokens, str):
        raise ValueError("tokens must be a sequence, not a string")
    tokens = list(tokens)
    if any(not isinstance(t, str) or not t.strip() for t in tokens):
        raise ValueError("tokens must be a sequence of nonempty strings")
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Use an empty vocabulary output directory")
    with preserve_rng(torch.device("cpu")):
        torch.manual_seed(seed)
        tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(base_path, local_files_only=True)
        old_count = len(tokenizer)
        input_weights = model.get_input_embeddings().weight.detach().clone()
        output_layer = model.get_output_embeddings()
        output_weights = (
            output_layer.weight.detach().clone() if output_layer is not None else None
        )
        if input_weights.shape[0] != old_count or (
            output_weights is not None and output_weights.shape[0] != old_count
        ):
            raise ValueError(
                "Tokenizer and base embedding sizes must match before expansion"
            )
        existing = tokenizer.get_vocab()
        requested = list(dict.fromkeys(t for t in tokens if t not in existing))
        added = tokenizer.add_tokens(requested)
        ids = {token: tokenizer.convert_tokens_to_ids(token) for token in requested}
        if sorted(ids.values()) != list(range(old_count, old_count + added)):
            raise ValueError("New tokens must occupy contiguous appended IDs")
        if added:
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
            with torch.no_grad():
                tables = [(model.get_input_embeddings().weight, input_weights)]
                if (
                    output_weights is not None
                    and model.get_output_embeddings().weight
                    is not model.get_input_embeddings().weight
                ):
                    tables.append(
                        (model.get_output_embeddings().weight, output_weights)
                    )
                for target, original in tables:
                    target[:old_count].copy_(original)
                    if initialization == "mean":
                        target[old_count:].copy_(
                            original.float().mean(0).to(target.dtype)
                        )
                    else:
                        target[old_count:].normal_(std=original.float().std().item())
        model.save_pretrained(destination, safe_serialization=True)
        tokenizer.save_pretrained(destination)
        write_artifact_manifest(
            destination,
            kind="base",
            base_path=base_path,
            tokenizer=tokenizer,
            new_tokens=ids,
        )
        return VocabularyResult(
            str(destination.resolve()), added, ids, tokenizer_fingerprint(tokenizer)
        )
