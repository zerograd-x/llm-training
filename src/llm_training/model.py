from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class CausalLMComponents:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase


def load_causal_lm(
    model_name_or_path: str,
    *,
    tokenizer_name_or_path: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
) -> CausalLMComponents:
    """Load a decoder-only Hugging Face model and its tokenizer."""

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path or model_name_or_path,
        **(tokenizer_kwargs or {}),
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define either pad_token_id or eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **(model_kwargs or {}),
    )
    return CausalLMComponents(model=model, tokenizer=tokenizer)


def save_causal_lm(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str | Path,
    *,
    accelerator: Accelerator | None = None,
) -> None:
    """Export model weights and tokenizer in ordinary Hugging Face format."""

    output_dir = Path(output_dir)
    if accelerator is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        return

    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        safe_serialization=True,
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
    accelerator.wait_for_everyone()
