from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


@dataclass
class CausalLMComponents:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase


def _load_causal_lm_model(
    model_name_or_path: str,
    model_kwargs: dict[str, Any],
) -> PreTrainedModel:
    """Load a text-only causal LM, including Qwen3.5 checkpoints.

    Qwen3.5 publishes a multimodal top-level checkpoint (``model_type=qwen3_5``)
    while Transformers exposes ``Qwen3_5ForCausalLM`` for text-only use. A
    previously exported text-only checkpoint may instead have
    ``model_type=qwen3_5_text``. Route both forms through the text-only class so
    training never needs an image processor or vision inputs.
    """

    config = AutoConfig.from_pretrained(model_name_or_path)
    if getattr(config, "model_type", None) in {"qwen3_5", "qwen3_5_text"}:
        try:
            from transformers import Qwen3_5ForCausalLM
        except ImportError as exc:
            raise ImportError(
                "Qwen3.5 text-only training requires a Transformers version "
                "that provides Qwen3_5ForCausalLM"
            ) from exc
        return Qwen3_5ForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )

    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )


def load_causal_lm(
    model_name_or_path: str,
    *,
    tokenizer_name_or_path: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
) -> CausalLMComponents:
    """Load a text-only Hugging Face causal LM and its tokenizer."""

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path or model_name_or_path,
        **(tokenizer_kwargs or {}),
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define either pad_token_id or eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_causal_lm_model(
        model_name_or_path,
        model_kwargs or {},
    )
    return CausalLMComponents(model=model, tokenizer=tokenizer)


def save_causal_lm(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase | None,
    output_dir: str | Path,
    *,
    accelerator: Accelerator | None = None,
) -> None:
    """Export model weights and optional tokenizer in Hugging Face format."""

    output_dir = Path(output_dir)
    if accelerator is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir, safe_serialization=True)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
        return

    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)
    unwrapped.save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=state_dict,
        safe_serialization=True,
    )
    if accelerator.is_main_process and tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    accelerator.wait_for_everyone()
