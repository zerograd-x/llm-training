from __future__ import annotations

import argparse
from pathlib import Path

import torch

from llm_training import load_causal_lm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    components = load_causal_lm(
        str(args.model_dir),
        model_kwargs={"torch_dtype": torch.bfloat16, "device_map": "cuda"},
    )
    batch = components.tokenizer(
        "Product: ceramic coffee mug. Category:",
        return_tensors="pt",
    )
    batch = {key: value.to("cuda") for key, value in batch.items()}

    with torch.no_grad():
        logits = components.model(**batch).logits

    if not torch.isfinite(logits).all():
        raise AssertionError("reloaded model produced non-finite logits")

    print(
        f"PASS qwen35_export_reload model_dir={args.model_dir} "
        f"logits_shape={tuple(logits.shape)}"
    )


if __name__ == "__main__":
    main()
