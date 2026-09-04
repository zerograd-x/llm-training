"""Effective TrainableTokens row checks, rebuilt from the public contract."""
from __future__ import annotations

from contextlib import nullcontext

import torch


def effective_rows(trainer) -> torch.Tensor:
    """Use the adapter's actual embedding forward, not its frozen base weight."""
    model = trainer.accelerator.unwrap_model(trainer.model)
    embedding = model.get_input_embeddings()
    vocab = int(model.config.vocab_size)
    parameters = list(embedding.parameters())
    if trainer.config.deepspeed is not None:
        import deepspeed
        context = deepspeed.zero.GatheredParameters(parameters, modifier_rank=None)
    else:
        context = nullcontext()
    with context, torch.no_grad():
        ids = torch.arange(vocab, device=trainer.accelerator.device)
        return embedding(ids).detach().cpu().clone()


def assert_token_rows(before: torch.Tensor, after: torch.Tensor,
                      selected: tuple[int, ...]) -> dict:
    assert before.shape == after.shape
    assert before.dtype == after.dtype, "Compare after backend preparation at matching dtype"
    old = torch.ones(before.shape[0], dtype=torch.bool)
    old[list(selected)] = False
    assert torch.equal(before[old], after[old]), "Unselected effective token rows changed"
    changed = (before[list(selected)] != after[list(selected)]).any(dim=1)
    assert bool(changed.any()), "No selected effective token row changed"
    return {"stable_old_rows": True, "changed_selected_rows": int(changed.sum()),
            "selected_token_ids": list(selected), "dtype": str(before.dtype)}
