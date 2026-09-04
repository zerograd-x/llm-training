import pytest
from accelerate import Accelerator
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from llm_training import TrainingConfig
from llm_training.checkpoint import (
    TrainerState,
    load_training_checkpoint,
    save_training_checkpoint,
)


def test_checkpoint_roundtrip_restores_model_optimizer_scheduler_and_rng(
    tmp_path,
) -> None:
    accelerator = Accelerator()
    model = nn.Linear(4, 2)
    optimizer = AdamW(model.parameters(), lr=0.01)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0 / (step + 1))
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    x = torch.ones(2, 4)
    loss = model(x).sum()
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    saved_weight = accelerator.unwrap_model(model).weight.detach().clone()
    saved_lr = optimizer.param_groups[0]["lr"]
    state = TrainerState(global_step=7, epoch=1, batch_in_epoch=3)
    config = TrainingConfig(output_dir=str(tmp_path))

    torch.manual_seed(1234)
    checkpoint_dir = save_training_checkpoint(
        accelerator,
        output_dir=tmp_path,
        state=state,
        config=config,
    )
    expected_random = torch.rand(5)

    with torch.no_grad():
        accelerator.unwrap_model(model).weight.add_(100)
    optimizer.param_groups[0]["lr"] = 99.0
    _ = torch.rand(20)

    restored = load_training_checkpoint(accelerator, checkpoint_dir)
    actual_random = torch.rand(5)

    assert restored == state
    assert torch.equal(accelerator.unwrap_model(model).weight, saved_weight)
    assert optimizer.param_groups[0]["lr"] == saved_lr
    assert torch.equal(actual_random, expected_random)


@pytest.mark.parametrize("damage", ["missing", "truncated", "unfinished"])
def test_incomplete_checkpoint_rejected_before_loading_runtime(
    tmp_path, monkeypatch, damage
):
    accelerator = Accelerator()
    model = nn.Linear(2, 2)
    accelerator.prepare(model)
    path = save_training_checkpoint(
        accelerator,
        output_dir=tmp_path,
        state=TrainerState(global_step=1),
        config=TrainingConfig(output_dir=str(tmp_path)),
    )
    called = []
    monkeypatch.setattr(accelerator, "load_state", lambda *args: called.append(True))
    if damage == "unfinished":
        (path / "checkpoint_manifest.json").unlink()
    else:
        victim = next(path.glob("random_states*"))
        if damage == "missing":
            victim.unlink()
        else:
            victim.write_bytes(b"incomplete")
    with pytest.raises(FileNotFoundError, match="[Cc]heckpoint"):
        load_training_checkpoint(accelerator, path)
    assert not called
