from pathlib import Path

import torch
from torch import nn

from llm_training import OptimizerConfig, save_causal_lm
from llm_training.optimizer import build_adamw


def test_adamw_uses_one_group_with_trainable_parameters_only() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    model[0].bias.requires_grad = False
    config = OptimizerConfig(
        learning_rate=3e-4,
        weight_decay=0.2,
        beta1=0.8,
        beta2=0.9,
        eps=1e-7,
    )

    optimizer = build_adamw(model, config)

    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert {id(parameter) for parameter in group["params"]} == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    assert group["lr"] == config.learning_rate
    assert group["weight_decay"] == config.weight_decay
    assert group["betas"] == (config.beta1, config.beta2)
    assert group["eps"] == config.eps


class SaveableModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.last_save_kwargs = None

    def save_pretrained(self, output_dir, **kwargs) -> None:
        self.last_save_kwargs = kwargs
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "model.saved").write_text("ok\n")


def test_save_causal_lm_allows_missing_tokenizer(tmp_path) -> None:
    save_causal_lm(SaveableModel(), None, tmp_path)

    assert (tmp_path / "model.saved").is_file()


class FakeAccelerator:
    is_main_process = True

    def __init__(self) -> None:
        self.state_dict_requested = False

    def wait_for_everyone(self) -> None:
        pass

    def unwrap_model(self, model):
        return model

    def get_state_dict(self, model):
        self.state_dict_requested = True
        return {"weight": model.weight.detach().clone()}

    def save(self, *args, **kwargs) -> None:
        pass


def test_save_causal_lm_uses_accelerator_state_dict(tmp_path) -> None:
    model = SaveableModel()
    accelerator = FakeAccelerator()

    save_causal_lm(model, None, tmp_path, accelerator=accelerator)

    assert accelerator.state_dict_requested
    assert model.last_save_kwargs is not None
    assert "state_dict" in model.last_save_kwargs
    torch.testing.assert_close(
        model.last_save_kwargs["state_dict"]["weight"],
        model.weight.detach(),
    )
