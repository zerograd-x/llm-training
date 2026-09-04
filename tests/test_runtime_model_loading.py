from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.integrations import deepspeed as hf_deepspeed

from llm_training import DeepSpeedConfig, DistributedRuntime, TrainingConfig
import llm_training.distributed as distributed
import llm_training.model as model_module


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    # Reproduce Accelerate's active HF loading hook despite zero3_init_flag=False.
    # No DeepSpeed installation is needed to verify Transformers skips that hook.
    class HFConfig:
        def is_zero3(self):
            return True

    hf_config = HFConfig()
    monkeypatch.setattr(hf_deepspeed, "_hf_deepspeed_config_weak_ref", None)
    accelerator = SimpleNamespace(
        num_processes=1,
        state=SimpleNamespace(deepspeed_plugin=SimpleNamespace(dschf=hf_config)),
        local_main_process_first=nullcontext,
        wait_for_everyone=lambda: None,
    )
    monkeypatch.setattr(distributed, "build_accelerator", lambda config: accelerator)
    value = DistributedRuntime(
        TrainingConfig(
            output_dir=str(tmp_path / "run"), deepspeed=DeepSpeedConfig(zero_stage=3)
        )
    )
    hf_deepspeed.set_hf_deepspeed_config(hf_config)
    return value


def test_runtime_loads_full_hf_weights_and_restores_zero3(
    runtime, monkeypatch, tmp_path
):
    with runtime.model_loading_context():
        model = GPT2LMHeadModel(
            GPT2Config(vocab_size=16, n_positions=8, n_embd=8, n_layer=1, n_head=1)
        )
        model.save_pretrained(tmp_path / "base")
    tokenizer = SimpleNamespace(pad_token_id=0)
    monkeypatch.setattr(
        model_module.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer
    )
    assert hf_deepspeed.is_deepspeed_zero3_enabled()
    components = runtime.load_causal_lm(str(tmp_path / "base"))
    assert components.model.get_input_embeddings().weight.shape == (16, 8)
    assert all(not hasattr(p, "ds_id") for p in components.model.parameters())
    assert hf_deepspeed.is_deepspeed_zero3_enabled()


def test_model_loading_context_restores_zero3_after_exception_and_nesting(runtime):
    with pytest.raises(ValueError, match="load failed"):
        with runtime.model_loading_context():
            assert not hf_deepspeed.is_deepspeed_zero3_enabled()
            with runtime.model_loading_context():
                assert not hf_deepspeed.is_deepspeed_zero3_enabled()
            assert not hf_deepspeed.is_deepspeed_zero3_enabled()
            raise ValueError("load failed")
    assert hf_deepspeed.is_deepspeed_zero3_enabled()


def test_single_process_runtime_does_not_require_backend_barriers():
    # A single-GPU DeepSpeed plugin can exist before prepare() creates a group.
    # Local model loading/materialization must still be usable at that point.
    from types import SimpleNamespace
    from llm_training import DistributedRuntime

    def premature_barrier():
        raise AssertionError("Backend group has not been initialized")

    runtime = object.__new__(DistributedRuntime)
    runtime.accelerator = SimpleNamespace(
        num_processes=1,
        wait_for_everyone=premature_barrier,
        main_process_first=premature_barrier,
        local_main_process_first=premature_barrier,
    )
    runtime.wait_for_everyone()
    with runtime.main_process_first(), runtime.local_main_process_first():
        pass
