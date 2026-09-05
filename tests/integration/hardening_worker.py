"""Fresh-process CPU/CUDA distributed boundaries, evaluation and artifact checks."""

from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import sys
import traceback
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel
from llm_training import (
    CheckpointConfig,
    DeepSpeedConfig,
    DistributedRuntime,
    EvaluationConfig,
    FreezeConfig,
    GradientCheckpointingConfig,
    LoRAConfig,
    OptimizerConfig,
    SchedulerConfig,
    Trainer,
    TrainableTokensConfig,
    TrainingConfig,
    merge_lora,
    merge_trainable_tokens,
    require_single_process,
)
from llm_training.loss import shift_causal_labels


def collate(rows):
    return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def snapshot(trainer):
    model = trainer.accelerator.unwrap_model(trainer.model)
    params = list(model.parameters())
    if trainer.config.deepspeed:
        import deepspeed

        context = deepspeed.zero.GatheredParameters(params, modifier_rank=None)
    else:
        context = nullcontext()
    with context:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
        }


def effective_token_rows(trainer):
    """Observe PEFT's effective embeddings, including its selected-row delta."""
    model = trainer.accelerator.unwrap_model(trainer.model)
    embedding = model.get_input_embeddings()
    context = nullcontext()
    if trainer.config.deepspeed:
        import deepspeed

        context = deepspeed.zero.GatheredParameters(
            list(embedding.parameters()), modifier_rank=None
        )
    with context, torch.no_grad():
        ids = torch.arange(model.config.vocab_size, device=trainer.accelerator.device)
        return embedding(ids).detach().cpu().clone()


def assert_token_rows(before, after):
    assert before.shape == after.shape and before.dtype == after.dtype
    assert torch.equal(before[:-2], after[:-2]), "Unselected token rows changed"
    changed = (before[-2:] != after[-2:]).any(dim=1)
    assert bool(changed.all()), "Both selected token rows must receive updates"
    return dict(
        stable_old_rows=True,
        changed_selected_rows=int(changed.sum()),
        selected_token_ids=[32, 33],
        dtype=str(before.dtype),
    )


def save_adapter_probe(trainer):
    """All ranks gather only trainable weights; only rank zero writes PEFT files."""
    import llm_training.artifacts as artifacts

    model = trainer.accelerator.unwrap_model(trainer.model)
    original_gather = artifacts._zero3_gather_context
    evidence = {}

    def gather(parameters, config):
        assert parameters and all(p.requires_grad for p in parameters)
        evidence["gathered_parameters"] = len(parameters)
        return original_gather(parameters, config)

    with patch.object(artifacts, "_zero3_gather_context", gather), patch.object(
        model, "save_pretrained", wraps=model.save_pretrained
    ) as save_call:
        path = trainer.save_adapter()
        assert save_call.call_count == int(trainer.runtime.is_main_process)
        evidence["save_pretrained_calls"] = save_call.call_count
    if trainer.runtime.is_main_process:
        from safetensors.torch import load_file

        state = load_file(str(path / "adapter_model.safetensors"))
        marker = "lora_" if trainer.config.lora else "trainable_tokens_delta"
        assert state and all(marker in name for name in state)
        assert not (path / "model.safetensors").exists()
        if trainer.config.freeze is None:
            assert not (path / "base_trainable_state.safetensors").exists()
    return evidence


def rng():
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def assert_rng(actual, expected):
    assert torch.equal(actual["torch"], expected["torch"])
    assert len(actual["cuda"]) == len(expected["cuda"])
    assert all(torch.equal(a, b) for a, b in zip(actual["cuda"], expected["cuda"]))
    assert actual["python"] == expected["python"]
    assert actual["numpy"][0] == expected["numpy"][0]
    assert np.array_equal(actual["numpy"][1], expected["numpy"][1])
    assert actual["numpy"][2:] == expected["numpy"][2:]


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


class Rows(torch.utils.data.Dataset):
    def __init__(self, tokens=False):
        self.tokens = tokens

    def __len__(self):
        return 9

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        ids = torch.tensor(
            [1, index + 3, 32 + index % 2 if self.tokens else index + 15, 2]
        )
        labels = ids.clone()
        labels[: 1 + index % 2] = -100
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)


class Stop(RuntimeError):
    pass


class Probe(Trainer):
    stop_step = None
    root = None

    def save_checkpoint(self):
        path = super().save_checkpoint()
        if self.stop_step == self.state.global_step:
            rank = self.runtime.process_index
            save(self.root / f"checkpoint-rng-{rank}.pt", rng())
            save(self.root / f"checkpoint-params-{rank}.pt", snapshot(self))
            save(self.root / f"checkpoint-scheduler-{rank}.pt", self.scheduler.state_dict())
            raise Stop()
        return path


def prepare(case, root):
    torch.manual_seed(123)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_positions=16,
            n_embd=16,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=2,
            attn_pdrop=0.2,
            resid_pdrop=0.2,
            embd_pdrop=0.2,
        )
    )
    if case.startswith("tt_"):
        model.resize_token_embeddings(34)
    model.save_pretrained(root / "base")
    return dict(prepared=True)


def train(case, phase, root):
    rank = int(os.environ["RANK"])
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group(
        "nccl" if cuda else "gloo", timeout=timedelta(seconds=30)
    )
    ds = "zero3" in case
    tokens = case.startswith("tt_")
    cfg = TrainingConfig(
        output_dir=str(
            root / ("baseline-run" if phase == "baseline" else "resumed-run")
        ),
        num_epochs=3,
        seed=123,
        mixed_precision="bf16" if cuda else "no",
        per_device_batch_size=2,
        gradient_accumulation_steps=2,
        log_every_steps=1,
        optimizer=OptimizerConfig(learning_rate=0.01, weight_decay=0),
        scheduler=SchedulerConfig(name="linear", warmup_steps=0),
        evaluation=EvaluationConfig(batch_size=2, every_steps=1),
        deepspeed=DeepSpeedConfig(zero_stage=3) if ds else None,
        lora=(
            LoRAConfig(target_modules=("c_attn",), rank=4, alpha=8)
            if case == "hybrid_zero3" or case.startswith("lora_")
            else None
        ),
        trainable_tokens=TrainableTokensConfig(n_tokens_added=2) if tokens else None,
        gradient_checkpointing=GradientCheckpointingConfig(
            enabled=case.endswith("_gc"), use_reentrant=False
        ),
        freeze=(
            FreezeConfig(
                train_last_n_layers=0,
                train_embeddings=False,
                train_lm_head=False,
                trainable_prefixes=("transformer.ln_f",),
            )
            if case == "hybrid_zero3"
            else None
        ),
        checkpoint=CheckpointConfig(
            save_every_steps=1,
            keep_last=8,
            resume_from="latest" if phase == "resume" else None,
        ),
    )
    runtime = DistributedRuntime(cfg)
    assert runtime.num_processes == 2 and runtime.process_index == rank
    assert runtime.local_process_index == int(os.environ["LOCAL_RANK"])
    assert runtime.is_main_process == (rank == 0)
    assert runtime.is_local_main_process == (runtime.local_process_index == 0)
    identity = dict(
        rank=rank,
        local_rank=runtime.local_process_index,
        host=socket.gethostname(),
        device=str(runtime.accelerator.device),
    )
    identities = [None, None]
    torch.distributed.all_gather_object(identities, identity)
    if os.environ.get("HARDENING_GPU") == "1":
        assert torch.distributed.get_backend() == "nccl"
        assert runtime.accelerator.device.type == "cuda"
        assert torch.cuda.current_device() == runtime.local_process_index
        assert len({item["host"] for item in identities}) == 2
    try:
        require_single_process("offline merge")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Distributed offline merge guard did not reject")
    if case == "failure_rank":
        control_group = torch.distributed.new_group(backend="gloo")
        if rank == 1:
            raise RuntimeError("intentional rank failure")
        torch.distributed.monitored_barrier(
            group=control_group, timeout=timedelta(seconds=10), wait_all_ranks=True
        )
        raise AssertionError("Control-plane failure was not propagated")
    if case == "failure_materialize":

        def bad(path):
            raise ValueError("intentional materialization failure")

        runtime.materialize_path(root / "materialized.txt", bad, scope="global")
        raise AssertionError("Materialization unexpectedly succeeded")
    with runtime.model_loading_context():
        model = AutoModelForCausalLM.from_pretrained(root / "base")
    seen = []

    def track(module, args, kwargs):
        # Hook the prepared outer model. DeepSpeedEngine's own training flag can
        # stay true while its wrapped model is in eval mode, so grad mode is the
        # reliable distinction for this test's train vs. evaluation forwards.
        if torch.is_grad_enabled():
            seen.extend(kwargs["input_ids"][:, 1].detach().cpu().tolist())

    import llm_training.trainer as trainer_module

    original_builder = trainer_module.build_training_optimizer
    coverage = {}

    def build_optimizer(adapted_model, config):
        optimizer = original_builder(adapted_model, config)
        expected = {id(p) for p in adapted_model.parameters() if p.requires_grad}
        actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
        assert expected == set(actual) and len(actual) == len(set(actual))
        coverage.update(expected=len(expected), actual=len(actual), duplicates=0)
        return optimizer

    with patch.object(trainer_module, "build_training_optimizer", build_optimizer):
        trainer = Probe(
            model=model,
            train_dataset=Rows(tokens),
            eval_dataset=Rows(tokens),
            collate_fn=collate,
            config=cfg,
            runtime=runtime,
        )
    assert coverage and trainer.runtime is runtime
    trainer.root = root
    # PEFT may call the base model's forward method directly, bypassing hooks
    # registered on the original model. Observe the prepared outer model.
    trainer.model.register_forward_pre_hook(track, with_kwargs=True)
    if case == "failure_checkpoint":
        from llm_training.checkpoint import load_training_checkpoint

        checkpoint = trainer.save_checkpoint()
        if rank == 0:
            victim = next(checkpoint.glob("random_states*"))
            victim.unlink()
        runtime.wait_for_everyone()
        load_training_checkpoint(trainer.accelerator, checkpoint)
        raise AssertionError("Broken checkpoint unexpectedly loaded")
    if case == "failure_masked":
        original = trainer.loss_fn.forward

        def masked(logits, labels):
            return original(
                logits, torch.full_like(labels, -100) if rank == 1 else labels
            )

        trainer.loss_fn.forward = masked
    unwrapped = runtime.accelerator.unwrap_model(trainer.model)
    trainable = {name for name, p in unwrapped.named_parameters() if p.requires_grad}
    assert trainable
    if cfg.lora:
        assert any("lora_" in name for name in trainable)
        if case == "hybrid_zero3":
            assert any("ln_f" in name for name in trainable)
        else:
            assert all("lora_" in name for name in trainable)
    if tokens:
        assert all("trainable_tokens_delta" in name for name in trainable)
    shards = []
    if ds:
        assert trainer.model.zero_optimization_stage() == 3
        assert trainer.model.bfloat16_enabled() == cuda
        shards = [
            dict(name=name, full=int(p.ds_numel), partition=int(p.ds_tensor.numel()))
            for name, p in unwrapped.named_parameters()
            if hasattr(p, "ds_tensor")
        ]
        assert shards and any(p["partition"] < p["full"] for p in shards)
    if cfg.gradient_checkpointing.enabled:
        assert unwrapped.is_gradient_checkpointing and unwrapped.config.use_cache is False
    stop_step = 2 if case in ("boundary_epoch_ddp", "full_zero3") else 3
    if phase == "resume":
        assert trainer.state.global_step == stop_step
        assert_rng(
            rng(), load(root / f"checkpoint-rng-{rank}.pt")
        )
        before = snapshot(trainer)
        saved = load(root / f"checkpoint-params-{rank}.pt")
        for name in before:
            torch.testing.assert_close(before[name], saved[name], atol=0, rtol=0)
        assert trainer.scheduler.state_dict() == load(root / f"checkpoint-scheduler-{rank}.pt")
    before = snapshot(trainer)
    token_before = effective_token_rows(trainer) if tokens else None
    if phase == "interrupt":
        trainer.stop_step = stop_step
    try:
        trainer.train()
    except Stop:
        assert phase == "interrupt"
    else:
        assert phase != "interrupt"
    after = snapshot(trainer)
    assert any(not torch.equal(before[n], after[n]) for n in trainable)
    for name in set(before) - trainable:
        torch.testing.assert_close(before[name], after[name], atol=0, rtol=0)
    if cfg.lora:
        for marker in (["lora_", "ln_f"] if case == "hybrid_zero3" else ["lora_"]):
            assert any(
                not torch.equal(before[n], after[n]) for n in trainable if marker in n
            )
    token_rows = assert_token_rows(token_before, effective_token_rows(trainer)) if tokens else {}
    digest = hashlib.sha256(
        b"".join(
            p.contiguous().view(torch.uint8).numpy().tobytes() for p in after.values()
        )
    ).hexdigest()
    digests = [None, None]
    torch.distributed.all_gather_object(digests, digest)
    assert digests[0] == digests[1]
    artifact_probe = {}
    if phase != "interrupt":
        assert (
            trainer.state.global_step == 6
            and trainer.state.epoch == 3
            and trainer.state.batch_in_epoch == 0
        )
        assert trainer.scheduler.state_dict()["last_epoch"] == 12
        assert all(group["lr"] == 0 for group in trainer.optimizer.param_groups)
        save(root / f"{phase}-params-{rank}.pt", after)
        trainer.model.eval()
        fixed = torch.tensor([[1, 3, 32 if tokens else 15, 2]], device=runtime.accelerator.device)
        with torch.no_grad(), runtime.accelerator.autocast():
            logits = trainer.model(input_ids=fixed).logits.float().cpu()
        save(root / f"{phase}-logits-{rank}.pt", logits)
        assert torch.isfinite(logits).all()
        if cfg.lora or tokens:
            artifact_probe = save_adapter_probe(trainer)
        else:
            trainer.export_model()
    return dict(
        state=asdict(trainer.state),
        seen=seen,
        evaluation=trainer.evaluation_history,
        digest=digest,
        world_size=2,
        rank=rank,
        local_rank=runtime.local_process_index,
        backend=str(torch.distributed.get_backend()),
        identities=identities,
        optimizer_coverage=coverage,
        shards=shards,
        gradient_checkpointing=cfg.gradient_checkpointing.enabled,
        token_rows=token_rows,
        artifact_probe=artifact_probe,
    )


def verify(case, root):
    if case.startswith("failure_"):
        ranks = (1,) if case == "failure_rank" else range(2)
        records = [
            json.loads((root / f"baseline-rank-{rank}.json").read_text())
            for rank in ranks
        ]
        assert all(r["expected_failure"] and r["ok"] for r in records)
        return dict(failure_propagated=True)
    tokens = case.startswith("tt_")
    adapter_case = tokens or case.startswith("lora_") or case == "hybrid_zero3"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fixed = torch.tensor([[1, 3, 32 if tokens else 15, 2]], device=device)
    reference = load(root / "resume-logits-0.pt")

    def reload_logits(model):
        model.to(device).eval()
        with torch.no_grad(), (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device == "cuda"
            else nullcontext()
        ):
            logits = model(fixed).logits.float().cpu()
        torch.testing.assert_close(
            logits,
            reference,
            atol=0.005 if device == "cuda" else 1e-6,
            rtol=0.05 if device == "cuda" else 1e-5,
        )
        return logits

    for rank in range(2):
        a = load(root / f"baseline-params-{rank}.pt")
        b = load(root / f"resume-params-{rank}.pt")
        for name in a:
            torch.testing.assert_close(a[name], b[name], atol=0, rtol=0)
        torch.testing.assert_close(
            load(root / f"baseline-logits-{rank}.pt"),
            load(root / f"resume-logits-{rank}.pt"),
            atol=0,
            rtol=0,
        )
        baseline = json.loads((root / f"baseline-rank-{rank}.json").read_text())
        interrupted = json.loads((root / f"interrupt-rank-{rank}.json").read_text())
        resumed = json.loads((root / f"resume-rank-{rank}.json").read_text())
        assert baseline["seen"] == interrupted["seen"] + resumed["seen"]
        assert len(baseline["seen"]) == 18
        assert all(
            e["eval_examples"] == 9 and e["eval_tokens"] == 23
            for e in baseline["evaluation"]
        )
        expected = baseline["evaluation"]
        actual = interrupted["evaluation"] + resumed["evaluation"]
        assert expected == actual
        for record in (baseline, interrupted, resumed):
            coverage = record["optimizer_coverage"]
            assert coverage["expected"] == coverage["actual"] > 0
            assert coverage["duplicates"] == 0
            if tokens:
                assert record["token_rows"]["stable_old_rows"]
                assert record["token_rows"]["changed_selected_rows"] == 2
            if case.endswith("_gc"):
                assert record["gradient_checkpointing"]
            if "zero3" in case:
                assert any(p["partition"] < p["full"] for p in record["shards"])
        if adapter_case:
            for record in (baseline, resumed):
                assert record["artifact_probe"]["gathered_parameters"] > 0
                assert record["artifact_probe"]["save_pretrained_calls"] == int(rank == 0)
    for epoch in range(3):
        all_ids = []
        for rank in range(2):
            all_ids += json.loads((root / f"baseline-rank-{rank}.json").read_text())[
                "seen"
            ][epoch * 6 : (epoch + 1) * 6]
        assert len(all_ids) == 12 and set(all_ids) == set(range(3, 12))
    adapter_reload_error = None
    if adapter_case:
        adapter = root / "resumed-run/adapter"
        if case == "hybrid_zero3":
            assert (adapter / "base_trainable_state.safetensors").is_file()
        else:
            from peft import PeftModel

            adapted = PeftModel.from_pretrained(
                AutoModelForCausalLM.from_pretrained(root / "base"), adapter
            )
            adapter_logits = reload_logits(adapted)
            adapter_reload_error = float((adapter_logits - reference).abs().max())
            del adapted
        merge = merge_trainable_tokens if tokens else merge_lora
        path = merge(root / "base", adapter, root / "merged")
    else:
        path = root / "resumed-run/final-model"
    model = AutoModelForCausalLM.from_pretrained(path).eval()
    logits = reload_logits(model)
    if tokens:
        original = AutoModelForCausalLM.from_pretrained(root / "base")
        assert_token_rows(
            original.get_input_embeddings().weight.detach().cpu(),
            model.get_input_embeddings().weight.detach().cpu(),
        )
    # Independent one-process token NLL over the complete, unpadded dataset.
    total = 0.0
    supervised_tokens = 0
    with torch.no_grad(), (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else nullcontext()
    ):
        for row in Rows(tokens):
            batch = collate([row])
            labels = shift_causal_labels(batch["labels"]).to(device)
            values = model(batch["input_ids"].to(device)).logits.float()
            total += F.cross_entropy(
                values.flatten(0, 1),
                labels.flatten(),
                ignore_index=-100,
                reduction="sum",
            ).item()
            supervised_tokens += int((labels != -100).sum())
    expected = json.loads((root / "baseline-rank-0.json").read_text())["evaluation"][
        -1
    ]["eval_loss"]
    assert abs(total / supervised_tokens - expected) < (0.005 if device == "cuda" else 1e-6)
    return dict(
        resume_exact=True,
        all_examples_covered=True,
        padded_examples_per_epoch=3,
        eval_examples=9,
        eval_tokens=supervised_tokens,
        eval_reference_error=abs(total / supervised_tokens - expected),
        reload_max_error=float((logits - reference).abs().max()),
        adapter_reload_max_error=adapter_reload_error,
        token_rows_verified=tokens,
    )


def main():
    case, phase, directory = sys.argv[1:]
    root = Path(directory) / case
    root.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", 0))
    result = dict(case=case, phase=phase, rank=rank, ok=False)
    code = 0
    try:
        if os.environ.get("HARDENING_GPU") == "1":
            assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        if phase == "prepare":
            result.update(prepare(case, root))
        elif phase == "verify":
            result.update(verify(case, root))
        else:
            result.update(train(case, phase, root))
        result["ok"] = True
    except Exception as exc:
        result.update(error=repr(exc), traceback=traceback.format_exc())
        traceback.print_exc()
        code = 1
        if (
            case.startswith("failure_")
            and phase == "baseline"
            and not isinstance(exc, AssertionError)
        ):
            needles = {
                "failure_materialize": "intentional materialization",
                "failure_checkpoint": "checkpoint file",
                "failure_masked": "Non-finite loss",
            }
            peer_error = case == "failure_rank" and (
                (rank == 1 and "intentional rank failure" in str(exc))
                or (rank == 0 and "monitoredbarrier" in str(exc).lower())
            )
            if peer_error or needles.get(case, "__not_expected__") in str(exc):
                result.update(ok=True, expected_failure=True)
    finally:
        name = (
            f"{phase}.json"
            if phase in ("prepare", "verify")
            else f"{phase}-rank-{rank}.json"
        )
        write(root / name, result)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return code


if __name__ == "__main__":
    sys.exit(main())
