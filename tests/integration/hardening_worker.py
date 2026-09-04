"""Fresh-process CPU/CUDA distributed boundaries, evaluation and artifact checks."""

from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpu"))
import fresh_case as helpers
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel
from llm_training import (
    CheckpointConfig,
    DeepSpeedConfig,
    DistributedRuntime,
    EvaluationConfig,
    FreezeConfig,
    LoRAConfig,
    OptimizerConfig,
    SchedulerConfig,
    Trainer,
    TrainingConfig,
    merge_lora,
)
from llm_training.loss import shift_causal_labels


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


class Rows(torch.utils.data.Dataset):
    def __len__(self):
        return 9

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        ids = torch.tensor([1, index + 3, index + 15, 2])
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
            helpers.save(self.root / f"checkpoint-rng-{rank}.pt", helpers.rng())
            helpers.save(
                self.root / f"checkpoint-params-{rank}.pt", helpers.snapshot(self)
            )
            raise Stop()
        return path


def prepare(root):
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
            if case == "hybrid_zero3"
            else None
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
    if case == "failure_rank":
        torch.distributed.barrier()
        if rank == 1:
            write(
                root / f"{phase}-rank-{rank}.json",
                dict(ok=True, expected_failure=True, error="intentional rank exit 23"),
            )
            os._exit(23)
        torch.distributed.all_reduce(torch.ones(1, device=runtime.accelerator.device))
        raise AssertionError("Peer failure was not propagated")
    if case == "failure_materialize":

        def bad(path):
            raise ValueError("intentional materialization failure")

        runtime.materialize_path(root / "materialized.txt", bad, scope="global")
        raise AssertionError("Materialization unexpectedly succeeded")
    with runtime.model_loading_context():
        model = AutoModelForCausalLM.from_pretrained(root / "base")
    seen = []

    def track(module, args, kwargs):
        if module.training:
            seen.extend(kwargs["input_ids"][:, 1].detach().cpu().tolist())

    model.register_forward_pre_hook(track, with_kwargs=True)
    trainer = Probe(
        model=model,
        train_dataset=Rows(),
        eval_dataset=Rows(),
        collate_fn=helpers.collate,
        config=cfg,
        runtime=runtime,
    )
    trainer.root = root
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
    if cfg.lora:
        assert any("lora_" in name for name in trainable) and any(
            "ln_f" in name for name in trainable
        )
    if ds:
        assert any(
            p.ds_tensor.numel() < p.ds_numel
            for p in unwrapped.parameters()
            if hasattr(p, "ds_tensor")
        )
    stop_step = 2 if case in ("boundary_epoch_ddp", "full_zero3") else 3
    if phase == "resume":
        assert trainer.state.global_step == stop_step
        helpers.assert_rng(
            helpers.rng(), helpers.load(root / f"checkpoint-rng-{rank}.pt")
        )
        before = helpers.snapshot(trainer)
        saved = helpers.load(root / f"checkpoint-params-{rank}.pt")
        for name in before:
            torch.testing.assert_close(before[name], saved[name], atol=0, rtol=0)
    before = helpers.snapshot(trainer)
    if phase == "interrupt":
        trainer.stop_step = stop_step
    try:
        trainer.train()
    except Stop:
        assert phase == "interrupt"
    else:
        assert phase != "interrupt"
    after = helpers.snapshot(trainer)
    assert any(not torch.equal(before[n], after[n]) for n in trainable)
    for name in set(before) - trainable:
        torch.testing.assert_close(before[name], after[name], atol=0, rtol=0)
    if cfg.lora:
        for marker in ["lora_", "ln_f"]:
            assert any(
                not torch.equal(before[n], after[n]) for n in trainable if marker in n
            )
    digest = hashlib.sha256(
        b"".join(
            p.contiguous().view(torch.uint8).numpy().tobytes() for p in after.values()
        )
    ).hexdigest()
    digests = [None, None]
    torch.distributed.all_gather_object(digests, digest)
    assert digests[0] == digests[1]
    if phase != "interrupt":
        assert (
            trainer.state.global_step == 6
            and trainer.state.epoch == 3
            and trainer.state.batch_in_epoch == 0
        )
        assert trainer.scheduler.state_dict()["last_epoch"] == 12
        assert all(group["lr"] == 0 for group in trainer.optimizer.param_groups)
        helpers.save(root / f"{phase}-params-{rank}.pt", after)
        trainer.model.eval()
        fixed = torch.tensor([[1, 3, 15, 2]], device=runtime.accelerator.device)
        with torch.no_grad(), runtime.accelerator.autocast():
            logits = trainer.model(input_ids=fixed).logits.float().cpu()
        helpers.save(root / f"{phase}-logits-{rank}.pt", logits)
        if cfg.lora:
            trainer.save_adapter()
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
    )


def verify(case, root):
    if case.startswith("failure_"):
        records = [
            json.loads((root / f"baseline-rank-{rank}.json").read_text())
            for rank in range(2)
        ]
        assert all(r["expected_failure"] and r["ok"] for r in records)
        return dict(failure_propagated=True)
    for rank in range(2):
        a = helpers.load(root / f"baseline-params-{rank}.pt")
        b = helpers.load(root / f"resume-params-{rank}.pt")
        for name in a:
            torch.testing.assert_close(a[name], b[name], atol=0, rtol=0)
        torch.testing.assert_close(
            helpers.load(root / f"baseline-logits-{rank}.pt"),
            helpers.load(root / f"resume-logits-{rank}.pt"),
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
    for epoch in range(3):
        all_ids = []
        for rank in range(2):
            all_ids += json.loads((root / f"baseline-rank-{rank}.json").read_text())[
                "seen"
            ][epoch * 6 : (epoch + 1) * 6]
        assert len(all_ids) == 12 and set(all_ids) == set(range(3, 12))
    if case == "hybrid_zero3":
        adapter = root / "resumed-run/adapter"
        assert (adapter / "base_trainable_state.safetensors").is_file()
        path = merge_lora(root / "base", adapter, root / "merged")
    else:
        path = root / "resumed-run/final-model"
    model = AutoModelForCausalLM.from_pretrained(path).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    with torch.no_grad(), (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else nullcontext()
    ):
        logits = (
            model(torch.tensor([[1, 3, 15, 2]], device=device)).logits.float().cpu()
        )
    reference = helpers.load(root / "resume-logits-0.pt")
    torch.testing.assert_close(
        logits,
        reference,
        atol=0.005 if device == "cuda" else 1e-6,
        rtol=0.05 if device == "cuda" else 1e-5,
    )
    # Independent one-process token NLL over the complete, unpadded dataset.
    total = 0.0
    tokens = 0
    with torch.no_grad(), (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else nullcontext()
    ):
        for row in Rows():
            batch = helpers.collate([row])
            labels = shift_causal_labels(batch["labels"]).to(device)
            values = model(batch["input_ids"].to(device)).logits.float()
            total += F.cross_entropy(
                values.flatten(0, 1),
                labels.flatten(),
                ignore_index=-100,
                reduction="sum",
            ).item()
            tokens += int((labels != -100).sum())
    expected = json.loads((root / "baseline-rank-0.json").read_text())["evaluation"][
        -1
    ]["eval_loss"]
    assert abs(total / tokens - expected) < (0.005 if device == "cuda" else 1e-6)
    return dict(
        resume_exact=True,
        all_examples_covered=True,
        padded_examples_per_epoch=3,
        eval_examples=9,
        eval_tokens=tokens,
        eval_reference_error=abs(total / tokens - expected),
        reload_max_error=float((logits - reference).abs().max()),
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
            result.update(prepare(root))
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
            peer_error = case == "failure_rank" and any(
                s in str(exc).lower()
                for s in (
                    "connection closed",
                    "connection reset",
                    "remote process",
                    "nccl",
                    "peer",
                )
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
