"""Distributed token-weighted evaluation without changing training RNG or loaders."""

from abc import ABC, abstractmethod
from contextlib import contextmanager
import math
import random

from accelerate.utils import send_to_device
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .distributed import raise_if_any_rank_failed
from .loss import IGNORE_INDEX, shift_causal_labels


class EvaluationMetric(ABC):
    """Additive per-example statistics, reduced globally before computing metrics.

    ``statistics`` returns a [batch, statistics] tensor and must ignore labels
    equal to -100. Padding examples are removed by the evaluator. ``compute``
    receives the summed vector on CPU and returns uniquely named scalar metrics.
    Instances should be stateless; never average rank-local metric values.
    """

    @abstractmethod
    def statistics(self, logits, labels):
        pass

    @abstractmethod
    def compute(self, totals):
        pass


@contextmanager
def preserve_rng(device):
    python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    mps = torch.mps.get_rng_state() if device.type == "mps" else None
    try:
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        if mps is not None:
            torch.mps.set_rng_state(mps)


def evaluate_model(
    model, accelerator, dataset, collate_fn, *, batch_size, seed, metrics=()
):
    """Every rank forwards equal-sized batches; only real examples contribute.

    Default loss is unsmoothed causal token NLL, independent of the training
    objective. Perplexity is exp(NLL). No sampling or generation is performed.
    """
    n = len(dataset)
    raise_if_any_rank_failed(
        ValueError("Evaluation dataset is empty") if n == 0 else None, "Evaluation"
    )
    world, rank = accelerator.num_processes, accelerator.process_index
    if torch.distributed.is_initialized():
        settings = [None] * world
        torch.distributed.all_gather_object(settings, (n, batch_size, len(metrics)))
        if len(set(settings)) != 1:
            raise ValueError(
                "Evaluation dataset length, batch size and metric count must agree across ranks"
            )
    count = math.ceil(n / (world * batch_size)) * world * batch_size
    indices = list(range(rank, count, world))
    device = accelerator.device
    stats_device = torch.device("cpu") if device.type == "mps" else device
    modes = [(module, module.training) for module in model.modules()]
    totals = torch.zeros(
        3, dtype=torch.float64, device=stats_device
    )  # NLL, tokens, examples
    metric_totals = [None] * len(metrics)

    # This is deliberately not an Accelerate-prepared loader: it must not alter
    # the active training loader's accumulation/end-of-dataloader bookkeeping.
    loader = DataLoader(
        indices,
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(seed),
    )
    with preserve_rng(device):
        model.eval()
        try:
            with torch.no_grad():
                for batch_indices in loader:
                    error, batch = None, None
                    try:
                        batch = collate_fn([dataset[int(i) % n] for i in batch_indices])
                        batch = send_to_device(batch, device)
                        if "labels" not in batch:
                            raise KeyError("Evaluation batch must contain labels")
                    except Exception as exc:
                        error = exc
                    raise_if_any_rank_failed(error, "Evaluation batch")
                    labels = shift_causal_labels(batch["labels"])
                    logits = model(
                        **{k: v for k, v in batch.items() if k != "labels"}
                    ).logits
                    valid = (batch_indices < n).to(device)
                    error = None
                    try:
                        losses = F.cross_entropy(
                            logits.float().reshape(-1, logits.shape[-1]),
                            labels.reshape(-1),
                            reduction="none",
                            ignore_index=IGNORE_INDEX,
                        ).reshape(labels.shape)
                        mask = (labels != IGNORE_INDEX) & valid.unsqueeze(-1)
                        value = (
                            losses.masked_fill(~mask, 0)
                            .to(stats_device, dtype=torch.float64)
                            .sum()
                        )
                        if not torch.isfinite(value):
                            raise FloatingPointError("Non-finite evaluation loss")
                        totals += torch.stack(
                            [
                                value,
                                mask.sum().to(stats_device, dtype=torch.float64),
                                valid.sum().to(stats_device, dtype=torch.float64),
                            ]
                        )
                        for i, metric in enumerate(metrics):
                            stats = metric.statistics(logits, labels).to(
                                device=stats_device, dtype=torch.float64
                            )
                            if stats.ndim != 2 or stats.shape[0] != len(valid):
                                raise ValueError(
                                    "Metric statistics must have shape [batch, statistics]"
                                )
                            stats = stats.masked_fill(
                                ~valid.to(stats_device).unsqueeze(-1), 0
                            ).sum(0)
                            if not torch.isfinite(stats).all():
                                raise ValueError("Metric statistics must be finite")
                            if metric_totals[i] is None:
                                metric_totals[i] = torch.zeros_like(stats)
                            metric_totals[i] += stats
                    except Exception as exc:
                        error = exc
                    raise_if_any_rank_failed(error, "Evaluation statistics")
            if torch.distributed.is_initialized():
                widths = [None] * world
                torch.distributed.all_gather_object(
                    widths, tuple(s.numel() for s in metric_totals)
                )
                if len(set(widths)) != 1:
                    raise ValueError(
                        "Evaluation metric statistic widths must agree across ranks"
                    )
            totals = accelerator.reduce(totals, reduction="sum").cpu()
            if totals[1] == 0:
                raise ValueError("Evaluation has no supervised tokens")
            loss = (totals[0] / totals[1]).item()
            result = dict(
                eval_loss=loss,
                eval_perplexity=math.exp(loss) if loss < 709 else float("inf"),
                eval_tokens=int(totals[1]),
                eval_examples=int(totals[2]),
            )
            for metric, stats in zip(metrics, metric_totals):
                global_stats = accelerator.reduce(stats, reduction="sum").cpu()
                error = None
                try:
                    values = metric.compute(global_stats)
                    for name, value in values.items():
                        if name in result:
                            raise ValueError(f"Duplicate evaluation metric: {name}")
                        if not math.isfinite(float(value)):
                            raise ValueError(f"Non-finite metric: {name}")
                        result[name] = float(value)
                except Exception as exc:
                    error = exc
                raise_if_any_rank_failed(error, "Evaluation metric computation")
            return result
        finally:
            for module, training in modes:
                module.training = training
