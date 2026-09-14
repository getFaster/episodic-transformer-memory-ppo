"""PPO optimizer helpers with exact effective-minibatch accumulation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

EFFECTIVE_MINIBATCH_SIZE = 2_048
MICROBATCH_SIZE = 256


@dataclass(frozen=True)
class AccumulationResult:
    loss: float
    microbatches: int
    samples: int
    gradient_norm: float | None


def linear_learning_rate(
    update: int,
    *,
    total_updates: int = 62,
    initial: float = 0.02,
    final: float = 0.00067,
) -> float:
    """Return the update-indexed LR, including both endpoints over 62 updates."""

    if total_updates < 2:
        raise ValueError("total_updates must be at least two")
    if update < 0 or update >= total_updates:
        raise ValueError(f"update must be in [0, {total_updates})")
    fraction = update / (total_updates - 1)
    return initial + (final - initial) * fraction


def full_minibatch_advantage_stats(
    advantages: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = advantages.to(dtype=torch.float32)
    if values.numel() < 2:
        raise ValueError("advantage normalization requires at least two samples")
    return values.mean(), values.std(correction=1)


def _batch_size(batch: Mapping[str, Any]) -> int:
    sizes = {int(value.shape[0]) for value in batch.values() if torch.is_tensor(value)}
    if len(sizes) != 1:
        raise ValueError("all tensor minibatch fields must share the first dimension")
    if not sizes:
        raise ValueError("minibatch must contain tensor fields")
    return sizes.pop()


def _slice_float32(batch: Mapping[str, Any], start: int, end: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            item = value[start:end]
            if item.is_floating_point():
                item = item.to(dtype=torch.float32)
            result[key] = item
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            result[key] = value[start:end]
        else:
            result[key] = value
    return result


def accumulate_effective_minibatch(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, Any],
    loss_fn: Callable[[Mapping[str, Any], torch.Tensor], torch.Tensor],
    microbatch_size: int = MICROBATCH_SIZE,
    expected_size: int = EFFECTIVE_MINIBATCH_SIZE,
    max_grad_norm: float | None = None,
) -> AccumulationResult:
    """Accumulate mean losses and take exactly one optimizer step.

    ``loss_fn`` receives one float32 microbatch and advantages normalized using
    statistics from the complete effective minibatch.  It must return the mean
    loss for that microbatch.  A fresh call per slice intentionally recomputes
    model routing under the current (unchanged) parameters.
    """

    size = _batch_size(batch)
    if size != expected_size:
        raise ValueError(f"expected {expected_size} samples, got {size}")
    if microbatch_size < 1 or size % microbatch_size:
        raise ValueError("microbatch_size must evenly divide the effective minibatch")
    advantages = batch.get("advantages")
    if not torch.is_tensor(advantages):
        raise ValueError("batch must contain tensor advantages")
    mean, std = full_minibatch_advantage_stats(advantages)

    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    count = 0
    for start in range(0, size, microbatch_size):
        end = start + microbatch_size
        microbatch = _slice_float32(batch, start, end)
        normalized = (microbatch["advantages"] - mean) / (std + 1e-8)
        loss = loss_fn(microbatch, normalized)
        if loss.ndim != 0:
            raise ValueError("loss_fn must return a scalar mean loss")
        weight = (end - start) / size
        (loss * weight).backward()
        total_loss += float(loss.detach()) * weight
        count += 1

    grad_norm: float | None = None
    if max_grad_norm is not None:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        grad_norm = float(norm)
    optimizer.step()
    return AccumulationResult(total_loss, count, size, grad_norm)


def is_lora_parameter_name(name: str) -> bool:
    parts = name.lower().split(".")
    if any(part in {"lora_a", "lora_b"} for part in parts):
        return True
    return parts[-1] in {"a", "b"} and any(
        part in {"lora", "adapter"} for part in parts[:-1]
    )


def create_muon_optimizer(
    named_parameters: Mapping[str, torch.nn.Parameter]
    | Sequence[tuple[str, torch.nn.Parameter]],
    *,
    lr: float = 0.02,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    weight_decay: float = 0.01,
    adjust_lr_fn: str = "original",
    expected_trainable_count: int | None = 73_728,
) -> torch.optim.Optimizer:
    """Construct native PyTorch Muon over LoRA matrices, failing closed."""

    source = (
        named_parameters.items()
        if isinstance(named_parameters, Mapping)
        else named_parameters
    )
    items = list(source)
    trainable = [
        (name, parameter) for name, parameter in items if parameter.requires_grad
    ]
    unexpected = [name for name, _ in trainable if not is_lora_parameter_name(name)]
    if unexpected:
        raise ValueError(f"non-LoRA trainable parameters: {unexpected}")
    if not trainable:
        raise ValueError("no trainable LoRA parameters found")
    non_matrix = [name for name, parameter in trainable if parameter.ndim != 2]
    if non_matrix:
        raise ValueError(f"Muon requires 2-D LoRA matrices: {non_matrix}")
    count = sum(parameter.numel() for _, parameter in trainable)
    if expected_trainable_count is not None and count != expected_trainable_count:
        raise ValueError(
            f"expected {expected_trainable_count} trainable LoRA parameters, "
            f"got {count}"
        )
    return torch.optim.Muon(
        [parameter for _, parameter in trainable],
        lr=lr,
        momentum=momentum,
        nesterov=nesterov,
        ns_steps=ns_steps,
        weight_decay=weight_decay,
        adjust_lr_fn=adjust_lr_fn,
    )
