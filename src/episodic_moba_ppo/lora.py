"""Hugging Face PEFT LoRA injection and trainable-set validation."""

from __future__ import annotations

from collections.abc import Iterator

from peft import LoraConfig, inject_adapter_in_model
from torch import nn


PEFT_ADAPTER_NAME = "default"
PEFT_TARGET_MODULES = ("query_lora", "key_lora", "value_lora", "output_lora")


def inject_full_width_lora(
    module: nn.Module,
    *,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> None:
    """Inject PEFT LoRA into pre-attached zero-base logical projections."""
    if rank <= 0:
        raise ValueError("LoRA rank must be positive")
    if dropout != 0.0:
        raise ValueError("this experiment requires zero LoRA dropout")
    targets = [
        name
        for name, child in module.named_modules()
        if name.rsplit(".", 1)[-1] in PEFT_TARGET_MODULES
        and isinstance(child, nn.Linear)
    ]
    if len(targets) != 12:
        raise AssertionError(f"expected 12 logical LoRA anchors, found {len(targets)}")
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(PEFT_TARGET_MODULES),
        bias="none",
        init_lora_weights=True,
    )
    inject_adapter_in_model(
        config,
        module,
        adapter_name=PEFT_ADAPTER_NAME,
        low_cpu_mem_usage=False,
    )


def iter_lora_parameters(module: nn.Module) -> Iterator[tuple[str, nn.Parameter]]:
    """Yield only Hugging Face PEFT LoRA A/B parameters."""
    for name, parameter in module.named_parameters():
        if ".lora_A." in name or ".lora_B." in name:
            yield name, parameter


def freeze_for_lora(module: nn.Module, *, expected_count: int | None = None) -> list[str]:
    """Freeze the base model, enable adapters, and validate the trainable set."""
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    trainable_names = []
    count = 0
    for name, parameter in iter_lora_parameters(module):
        parameter.requires_grad_(True)
        trainable_names.append(name)
        count += parameter.numel()
    assert_lora_trainable_set(module, expected_count=expected_count)
    return trainable_names


def assert_lora_trainable_set(
    module: nn.Module, *, expected_count: int | None = None
) -> int:
    """Fail if a base parameter is trainable or the adapter count is wrong."""
    lora_ids = {id(parameter) for _, parameter in iter_lora_parameters(module)}
    trainable = [(name, parameter) for name, parameter in module.named_parameters() if parameter.requires_grad]
    unexpected = [name for name, parameter in trainable if id(parameter) not in lora_ids]
    if unexpected:
        raise AssertionError(f"non-LoRA parameters require gradients: {unexpected}")
    count = sum(parameter.numel() for _, parameter in trainable)
    if expected_count is not None and count != expected_count:
        raise AssertionError(
            f"expected {expected_count} trainable LoRA parameters, found {count}"
        )
    return count
