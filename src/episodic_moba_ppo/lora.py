"""Hugging Face PEFT LoRA injection and trainable-set validation."""

from __future__ import annotations

from collections.abc import Iterator

from peft import LoraConfig, inject_adapter_in_model
from torch import nn


PEFT_ADAPTER_NAME = "default"
PEFT_TARGET_MODULES = ("query_lora", "key_lora", "value_lora", "output_lora")

# These are the complete actor and critic heads on the checkout's
# ``ActorCriticModel``.  They are deliberately named as *modules*, rather than
# individual tensors, so a policy with multiple action branches remains fully
# trainable and a later head bias cannot accidentally be left frozen.
POLICY_HEAD_MODULES = ("lin_policy", "policy_branches")
VALUE_HEAD_MODULES = ("lin_value", "value")
TRAINABLE_HEAD_MODULES = POLICY_HEAD_MODULES + VALUE_HEAD_MODULES


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


def _head_module_names(module: nn.Module) -> tuple[str, ...]:
    """Return the complete actor/critic head set when ``module`` has one.

    A bare ``Transformer`` is intentionally supported by the low-level LoRA
    tests and has none of these modules.  A partially matching actor/critic is
    a wiring error: silently training just one side would violate the matched
    arm contract.
    """

    direct_children = set(module._modules)
    present = tuple(
        name for name in TRAINABLE_HEAD_MODULES if name in direct_children
    )
    if present and present != TRAINABLE_HEAD_MODULES:
        missing = sorted(set(TRAINABLE_HEAD_MODULES) - set(present))
        raise AssertionError(
            "actor/critic model is missing required trainable head modules: "
            f"{missing}"
        )
    return present


def iter_trainable_head_parameters(
    module: nn.Module,
) -> Iterator[tuple[str, nn.Parameter]]:
    """Yield every tensor in the complete policy and value heads.

    The names are rooted at the supplied model.  In particular, a
    ``ModuleList`` policy head yields every branch, including each branch bias.
    """

    for module_name in _head_module_names(module):
        head = module.get_submodule(module_name)
        for name, parameter in head.named_parameters(prefix=module_name):
            yield name, parameter


def _allowed_trainable_parameters(module: nn.Module) -> dict[int, str]:
    """Map the identity of each explicitly allowed trainable tensor to its name."""

    allowed: dict[int, str] = {}
    for name, parameter in iter_lora_parameters(module):
        allowed[id(parameter)] = name
    for name, parameter in iter_trainable_head_parameters(module):
        previous = allowed.setdefault(id(parameter), name)
        if previous != name:
            raise AssertionError(
                "a LoRA adapter and actor/critic head unexpectedly share a parameter"
            )
    return allowed


def freeze_for_lora(module: nn.Module, *, expected_count: int | None = None) -> list[str]:
    """Train Q/K/V/O adapters plus the complete policy and value heads.

    All other checkpoint/base parameters are frozen.  ``expected_count`` is a
    total over adapters *and* heads; callers should derive it from the actual
    action space instead of assuming a Mortar-specific constant.
    """
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    allowed = _allowed_trainable_parameters(module)
    trainable_names: list[str] = []
    for name, parameter in module.named_parameters():
        if id(parameter) not in allowed:
            continue
        parameter.requires_grad_(True)
        trainable_names.append(name)
    assert_lora_trainable_set(module, expected_count=expected_count)
    return trainable_names


def assert_lora_trainable_set(
    module: nn.Module, *, expected_count: int | None = None
) -> int:
    """Fail unless exactly adapters and complete actor/critic heads train.

    The historical function name is retained for the public interface, even
    though MiniGrid fine-tuning intentionally includes the non-adapter heads.
    """

    allowed = _allowed_trainable_parameters(module)
    trainable = [
        (name, parameter)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]
    trainable_ids = {id(parameter) for _, parameter in trainable}
    unexpected = [name for name, parameter in trainable if id(parameter) not in allowed]
    if unexpected:
        raise AssertionError(f"unexpected parameters require gradients: {unexpected}")
    missing = [
        name for parameter_id, name in allowed.items() if parameter_id not in trainable_ids
    ]
    if missing:
        raise AssertionError(f"required LoRA/head parameters are frozen: {missing}")
    count = sum(parameter.numel() for _, parameter in trainable)
    if expected_count is not None and count != expected_count:
        raise AssertionError(
            f"expected {expected_count} trainable LoRA/head parameters, found {count}"
        )
    return count
