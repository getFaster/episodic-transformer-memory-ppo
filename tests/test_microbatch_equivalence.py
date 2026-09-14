import copy

import pytest
import torch

from episodic_moba_ppo.ppo import (
    accumulate_effective_minibatch,
    full_minibatch_advantage_stats,
    linear_learning_rate,
)


def test_eight_microbatches_match_one_effective_minibatch_gradient() -> None:
    generator = torch.Generator().manual_seed(19)
    inputs = torch.randn(2048, 5, generator=generator)
    targets = torch.randn(2048, generator=generator)
    advantages = torch.randn(2048, generator=generator)
    full_model = torch.nn.Linear(5, 1, bias=False)
    micro_model = copy.deepcopy(full_model)

    mean, std = full_minibatch_advantage_stats(advantages)
    normalized = (advantages - mean) / (std + 1e-8)
    full_loss = ((full_model(inputs).squeeze(1) - targets).square() * normalized).mean()
    full_loss.backward()

    optimizer = torch.optim.SGD(micro_model.parameters(), lr=0.0)

    def loss_fn(batch, normalized_advantages):
        prediction = micro_model(batch["inputs"]).squeeze(1)
        return ((prediction - batch["targets"]).square() * normalized_advantages).mean()

    result = accumulate_effective_minibatch(
        model=micro_model,
        optimizer=optimizer,
        batch={"inputs": inputs, "targets": targets, "advantages": advantages},
        loss_fn=loss_fn,
    )
    assert result.microbatches == 8
    assert result.samples == 2048
    torch.testing.assert_close(
        micro_model.weight.grad, full_model.weight.grad, rtol=2e-5, atol=2e-6
    )


def test_accumulation_requires_original_effective_size() -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    with pytest.raises(ValueError, match="expected 2048"):
        accumulate_effective_minibatch(
            model=model,
            optimizer=optimizer,
            batch={"x": torch.ones(256, 2), "advantages": torch.ones(256)},
            loss_fn=lambda batch, advantages: model(batch["x"]).mean(),
        )


def test_linear_lr_has_locked_endpoints() -> None:
    assert linear_learning_rate(0) == pytest.approx(0.02)
    assert linear_learning_rate(61) == pytest.approx(0.00067)
    assert linear_learning_rate(30) > linear_learning_rate(31)
