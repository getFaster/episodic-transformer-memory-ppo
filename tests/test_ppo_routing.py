import json

import pytest
import torch

from episodic_moba_ppo.ppo import accumulate_effective_minibatch
from episodic_moba_ppo.runtime import BaselineGateError, require_baseline_gate


class CountingSGD(torch.optim.SGD):
    def __init__(self, params):
        super().__init__(params, lr=0.0)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


def test_route_dependent_forward_runs_for_every_microbatch_before_one_step() -> None:
    model = torch.nn.Linear(3, 1, bias=False)
    optimizer = CountingSGD(model.parameters())
    inputs = torch.arange(2048 * 3, dtype=torch.float32).reshape(2048, 3) / 1000
    advantages = torch.linspace(-1, 1, 2048)
    weights_seen = []
    route_calls = []

    def loss_fn(batch, normalized_advantages):
        # This callback stands at the model/retrieval seam: its invocation is the
        # route recomputation, and no selected route is present in the batch.
        route_calls.append(int(batch["query_timestep"][0]))
        weights_seen.append(model.weight.detach().clone())
        route_score = model(batch["inputs"]).squeeze(1)
        return (route_score * normalized_advantages).mean()

    result = accumulate_effective_minibatch(
        model=model,
        optimizer=optimizer,
        batch={
            "inputs": inputs,
            "query_timestep": torch.arange(2048),
            "advantages": advantages,
        },
        loss_fn=loss_fn,
    )
    assert result.microbatches == 8
    assert route_calls == list(range(0, 2048, 256))
    assert optimizer.step_calls == 1
    assert all(torch.equal(weights_seen[0], weight) for weight in weights_seen[1:])


def test_training_fails_closed_on_failed_baseline(tmp_path) -> None:
    gate = tmp_path / "baseline_reference.json"
    gate.write_text(json.dumps({"passed": False}), encoding="utf-8")
    with pytest.raises(BaselineGateError, match="did not pass"):
        require_baseline_gate(gate)


def test_training_fails_closed_on_spoofed_baseline_protocol(tmp_path) -> None:
    gate = tmp_path / "baseline_reference.json"
    gate.write_text(
        json.dumps(
            {
                "passed": True,
                "protocol": {"command_count": 10},
                "summary": {"success_rate": 1.0, "mean_normalized_return": 1.0},
                "thresholds": {
                    "success_rate": 0.1,
                    "mean_normalized_return": 0.1,
                },
                "episodes": [],
                "provenance": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(BaselineGateError, match="locked protocol"):
        require_baseline_gate(gate)
