import torch

from episodic_moba_ppo.lora import (
    TRAINABLE_HEAD_MODULES,
    assert_lora_trainable_set,
    freeze_for_lora,
)
from episodic_moba_ppo.ppo import MuonWithAdamWHeads, create_muon_optimizer
from model import ActorCriticModel
from transformer import Transformer


def _config():
    return {
        "num_blocks": 3,
        "embed_dim": 384,
        "num_heads": 4,
        "positional_encoding": "relative",
        "layer_norm": "pre",
        "gtrxl": False,
    }


def test_zero_effect_lora_and_exact_trainable_count():
    torch.manual_seed(4)
    transformer = Transformer(_config(), input_dim=384, max_episode_steps=512)
    attention = transformer.transformer_blocks[0].attention
    values = torch.randn(2, 9, 384)
    queries = torch.randn(2, 1, 384)
    mask = torch.ones(2, 9, dtype=torch.bool)
    expected, expected_weights = attention(values, values, queries, mask)

    transformer.enable_lora(rank=8, alpha=16, dropout=0.0)
    actual, actual_weights = attention(values, values, queries, mask)
    names = freeze_for_lora(transformer, expected_count=73_728)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)
    assert len(names) == 24
    assert assert_lora_trainable_set(transformer, expected_count=73_728) == 73_728
    assert all("_lora.lora_" in name for name in names)
    assert all(
        parameter.requires_grad == ("_lora.lora_" in name)
        for name, parameter in transformer.named_parameters()
    )


def test_full_width_delta_changes_logical_projection_only_after_b_updates():
    transformer = Transformer(_config(), input_dim=384, max_episode_steps=512)
    attention = transformer.transformer_blocks[0].attention
    inputs = torch.randn(1, 3, 384)
    base = attention.project_queries(inputs)
    transformer.enable_lora()
    zero_effect = attention.project_queries(inputs)
    torch.testing.assert_close(zero_effect, base, rtol=0, atol=0)

    with torch.no_grad():
        attention.query_lora.lora_B["default"].weight.fill_(0.01)
    adapted = attention.project_queries(inputs)
    assert adapted.shape == (1, 3, 4, 96)
    assert not torch.equal(adapted, base)


class _VectorObservationSpace:
    shape = (12,)


def _actor_critic() -> ActorCriticModel:
    return ActorCriticModel(
        {"hidden_layer_size": 384, "transformer": _config()},
        _VectorObservationSpace(),
        (4,),
        512,
    )


def test_freeze_for_lora_keeps_complete_policy_and_value_heads_trainable():
    model = _actor_critic()
    model.enable_lora(rank=8, alpha=16, dropout=0.0)
    trainable_names = freeze_for_lora(model)

    expected_head_names = {
        name
        for name, _ in model.named_parameters()
        if any(name.startswith(f"{head}.") for head in TRAINABLE_HEAD_MODULES)
    }
    actual_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    lora_names = {name for name in actual_names if "_lora.lora_" in name}

    assert actual_names == lora_names | expected_head_names
    assert set(trainable_names) == actual_names
    assert assert_lora_trainable_set(model) == sum(
        parameter.numel()
        for _, parameter in model.named_parameters()
        if parameter.requires_grad
    )


def test_lora_and_complete_heads_receive_gradients_and_optimizer_steps():
    torch.manual_seed(19)
    model = _actor_critic()
    model.enable_lora(rank=8, alpha=16, dropout=0.0)
    freeze_for_lora(model)
    # PEFT initializes B to zero, so A receives no signal on the first pass.
    # A tiny nonzero B makes this a meaningful all-allowed-parameters test.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "_lora.lora_B." in name:
                parameter.fill_(0.001)

    memory = torch.zeros((2, 3, 3, 384))
    memory_mask = torch.ones((2, 3), dtype=torch.bool)
    memory_indices = torch.zeros((2, 3), dtype=torch.long)
    policy, value, _ = model(
        torch.randn(2, 12), memory, memory_mask, memory_indices
    )
    loss = value.square().mean() + sum(
        branch.logits.square().mean() for branch in policy
    )
    loss.backward()

    allowed = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert all(model.get_parameter(name).grad is not None for name in allowed)
    assert all(
        model.get_parameter(name).grad is None
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    )

    optimizer = create_muon_optimizer(list(model.named_parameters()))
    assert isinstance(optimizer, MuonWithAdamWHeads)
    optimizer.step()
