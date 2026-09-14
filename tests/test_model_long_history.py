import torch

from episodic_moba_ppo.episodic_memory import EpisodeTrace
from episodic_moba_ppo.lora import freeze_for_lora
from model import ActorCriticModel


class _VectorObservationSpace:
    shape = (12,)


def _model():
    config = {
        "hidden_layer_size": 384,
        "transformer": {
            "num_blocks": 3,
            "embed_dim": 384,
            "num_heads": 4,
            "positional_encoding": "relative",
            "layer_norm": "pre",
            "gtrxl": False,
        },
    }
    return ActorCriticModel(config, _VectorObservationSpace(), (4,), 512)


def _attention_config():
    return {
        "budget": 256,
        "dense_recent": 128,
        "search_horizon": 2560,
        "retrieval": {"block_size": 16, "retrieved_blocks": 8},
    }


def test_model_long_history_returns_actor_critic_outputs_and_routing():
    torch.manual_seed(17)
    model = _model()
    trace = EpisodeTrace(trace_id=0, num_layers=3, width=384)
    for timestep in range(300):
        trace.append(torch.randn(3, 384), timestep=timestep)
    context = trace.context(
        300, dense_recent=128, search_horizon=2560, block_size=16
    )

    policy, value, memories, routing = model.forward_long_history(
        torch.randn(1, 12),
        [context],
        arm="trxl_moba",
        attention_config=_attention_config(),
    )

    assert len(policy) == 1
    assert policy[0].logits.shape == (1, 4)
    assert value.shape == (1,)
    assert memories.shape == (1, 3, 384)
    assert not memories.requires_grad
    assert len(routing) == 3
    assert all(layer[0] is not None for layer in routing)


def test_model_enable_lora_supports_exact_full_model_freezing_contract():
    model = _model()
    original_keys = set(model.state_dict())
    model.enable_lora(rank=8, alpha=16, dropout=0.0)
    trainable = freeze_for_lora(model, expected_count=73_728)

    assert len(trainable) == 24
    assert all("_lora.lora_" in name for name in trainable)
    assert not any("lora" in name for name in original_keys)
    assert all(
        parameter.requires_grad == ("_lora.lora_" in name)
        for name, parameter in model.named_parameters()
    )


def test_model_long_history_accepts_validated_attention_models():
    from episodic_moba_ppo.config import AttentionConfig

    attention_data = _attention_config()
    attention_data["retrieval"].update(
        {
            "enabled": True,
            "retrieved_tokens": 128,
            "score_reduction": "max_head",
            "tie_break": "chronological_block_index",
        }
    )
    attention = AttentionConfig.model_validate(attention_data)
    model = _model()
    trace = EpisodeTrace(trace_id=0, num_layers=3, width=384)
    for timestep in range(8):
        trace.append(torch.randn(3, 384), timestep=timestep)
    context = trace.context(
        8, dense_recent=128, search_horizon=2560, block_size=16
    )

    _, _, memories, routing = model.forward_long_history(
        torch.randn(1, 12),
        [context],
        arm="trxl_moba",
        attention_config=attention,
    )
    assert memories.shape == (1, 3, 384)
    assert all(layer[0].context_timesteps.tolist() == list(range(8)) for layer in routing)
