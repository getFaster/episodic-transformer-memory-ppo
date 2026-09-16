import pytest
import torch

from episodic_moba_ppo.episodic_memory import EpisodeTrace
from episodic_moba_ppo.moba_retrieval import select_moba_context
from transformer import Transformer


def _heads(inputs):
    return inputs.reshape(inputs.shape[0], inputs.shape[1], 4, 96)


@pytest.mark.parametrize("query_timestep", [0, 1, 127, 128, 200, 256, 400, 2559])
def test_context_never_exceeds_256_tokens(query_timestep):
    history = torch.randn(query_timestep, 384)
    selection = select_moba_context(
        torch.randn(384),
        history,
        torch.arange(query_timestep),
        query_timestep,
        project_queries=_heads,
        project_keys=_heads,
        dense_recent=128,
        search_horizon=2560,
        block_size=16,
        retrieved_blocks=8,
        attention_budget=256,
    )
    assert selection.context_indices.numel() <= 256
    assert not set(selection.context_indices[:-128].tolist()) & set(
        selection.dense_indices.tolist()
    )


@pytest.mark.parametrize("query_timestep", [1, 64, 127, 128, 200, 256])
def test_through_256_all_history_is_attended_chronologically(query_timestep):
    history = torch.randn(query_timestep, 384)
    selection = select_moba_context(
        torch.randn(384),
        history,
        torch.arange(query_timestep),
        query_timestep,
        project_queries=_heads,
        project_keys=_heads,
        dense_recent=128,
        search_horizon=2560,
        block_size=16,
        retrieved_blocks=8,
        attention_budget=256,
    )
    assert selection.context_timesteps.tolist() == list(range(query_timestep))


def test_invalid_budget_is_rejected():
    with pytest.raises(ValueError, match="exceed"):
        select_moba_context(
            torch.randn(384),
            torch.randn(300, 384),
            torch.arange(300),
            300,
            project_queries=_heads,
            project_keys=_heads,
            dense_recent=129,
            block_size=16,
            retrieved_blocks=8,
            attention_budget=256,
        )


@pytest.mark.parametrize("layer_norm", ["pre", "post"])
def test_long_history_forward_runs_all_layers_and_returns_diagnostics(layer_norm):
    config = {
        "num_blocks": 3,
        "embed_dim": 384,
        "num_heads": 4,
        "positional_encoding": "relative",
        "layer_norm": layer_norm,
        "gtrxl": False,
    }
    torch.manual_seed(13)
    transformer = Transformer(config, input_dim=384, max_episode_steps=512)
    transformer.enable_lora()
    trace = EpisodeTrace(trace_id=0, num_layers=3, width=384)
    for timestep in range(300):
        trace.append(torch.randn(3, 384), timestep=timestep)
    context = trace.context(
        300, dense_recent=128, search_horizon=2560, block_size=16
    )

    output = transformer.forward_long_history(
        torch.randn(1, 384),
        [context],
        arm="trxl_moba",
        attention_budget=256,
        dense_recent=128,
        search_horizon=2560,
        block_size=16,
        retrieved_blocks=8,
    )

    assert output.hidden.shape == (1, 384)
    assert output.memories.shape == (1, 3, 384)
    assert not output.memories.requires_grad
    assert len(output.routing) == 3
    assert len(output.attention_weights) == 3
    for layer_routing, layer_weights in zip(
        output.routing, output.attention_weights, strict=True
    ):
        selection = layer_routing[0]
        assert selection is not None
        assert selection.context_indices.numel() <= 256
        assert layer_weights[0].shape[-1] == selection.context_indices.numel()
        assert 0.0 <= selection.retrieved_attention_mass <= 1.0
        assert selection.useful_retrieval
