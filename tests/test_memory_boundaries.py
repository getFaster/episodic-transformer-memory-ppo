import pytest
import torch

from episodic_moba_ppo.episodic_memory import TraceRegistry
from transformer import Transformer


def _state(timestep: int) -> torch.Tensor:
    value = torch.full((3, 4), float(timestep), requires_grad=True)
    return value


def test_trace_detaches_to_cpu_and_preserves_absolute_time() -> None:
    registry = TraceRegistry(num_layers=3, width=4)
    trace = registry.start_worker(0)
    for timestep in range(20):
        reference = registry.append(0, _state(timestep))
        assert reference.query_timestep == timestep

    assert trace.states.device.type == "cpu"
    assert trace.states.dtype == torch.float32
    assert not trace.states.requires_grad
    context = registry.context(
        trace.reference(), dense_recent=4, search_horizon=12, block_size=4
    )
    assert context.dense_timesteps.tolist() == [16, 17, 18, 19]
    assert [block.block_index for block in context.old_blocks] == [2, 3]
    assert [block.timesteps.tolist() for block in context.old_blocks] == [
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ]


def test_fixed_blocks_are_clipped_at_search_and_dense_boundaries() -> None:
    registry = TraceRegistry(num_layers=3, width=4)
    trace = registry.start_worker(0)
    for timestep in range(23):
        registry.append(0, _state(timestep))

    context = registry.context(
        trace.reference(), dense_recent=6, search_horizon=14, block_size=8
    )
    assert context.dense_timesteps.tolist() == [17, 18, 19, 20, 21, 22]
    assert [block.block_index for block in context.old_blocks] == [1, 2]
    assert context.old_blocks[0].timesteps.tolist() == list(range(9, 16))
    assert context.old_blocks[1].timesteps.tolist() == [16]
    old = torch.cat([block.timesteps for block in context.old_blocks])
    assert int(old.max()) < int(context.dense_timesteps.min())
    assert int(old.min()) >= 23 - 14


def test_reset_closes_old_trace_and_prevents_cross_episode_lookup() -> None:
    registry = TraceRegistry(num_layers=3, width=4)
    first = registry.start_worker(7)
    for timestep in range(3):
        registry.append(7, _state(timestep))
    old_ref = first.reference()
    second = registry.reset_worker(7)
    registry.append(7, _state(100))

    assert first.closed
    assert first.trace_id != second.trace_id
    assert registry.context(
        second.reference(), dense_recent=8, search_horizon=32, block_size=4
    ).dense_timesteps.tolist() == [0]
    assert registry.context(
        old_ref, dense_recent=8, search_horizon=32, block_size=4
    ).dense_timesteps.tolist() == [0, 1, 2]
    with pytest.raises(RuntimeError, match="closed"):
        first.append(_state(3))


def test_future_query_is_rejected() -> None:
    registry = TraceRegistry(num_layers=3, width=4)
    trace = registry.start_worker(0)
    registry.append(0, _state(0))
    with pytest.raises(ValueError, match="future"):
        trace.context(2, dense_recent=1, search_horizon=2, block_size=1)


def test_closed_traces_can_be_pruned_without_losing_active_history() -> None:
    registry = TraceRegistry(num_layers=3, width=4)
    first = registry.start_worker(0)
    registry.append(0, _state(0))
    second = registry.reset_worker(0)
    registry.append(0, _state(1))

    assert registry.prune_closed() == 1
    with pytest.raises(KeyError, match="unknown"):
        registry.get(first.trace_id)
    assert registry.get(second.trace_id) is second


def test_routing_summary_means_tokenwise_normalized_cpu_states() -> None:
    """Routing must use mean(LayerNorm(token)), not LayerNorm(mean(token))."""
    config = {
        "num_blocks": 3,
        "embed_dim": 384,
        "num_heads": 4,
        "positional_encoding": "relative",
        "layer_norm": "pre",
        "gtrxl": False,
    }
    torch.manual_seed(23)
    transformer = Transformer(config, input_dim=384, max_episode_steps=512)
    trace = TraceRegistry(num_layers=3, width=384).start_worker(0)
    for timestep in range(12):
        trace.append(torch.randn(3, 384), timestep=timestep)
    context = trace.context(12, dense_recent=4, search_horizon=12, block_size=4)
    block = transformer.transformer_blocks[0]

    actual = transformer._normalized_block_summaries_cpu(
        block, context.old_blocks, layer_index=0
    )
    expected = torch.stack(
        [block.norm_kv(old.states[:, 0]).mean(dim=0) for old in context.old_blocks]
    )
    wrong_order = torch.stack(
        [block.norm_kv(old.states[:, 0].mean(dim=0)) for old in context.old_blocks]
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.allclose(actual, wrong_order)
    assert actual.device.type == "cpu"
    assert all(old.states.device.type == "cpu" for old in context.old_blocks)


def test_post_norm_routing_summary_uses_raw_cpu_states() -> None:
    config = {
        "num_blocks": 3,
        "embed_dim": 384,
        "num_heads": 4,
        "positional_encoding": "relative",
        "layer_norm": "post",
        "gtrxl": False,
    }
    transformer = Transformer(config, input_dim=384, max_episode_steps=512)
    trace = TraceRegistry(num_layers=3, width=384).start_worker(0)
    for timestep in range(12):
        trace.append(torch.randn(3, 384), timestep=timestep)
    context = trace.context(12, dense_recent=4, search_horizon=12, block_size=4)

    actual = transformer._normalized_block_summaries_cpu(
        transformer.transformer_blocks[0], context.old_blocks, layer_index=0
    )
    expected = torch.stack(
        [old.states[:, 0].mean(dim=0) for old in context.old_blocks]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
