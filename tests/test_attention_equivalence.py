import torch

from episodic_moba_ppo.episodic_memory import EpisodeTrace
from transformer import MultiHeadAttention
from transformer import Transformer


def _legacy_attention(module, values, keys, queries, mask):
    batch_size = queries.shape[0]
    value_len = values.shape[1]
    key_len = keys.shape[1]
    query_len = queries.shape[1]

    values = values.reshape(
        batch_size, value_len, module.num_heads, module.head_size
    )
    keys = keys.reshape(batch_size, key_len, module.num_heads, module.head_size)
    queries = queries.reshape(
        batch_size, query_len, module.num_heads, module.head_size
    )

    values = module.values(values)
    keys = module.keys(keys)
    queries = module.queries(queries)
    energy = torch.einsum("nqhd,nkhd->nhqk", queries, keys)
    if mask is not None:
        energy = energy.masked_fill(
            mask.unsqueeze(1).unsqueeze(1) == 0, float("-1e20")
        )
    attention = torch.softmax(energy / (module.embed_dim**0.5), dim=3)
    output = torch.einsum("nhql,nlhd->nqhd", attention, values).reshape(
        batch_size, query_len, module.embed_dim
    )
    return module.fc_out(output), attention


def test_refactored_attention_matches_legacy_implementation():
    torch.manual_seed(7)
    attention = MultiHeadAttention(embed_dim=384, num_heads=4)
    values = torch.randn(3, 11, 384)
    keys = torch.randn(3, 11, 384)
    queries = torch.randn(3, 2, 384)
    mask = torch.tensor(
        [
            [1] * 11,
            [1] * 7 + [0] * 4,
            [1] * 3 + [0] * 8,
        ],
        dtype=torch.bool,
    )

    expected_output, expected_weights = _legacy_attention(
        attention, values, keys, queries, mask
    )
    actual_output, actual_weights = attention(values, keys, queries, mask)

    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=1e-6)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=1e-7)


def test_projection_helpers_preserve_legacy_shapes_and_names():
    attention = MultiHeadAttention(embed_dim=384, num_heads=4)
    inputs = torch.randn(2, 5, 384)

    assert attention.project_queries(inputs).shape == (2, 5, 4, 96)
    assert attention.project_keys(inputs).shape == (2, 5, 4, 96)
    assert attention.project_values(inputs).shape == (2, 5, 4, 96)
    assert attention.project_output(inputs).shape == (2, 5, 384)
    assert tuple(attention.queries.weight.shape) == (96, 96)
    assert tuple(attention.keys.weight.shape) == (96, 96)
    assert tuple(attention.values.weight.shape) == (96, 96)
    assert tuple(attention.fc_out.weight.shape) == (384, 384)
    assert set(attention.state_dict()) == {
        "values.weight",
        "keys.weight",
        "queries.weight",
        "fc_out.weight",
        "fc_out.bias",
    }


def test_dense_long_history_forward_matches_legacy_forward():
    config = {
        "num_blocks": 3,
        "embed_dim": 384,
        "num_heads": 4,
        "positional_encoding": "relative",
        "layer_norm": "pre",
        "gtrxl": False,
    }
    torch.manual_seed(11)
    transformer = Transformer(config, input_dim=384, max_episode_steps=512)
    inputs = torch.randn(2, 384)
    memories = torch.randn(2, 118, 3, 384)
    indices = torch.arange(118).expand(2, -1)
    mask = torch.ones(2, 118, dtype=torch.bool)
    expected_hidden, expected_memories = transformer(inputs, memories, mask, indices)

    contexts = []
    for sample in range(2):
        trace = EpisodeTrace(trace_id=sample, num_layers=3, width=384)
        for timestep in range(118):
            trace.append(memories[sample, timestep], timestep=timestep)
        contexts.append(
            trace.context(
                118,
                dense_recent=118,
                search_horizon=512,
                block_size=16,
            )
        )
    actual = transformer.forward_long_history(
        inputs, contexts, arm="trxl", attention_budget=256
    )

    torch.testing.assert_close(actual.hidden, expected_hidden, rtol=0, atol=1e-6)
    torch.testing.assert_close(actual.memories, expected_memories, rtol=0, atol=1e-6)
    assert all(selection is None for layer in actual.routing for selection in layer)
