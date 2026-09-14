import torch

from episodic_moba_ppo.lora import assert_lora_trainable_set, freeze_for_lora
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
