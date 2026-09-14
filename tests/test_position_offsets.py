import torch

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


def test_absolute_positions_do_not_depend_on_retrieval_order():
    transformer = Transformer(_config(), input_dim=384, max_episode_steps=2560)
    chronological = torch.tensor([100, 115, 400, 415, 799])
    permuted = chronological[torch.tensor([2, 0, 4, 1, 3])]
    chronological_pos = transformer.position_embeddings(chronological)
    permuted_pos = transformer.position_embeddings(permuted)

    for index, timestep in enumerate(permuted):
        original = torch.nonzero(chronological == timestep, as_tuple=False).item()
        torch.testing.assert_close(permuted_pos[index], chronological_pos[original])


def test_dense_position_helper_matches_original_forward_expression():
    transformer = Transformer(_config(), input_dim=384, max_episode_steps=512)
    indices = torch.tensor([[0, 1, 117], [118, 255, 511]])
    expected = transformer.pos_embedding(512)[indices]
    torch.testing.assert_close(transformer.position_embeddings(indices), expected)


def test_routed_context_adds_true_absolute_positions_after_selection():
    torch.manual_seed(3)
    transformer = Transformer(_config(), input_dim=384, max_episode_steps=512)
    history = torch.zeros(192, 384)
    timesteps = torch.arange(192)
    context, selection = transformer.route_layer_context(
        0,
        torch.ones(384),
        history,
        timesteps,
        192,
        dense_recent=128,
        block_size=16,
        retrieved_blocks=4,
        attention_budget=192,
    )
    expected = transformer.position_embeddings(selection.context_timesteps)
    torch.testing.assert_close(context, expected)
