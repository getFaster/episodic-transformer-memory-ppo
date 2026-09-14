import torch

from episodic_moba_ppo.moba_retrieval import select_moba_context


def _heads(inputs):
    return inputs.reshape(inputs.shape[0], inputs.shape[1], 4, 96)


def test_max_head_routing_ties_and_chronological_restoration():
    history = torch.zeros(320, 384)
    timesteps = torch.arange(320)
    # Old candidate blocks 0..11 tie at zero. Blocks 10 and 11 have the
    # largest scores in different heads and must be returned chronologically.
    history[160:176, :96] = 4.0
    history[176:192, 96:192] = 5.0
    query = torch.ones(384)

    selection = select_moba_context(
        query,
        history,
        timesteps,
        320,
        project_queries=_heads,
        project_keys=_heads,
        dense_recent=128,
        search_horizon=2560,
        block_size=16,
        retrieved_blocks=2,
        attention_budget=160,
    )

    assert selection.selected_block_indices.tolist() == [10, 11]
    assert selection.selected_block_ranges == ((160, 175), (176, 191))
    assert selection.context_timesteps.tolist() == list(range(160, 320))
    assert selection.context_indices.requires_grad is False


def test_equal_score_ties_choose_earliest_chronological_blocks():
    history = torch.ones(192, 384)
    selection = select_moba_context(
        torch.ones(384),
        history,
        torch.arange(192),
        192,
        project_queries=_heads,
        project_keys=_heads,
        dense_recent=128,
        block_size=16,
        retrieved_blocks=2,
        attention_budget=160,
    )
    assert selection.selected_block_indices.tolist() == [0, 1]


def test_partial_blocks_are_clipped_at_horizon_and_dense_boundaries():
    timesteps = torch.arange(300)
    history = torch.randn(300, 384)
    selection = select_moba_context(
        torch.randn(384),
        history,
        timesteps,
        300,
        project_queries=_heads,
        project_keys=_heads,
        dense_recent=128,
        search_horizon=145,
        block_size=16,
        retrieved_blocks=8,
        attention_budget=256,
    )
    # Eligible range is 155..299; old range 155..171 crosses both fixed blocks.
    assert selection.candidate_block_indices.tolist() == [9, 10]
    assert selection.context_timesteps[-128:].tolist() == list(range(172, 300))
    assert all(155 <= step < 172 for step in selection.context_timesteps[:-128].tolist())


def test_selection_indices_are_detached_but_scores_remain_differentiable():
    history = torch.randn(192, 384, requires_grad=True)
    query = torch.randn(384, requires_grad=True)
    selection = select_moba_context(
        query,
        history,
        torch.arange(192),
        192,
        project_queries=_heads,
        project_keys=_heads,
        dense_recent=128,
        block_size=16,
        retrieved_blocks=2,
        attention_budget=160,
    )
    assert selection.context_indices.grad_fn is None
    assert selection.selected_block_indices.grad_fn is None
    assert selection.routing_scores.requires_grad
