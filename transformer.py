from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utils import Module


@dataclass(frozen=True)
class LongHistoryOutput:
    """Result of an actual dense/MoBA long-history Transformer forward."""

    hidden: torch.Tensor
    memories: torch.Tensor
    routing: tuple[tuple[Any | None, ...], ...]
    attention_weights: tuple[tuple[torch.Tensor, ...], ...]

class MultiHeadAttention(nn.Module):
    """Multi Head Attention without dropout inspired by https://github.com/aladdinpersson/Machine-Learning-Collection
    https://youtu.be/U0s0f995w14"""
    def __init__(self, embed_dim, num_heads):
        """
        Arguments:
            embed_dim {int} -- Size of the embedding dimension
            num_heads {int} -- Number of attention heads
        """
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads

        assert (
            self.head_size * num_heads == embed_dim
        ), "Embedding dimension needs to be divisible by the number of heads"

        # The shipped pretrained checkpoints were produced by the original
        # attention implementation.  It applies one shared head-sized
        # projection independently to every head.  Keep these child names and
        # shapes stable so those state dicts continue to load strictly.
        self.values = nn.Linear(self.head_size, self.head_size, bias=False)
        self.keys = nn.Linear(self.head_size, self.head_size, bias=False)
        self.queries = nn.Linear(self.head_size, self.head_size, bias=False)
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        self.query_lora = None
        self.key_lora = None
        self.value_lora = None
        self.output_lora = None

    @property
    def lora_enabled(self):
        return self.query_lora is not None

    def enable_lora(self, rank=8, alpha=16, dropout=0.0):
        """Attach zero-base logical projections for later PEFT injection."""
        if self.lora_enabled:
            raise RuntimeError("LoRA anchors are already attached")
        device = self.fc_out.weight.device
        dtype = self.fc_out.weight.dtype
        for name in ("query_lora", "key_lora", "value_lora", "output_lora"):
            anchor = nn.Linear(self.embed_dim, self.embed_dim, bias=False).to(
                device=device, dtype=dtype
            )
            nn.init.zeros_(anchor.weight)
            setattr(self, name, anchor)

    def _project_shared_heads(self, inputs, projection):
        """Apply a checkpoint-compatible projection shared across heads."""
        batch_size, sequence_length, _ = inputs.shape
        inputs = inputs.reshape(
            batch_size, sequence_length, self.num_heads, self.head_size
        )
        return projection(inputs)

    def _project_input(self, inputs, projection, adapter):
        projected = self._project_shared_heads(inputs, projection)
        if adapter is not None:
            projected = projected + adapter(inputs).reshape(
                *inputs.shape[:-1], self.num_heads, self.head_size
            )
        return projected

    def project_values(self, values):
        """Project values to ``(batch, sequence, heads, head_size)``."""
        return self._project_input(values, self.values, self.value_lora)

    def project_keys(self, keys):
        """Project keys to ``(batch, sequence, heads, head_size)``."""
        return self._project_input(keys, self.keys, self.key_lora)

    def project_queries(self, queries):
        """Project queries to ``(batch, sequence, heads, head_size)``."""
        return self._project_input(queries, self.queries, self.query_lora)

    def project_output(self, attention_output):
        """Apply the checkpoint-compatible output projection."""
        output = self.fc_out(attention_output)
        if self.output_lora is not None:
            output = output + self.output_lora(attention_output)
        return output

    def forward(self, values, keys, queries, mask):
        """
        The forward pass of the multi head attention layer.
        
        Arguments:
            values {torch.tensor} -- Value in shape of (N, L, D)
            keys {torch.tensor} -- Keys in shape of (N, L, D)
            queries {torch.tensor} -- Queries in shape of (N, L, D)
            mask {torch.tensor} -- Attention mask in shape of (N, L)
            
        Returns:
            torch.tensor -- Output
            torch.tensor -- Attention weights
        """
        # Get number of training examples and sequence lengths
        N = queries.shape[0]
        query_len = queries.shape[1]

        # Split into heads before projection. The same head-sized Q/K/V
        # projection is intentionally shared by all heads for compatibility
        # with the pretrained checkpoint.
        values = self.project_values(values)
        keys = self.project_keys(keys)
        queries = self.project_queries(queries)

        # Einsum does matrix mult. for query*keys for each training example
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their attention weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, float("-1e20")) # -inf causes NaN

        # Normalize energy values and apply softmax wo retreive the attention scores
        attention = torch.softmax(energy / (self.embed_dim ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        # Scale values by attention weights
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.num_heads * self.head_size
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        # Forward projection
        out = self.project_output(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_dim)

        return out, attention
        
class TransformerBlock(Module):
    def __init__(self, embed_dim, num_heads, config):
        """Transformer Block made of LayerNorms, Multi Head Attention and one fully connected feed forward projection.
        Arguments:
            embed_dim {int} -- Size of the embeddding dimension
            num_heads {int} -- Number of attention headds
            config {dict} -- General config
        """
        super(TransformerBlock, self).__init__()

        # Attention
        self.attention = MultiHeadAttention(embed_dim, num_heads)

        # Setup GTrXL if used
        self.use_gtrxl = config["gtrxl"] if "gtrxl" in config else False
        if self.use_gtrxl:
            self.gate1 = GRUGate(embed_dim, config["gtrxl_bias"])
            self.gate2 = GRUGate(embed_dim, config["gtrxl_bias"])

        # LayerNorms
        self.layer_norm = config["layer_norm"]
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        if self.layer_norm == "pre":
            self.norm_kv = nn.LayerNorm(embed_dim)

        # Feed forward projection
        self.fc = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())

    def forward(self, value, key, query, mask):
        """
        Arguments:
            values {torch.tensor} -- Value in shape of (N, L, D)
            keys {torch.tensor} -- Keys in shape of (N, L, D)
            query {torch.tensor} -- Queries in shape of (N, L, D)
            mask {torch.tensor} -- Attention mask in shape of (N, L)
        Returns:
            torch.tensor -- Output
            torch.tensor -- Attention weights
        """
        # Apply pre-layer norm across the attention input
        if self.layer_norm == "pre":
            query_ = self.norm1(query)
            value = self.norm_kv(value)
            key = value
        else:
            query_ = query

        # Forward MultiHeadAttention
        attention, attention_weights = self.attention(value, key, query_, mask)

        # GRU Gate or skip connection
        if self.use_gtrxl:
            # Forward GRU gating
            h = self.gate1(query, attention)
        else:
            # Skip connection
            h = attention + query
        
        # Apply post-layer norm across the attention output (i.e. projection input)
        if self.layer_norm == "post":
            h = self.norm1(h)

        # Apply pre-layer norm across the projection input (i.e. attention output)
        if self.layer_norm == "pre":
            h_ = self.norm2(h)
        else:
            h_ = h

        # Forward projection
        forward = self.fc(h_)

        # GRU Gate or skip connection
        if self.use_gtrxl:
            # Forward GRU gating
            out = self.gate2(h, forward)
        else:
            # Skip connection
            out = forward + h
        
        # Apply post-layer norm across the projection output
        if self.layer_norm == "post":
            out = self.norm2(out)

        return out, attention_weights

class SinusoidalPosition(nn.Module):
    """Relative positional encoding"""
    def __init__(self, dim, min_timescale = 2., max_timescale = 1e4):
        super().__init__()
        freqs = torch.arange(0, dim, min_timescale)
        inv_freqs = max_timescale ** (-freqs / dim)
        self.register_buffer('inv_freqs', inv_freqs)

    def forward(self, seq_len):
        seq = torch.arange(
            seq_len - 1,
            -1,
            -1,
            device=self.inv_freqs.device,
            dtype=self.inv_freqs.dtype,
        )
        # Plain broadcasting is exactly equivalent to the former einops
        # rearranges and avoids importing compiler/distributed machinery in a
        # simple eager evaluation path.
        sinusoidal_inp = seq[:, None] * self.inv_freqs[None, :]
        pos_emb = torch.cat((sinusoidal_inp.sin(), sinusoidal_inp.cos()), dim = -1)
        return pos_emb

class Transformer(nn.Module):
    """Transformer encoder architecture without dropout. Positional encoding can be either "relative", "learned" or "" (none)."""
    def __init__(self, config, input_dim, max_episode_steps) -> None:
        """Sets up the input embedding, positional encoding and the transformer blocks.
        Arguments:
            config {dict} -- Transformer config
            input_dim {int} -- Dimension of the input
            max_episode_steps {int} -- Maximum number of steps in an episode
        """
        super().__init__()
        self.config = config
        self.num_blocks = config["num_blocks"]
        self.embed_dim = config["embed_dim"]
        self.num_heads = config["num_heads"]
        self.max_episode_steps = max_episode_steps
        self.activation = nn.ReLU()

        # Input embedding layer
        self.linear_embedding = nn.Linear(input_dim, self.embed_dim)
        nn.init.orthogonal_(self.linear_embedding.weight, np.sqrt(2))

        # Determine positional encoding
        if config["positional_encoding"] == "relative":
            self.pos_embedding = SinusoidalPosition(dim = self.embed_dim)
        elif config["positional_encoding"] == "learned":
            self.pos_embedding = nn.Parameter(torch.randn(self.max_episode_steps, self.embed_dim)) # (batch size, max episoded steps, num layers, layer size)
        else:
            pass    # No positional encoding is used
        
        # Instantiate transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads, config) 
            for _ in range(self.num_blocks)])

    def enable_lora(self, rank=8, alpha=16, dropout=0.0):
        """Attach and inject PEFT Q/K/V/O adapters in every attention layer."""
        if any(block.attention.lora_enabled for block in self.transformer_blocks):
            return
        for block in self.transformer_blocks:
            block.attention.enable_lora(rank=rank, alpha=alpha, dropout=dropout)
        from episodic_moba_ppo.lora import inject_full_width_lora

        inject_full_width_lora(self, rank=rank, alpha=alpha, dropout=dropout)

    def position_embeddings(self, memory_indices):
        """Return pretrained positional vectors for absolute episode indices."""
        if self.config["positional_encoding"] == "relative":
            return self.pos_embedding(self.max_episode_steps)[memory_indices]
        if self.config["positional_encoding"] == "learned":
            return self.pos_embedding[memory_indices]
        return None

    def route_layer_context(
        self,
        layer_index,
        query,
        history,
        history_timesteps,
        query_timestep,
        *,
        dense_recent=128,
        search_horizon=2560,
        block_size=16,
        retrieved_blocks=8,
        attention_budget=256,
    ):
        """Select and position one sample's context for a Transformer layer.

        Routing is content-only. Positional vectors are applied only after the
        selected tokens have been restored to chronological order.
        """
        from episodic_moba_ppo.moba_retrieval import select_moba_context

        block = self.transformer_blocks[layer_index]
        query_2d = query.reshape(-1, self.embed_dim)
        history_2d = history.reshape(-1, self.embed_dim)
        normalized_query = block.norm1(query_2d)
        normalized_history = block.norm_kv(history_2d)
        selection = select_moba_context(
            normalized_query,
            normalized_history,
            history_timesteps,
            query_timestep,
            project_queries=block.attention.project_queries,
            project_keys=block.attention.project_keys,
            dense_recent=dense_recent,
            search_horizon=search_horizon,
            block_size=block_size,
            retrieved_blocks=retrieved_blocks,
            attention_budget=attention_budget,
        )
        context = history_2d.index_select(0, selection.context_indices)
        positions = self.position_embeddings(selection.context_timesteps)
        if positions is not None:
            context = context + positions
        return context, selection

    @staticmethod
    def _normalized_block_summaries_cpu(block, candidate_blocks, layer_index):
        """Return ``mean(norm_kv(token))`` summaries without moving old bodies.

        Episode traces intentionally retain all old token bodies on CPU.  The
        pretrained layer norm is frozen, so applying its detached parameters
        to each CPU token is exact and lets routing transfer only one
        full-width summary per candidate block.  In particular, this must not
        be replaced by ``norm_kv(mean(token))``: LayerNorm is nonlinear.
        """
        if not candidate_blocks:
            return torch.empty(
                (0, block.attention.embed_dim), dtype=torch.float32, device="cpu"
            )
        norm = block.norm_kv
        weight = norm.weight.detach().to(device="cpu", dtype=torch.float32)
        bias = norm.bias.detach().to(device="cpu", dtype=torch.float32)
        summaries = []
        for historical_block in candidate_blocks:
            token_bodies = historical_block.states[:, layer_index]
            if token_bodies.device.type != "cpu":
                raise AssertionError("unselected historical token bodies must stay on CPU")
            normalized = F.layer_norm(
                token_bodies,
                norm.normalized_shape,
                weight,
                bias,
                norm.eps,
            )
            summaries.append(normalized.mean(dim=0))
        return torch.stack(summaries)

    def forward_long_history(
        self,
        h,
        contexts: Sequence[Any],
        *,
        arm,
        attention_budget=256,
        dense_recent=128,
        search_horizon=2560,
        block_size=16,
        retrieved_blocks=8,
    ) -> LongHistoryOutput:
        """Run all layers against episode-local long-history contexts.

        Each context is an ``EpisodeContext`` materialized for the current PPO
        sample. The method intentionally processes samples independently so
        variable selected-context lengths never require transferring or
        padding unrelated episode history. Routes are recomputed on every call
        under the current Q/K adapter weights.
        """
        if arm not in {"trxl", "trxl_moba"}:
            raise ValueError(f"unsupported attention arm: {arm}")
        if len(contexts) != h.shape[0]:
            raise ValueError("one episode context is required per input sample")

        h = self.activation(self.linear_embedding(h))
        out_memories = []
        routing_by_layer = []
        weights_by_layer = []

        for layer_index, block in enumerate(self.transformer_blocks):
            out_memories.append(h.detach())
            next_hidden = []
            layer_routing = []
            layer_weights = []
            for sample_index, episode_context in enumerate(contexts):
                if arm == "trxl_moba":
                    from episodic_moba_ppo.moba_retrieval import (
                        USEFUL_ATTENTION_THRESHOLD,
                        MobaSelection,
                        select_moba_blocks,
                    )

                    candidate_blocks = episode_context.old_blocks
                    if candidate_blocks:
                        # Apply norm_kv to every raw CPU token before taking
                        # the block mean.  Only these compact summaries cross
                        # to the model device for current-Q/K routing.
                        normalized_summaries = (
                            self._normalized_block_summaries_cpu(
                                block, candidate_blocks, layer_index
                            ).to(device=h.device, dtype=h.dtype)
                        )
                        block_indices = torch.tensor(
                            [old.block_index for old in candidate_blocks],
                            dtype=torch.long,
                            device=h.device,
                        )
                    else:
                        normalized_summaries = h.new_empty((0, self.embed_dim))
                        block_indices = torch.empty(
                            0, dtype=torch.long, device=h.device
                        )
                    normalized_query = block.norm1(h[sample_index].reshape(1, -1))
                    block_selection = select_moba_blocks(
                        normalized_query,
                        normalized_summaries,
                        block_indices,
                        project_queries=block.attention.project_queries,
                        project_keys=block.attention.project_keys,
                        retrieved_blocks=retrieved_blocks,
                    )

                    selected_states = []
                    selected_timesteps = []
                    selected_ranges = []
                    for candidate_position in block_selection.candidate_positions.tolist():
                        old = candidate_blocks[candidate_position]
                        selected_states.append(
                            old.states[:, layer_index].to(
                                device=h.device, dtype=h.dtype
                            )
                        )
                        selected_timesteps.append(old.timesteps.to(device=h.device))
                        selected_ranges.append(
                            (int(old.timesteps[0]), int(old.timesteps[-1]))
                        )
                    selected_states.append(
                        episode_context.dense_states[:, layer_index]
                        .detach()
                        .to(device=h.device, dtype=h.dtype)
                    )
                    selected_timesteps.append(
                        episode_context.dense_timesteps.to(device=h.device)
                    )
                    attended = torch.cat(selected_states, dim=0)
                    attended_timesteps = torch.cat(selected_timesteps, dim=0)
                    if attended.shape[0] > attention_budget:
                        raise AssertionError("MoBA selection exceeded attention budget")
                    positions = self.position_embeddings(attended_timesteps)
                    if positions is not None:
                        attended = attended + positions
                    selection = MobaSelection(
                        context_indices=attended_timesteps.detach(),
                        context_timesteps=attended_timesteps.detach(),
                        dense_indices=episode_context.dense_timesteps.detach(),
                        selected_block_indices=block_selection.selected_block_indices,
                        selected_block_ranges=tuple(selected_ranges),
                        candidate_block_indices=block_indices.detach(),
                        routing_scores=block_selection.routing_scores,
                    )
                else:
                    dense_states = (
                        episode_context.dense_states[:, layer_index]
                        .detach()
                        .to(device=h.device, dtype=h.dtype)
                    )
                    dense_timesteps = episode_context.dense_timesteps.to(device=h.device)
                    needed = max(0, attention_budget - dense_states.shape[0])
                    old_state_parts = []
                    old_time_parts = []
                    for old in reversed(episode_context.old_blocks):
                        if needed == 0:
                            break
                        take = min(needed, old.states.shape[0])
                        old_state_parts.insert(
                            0,
                            old.states[-take:, layer_index].to(
                                device=h.device, dtype=h.dtype
                            ),
                        )
                        old_time_parts.insert(0, old.timesteps[-take:].to(h.device))
                        needed -= take
                    old_state_parts.append(dense_states)
                    old_time_parts.append(dense_timesteps)
                    attended = torch.cat(old_state_parts, dim=0)
                    attended_timesteps = torch.cat(old_time_parts, dim=0)
                    positions = self.position_embeddings(attended_timesteps)
                    if positions is not None:
                        attended = attended + positions
                    selection = None

                # The first episode step has no historical tokens. A single
                # content-zero slot keeps the block callable; it is not exposed
                # as an attended historical timestep or routing selection.
                if attended.shape[0] == 0:
                    attended = h.new_zeros((1, self.embed_dim))
                mask = torch.ones(
                    (1, attended.shape[0]), dtype=torch.bool, device=attended.device
                )
                sample_hidden, sample_weights = block(
                    attended.unsqueeze(0),
                    attended.unsqueeze(0),
                    h[sample_index].reshape(1, 1, self.embed_dim),
                    mask,
                )
                if selection is not None:
                    retrieved_tokens = attended.shape[0] - int(
                        episode_context.dense_timesteps.numel()
                    )
                    retrieved_mass = (
                        float(
                            sample_weights[..., :retrieved_tokens]
                            .sum(dim=-1)
                            .mean()
                            .detach()
                            .cpu()
                        )
                        if retrieved_tokens
                        else 0.0
                    )
                    has_distant_block = any(
                        episode_context.query_timestep - end > dense_recent
                        for _, end in selection.selected_block_ranges
                    )
                    selection = replace(
                        selection,
                        retrieved_attention_mass=retrieved_mass,
                        useful_retrieval=(
                            has_distant_block
                            and retrieved_mass > USEFUL_ATTENTION_THRESHOLD
                        ),
                    )
                next_hidden.append(sample_hidden.reshape(self.embed_dim))
                layer_routing.append(selection)
                layer_weights.append(sample_weights)

            h = torch.stack(next_hidden, dim=0)
            routing_by_layer.append(tuple(layer_routing))
            weights_by_layer.append(tuple(layer_weights))

        return LongHistoryOutput(
            hidden=h,
            memories=torch.stack(out_memories, dim=1),
            routing=tuple(routing_by_layer),
            attention_weights=tuple(weights_by_layer),
        )

    def forward(self, h, memories, mask, memory_indices):
        """
        Arguments:
            h {torch.tensor} -- Input (query)
            memories {torch.tesnor} -- Whole episoded memories of shape (N, L, num blocks, D)
            mask {torch.tensor} -- Attention mask (dtype: bool) of shape (N, L)
            memory_indices {torch.tensor} -- Memory window indices (dtype: long) of shape (N, L)
        Returns:
            {torch.tensor} -- Output of the entire transformer encoder
            {torch.tensor} -- Out memories (i.e. inputs to the transformer blocks)
        """
        # Feed embedding layer and activate
        h = self.activation(self.linear_embedding(h))

        # Add positional encoding to every transformer block input
        pos_embedding = self.position_embeddings(memory_indices)
        if pos_embedding is not None:
            memories = memories + pos_embedding.unsqueeze(2)

        # Forward transformer blocks
        out_memories = []
        for i, block in enumerate(self.transformer_blocks):
            out_memories.append(h.detach())
            h, attention_weights = block(memories[:, :, i], memories[:, :, i], h.unsqueeze(1), mask) # args: value, key, query, mask
            h = h.squeeze()
            if len(h.shape) == 1:
                h = h.unsqueeze(0)
        return h, torch.stack(out_memories, dim=1)
    
class GRUGate(nn.Module):
    """
    Overview:
        GRU Gating Unit used in GTrXL.
        Inspired by https://github.com/dhruvramani/Transformers-RL/blob/master/layers.py
    """

    def __init__(self, input_dim: int, bg: float = 0.0):
        """
        Arguments:
            input_dim {int} -- Input dimension
            bg {float} -- Initial gate bias value. By setting bg > 0 we can explicitly initialize the gating mechanism to
            be close to the identity map. This can greatly improve the learning speed and stability since it
            initializes the agent close to a Markovian policy (ignore attention at the beginning). (default: {0.0})
        """
        super(GRUGate, self).__init__()
        self.Wr = nn.Linear(input_dim, input_dim, bias=False)
        self.Ur = nn.Linear(input_dim, input_dim, bias=False)
        self.Wz = nn.Linear(input_dim, input_dim, bias=False)
        self.Uz = nn.Linear(input_dim, input_dim, bias=False)
        self.Wg = nn.Linear(input_dim, input_dim, bias=False)
        self.Ug = nn.Linear(input_dim, input_dim, bias=False)
        self.bg = nn.Parameter(torch.full([input_dim], bg))  # bias
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        nn.init.xavier_uniform_(self.Wr.weight)
        nn.init.xavier_uniform_(self.Ur.weight)
        nn.init.xavier_uniform_(self.Wz.weight)
        nn.init.xavier_uniform_(self.Uz.weight)
        nn.init.xavier_uniform_(self.Wg.weight)
        nn.init.xavier_uniform_(self.Ug.weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """        
        Arguments:
            x {torch.tensor} -- First input
            y {torch.tensor} -- Second input
        Returns:
            {torch.tensor} -- Output
        """
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))
        return torch.mul(1 - z, x) + torch.mul(z, h)
