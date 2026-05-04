"""Perceiver resampler module for soft prompt compression."""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


def _default_kernel_init():
    return nn.initializers.xavier_uniform()


class FeedForward(nn.Module):
    dim: int
    mult: int = 4

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        inner_dim = int(self.dim * self.mult)
        x = nn.LayerNorm()(x)
        x = nn.Dense(inner_dim, use_bias=False, kernel_init=_default_kernel_init())(x)
        x = nn.gelu(x)
        x = nn.Dense(self.dim, use_bias=False, kernel_init=_default_kernel_init())(x)
        return x


class PerceiverAttention(nn.Module):
    dim: int
    dim_head: int = 64
    heads: int = 8

    @nn.compact
    def __call__(
        self,
        tokens: jax.Array,
        latents: jax.Array,
        mask: jax.Array | None = None,
        return_weights: bool = False,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        tokens = nn.LayerNorm()(tokens)
        latents = nn.LayerNorm()(latents)

        inner_dim = self.dim_head * self.heads
        to_q = nn.Dense(inner_dim, use_bias=False, kernel_init=_default_kernel_init())
        to_kv = nn.Dense(inner_dim * 2, use_bias=False, kernel_init=_default_kernel_init())
        to_out = nn.Dense(self.dim, use_bias=False, kernel_init=_default_kernel_init())

        q = to_q(latents)
        kv_input = jnp.concatenate([tokens, latents], axis=1)
        k, v = jnp.split(to_kv(kv_input), 2, axis=-1)

        batch_size, num_latents = latents.shape[:2]
        num_kv = kv_input.shape[1]

        q = q.reshape(batch_size, num_latents, self.heads, self.dim_head)
        k = k.reshape(batch_size, num_kv, self.heads, self.dim_head)
        v = v.reshape(batch_size, num_kv, self.heads, self.dim_head)

        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        scale = self.dim_head**-0.5
        q = q * scale

        sim = jnp.einsum("bhid,bhjd->bhij", q, k)

        if mask is not None:
            mask = mask.astype(bool)
            latents_mask = jnp.ones((mask.shape[0], num_latents), dtype=bool)
            kv_mask = jnp.concatenate([mask, latents_mask], axis=1)
            sim = jnp.where(kv_mask[:, None, None, :], sim, jnp.finfo(sim.dtype).min)

        attn = jax.nn.softmax(sim, axis=-1)
        out = jnp.einsum("bhij,bhjd->bhid", attn, v)
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(batch_size, num_latents, inner_dim)
        out = to_out(out)

        if return_weights:
            # Return only the trajectory-token slice: [B, heads, num_latents, T]
            num_traj = tokens.shape[1]
            return out, attn[:, :, :, :num_traj]
        return out


class PerceiverResampler(nn.Module):
    dim: int
    depth: int = 6
    dim_head: int = 64
    heads: int = 8
    num_latents: int = 64
    ff_mult: int = 4

    @nn.compact
    def __call__(self, tokens: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        latents = self.param("latents", nn.initializers.normal(stddev=1.0), (self.num_latents, self.dim))
        latents = jnp.broadcast_to(latents, (tokens.shape[0], self.num_latents, self.dim))

        for _ in range(self.depth):
            latents = latents + PerceiverAttention(dim=self.dim, dim_head=self.dim_head, heads=self.heads)(
                tokens, latents, mask=mask
            )
            latents = latents + FeedForward(dim=self.dim, mult=self.ff_mult)(latents)

        return nn.LayerNorm()(latents)


class TrajPerceiverResampler(nn.Module):
    """Perceiver resampler conditioned on a query observation embedding.

    Unlike PerceiverResampler (fixed learned latents), the num_latents initial
    queries are seeded from the current observation embedding so that cross-
    attention over the reference trajectory is query-conditioned: the model
    extracts what is relevant to *this* timestep from the trajectory.
    """

    dim: int
    depth: int = 6
    dim_head: int = 64
    heads: int = 8
    num_latents: int = 32
    ff_mult: int = 4

    @nn.compact
    def __call__(
        self,
        traj_tokens: jax.Array,
        query_embed: jax.Array,
        traj_mask: jax.Array | None = None,
        return_attn_weights: bool = False,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        # query_embed: [B, D] — aggregated query observation vector
        # traj_tokens: [B, T, D] — reference trajectory token sequence
        # Seed num_latents slots from query_embed; FFN layers differentiate them.
        latents = jnp.broadcast_to(
            query_embed[:, None, :], (query_embed.shape[0], self.num_latents, self.dim)
        )
        latents = jnp.array(latents)  # make writeable copy

        all_attn_weights = []
        for _ in range(self.depth):
            attn_out = PerceiverAttention(dim=self.dim, dim_head=self.dim_head, heads=self.heads)(
                traj_tokens, latents, mask=traj_mask, return_weights=return_attn_weights
            )
            if return_attn_weights:
                attn_out, weights = attn_out  # weights: [B, heads, num_latents, T]
                all_attn_weights.append(weights)
            latents = latents + attn_out
            latents = latents + FeedForward(dim=self.dim, mult=self.ff_mult)(latents)

        latents = nn.LayerNorm()(latents)

        if return_attn_weights:
            # Stack over layers, average over heads -> [B, num_latents, T]
            stacked = jnp.stack(all_attn_weights, axis=0)  # [depth, B, heads, num_latents, T]
            avg_weights = stacked.mean(axis=(0, 2))          # [B, num_latents, T]
            return latents, avg_weights
        return latents
