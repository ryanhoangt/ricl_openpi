"""Pi0-FAST with query-conditioned trajectory perceiver for in-context learning.

Instead of per-timestep retrieval, a single reference demo trajectory is
compressed into num_latents tokens via TrajPerceiverResampler, where the
query is the current observation embedding.  These latents are prepended as a
bidirectional soft prefix before the query token sequence.

Context layout:
    [traj_latents × K | query_images | query_prefix | query_action tokens →]
"""

from __future__ import annotations

import dataclasses
import logging

import einops
import flax.linen as nn
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma_fast as _gemma
from openpi.models.perceiver_resampler import TrajPerceiverResampler
import openpi.models.siglip as _siglip
from openpi.models.pi0_fast_ricl import make_attn_mask, put_along_last_axis
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")

PALIGEMMA_EOS_TOKEN = 1


@dataclasses.dataclass(frozen=True)
class Pi0FASTTrajPerceiverConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"

    action_dim: int = 7
    state_dim: int = 8
    action_horizon: int = 10
    max_token_len: int = 180

    # Number of latent tokens the TrajPerceiverResampler outputs.
    num_latents: int = 32
    # Reference trajectory is subsampled / padded to this length.
    max_traj_len: int = 300
    # Mean-pooled DINOv2 ViT-B/14 embedding dim per camera (64 patches × 768 → pooled to 768).
    traj_dino_emb_dim: int = 768

    # Kept for DataLoader routing compatibility — set True so distances are
    # loaded by the dataset if needed in future. Has no effect on this model.
    use_action_interpolation: bool = False

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_FAST

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0FASTTrajPerceiver":
        return Pi0FASTTrajPerceiver(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "base_1_rgb": image_spec,
                    "wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "base_1_rgb": image_mask_spec,
                    "wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                token_ar_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                token_loss_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        if "lora" in self.paligemma_variant:
            return nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*lora.*")))
        return nnx.Nothing

    def get_freeze_filter_with_frozen_img_encoder(self) -> nnx.filterlib.Filter:
        if "lora" in self.paligemma_variant:
            return nnx.Any(
                nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*lora.*"))),
                nnx_utils.PathRegex(".*img.*"),
            )
        return nnx.All(nnx_utils.PathRegex(".*img.*"), nnx.Not(nnx_utils.PathRegex(".*llm.*")))

    def get_freeze_filter_full_backbone(self) -> nnx.filterlib.Filter:
        """Freeze entire VLM (SigLIP + Gemma); only traj_proj and perceiver are trainable."""
        return nnx.Any(nnx_utils.PathRegex(".*img.*"), nnx_utils.PathRegex(".*llm.*"))


class Pi0FASTTrajPerceiver(_model.BaseModel):
    def __init__(self, config: Pi0FASTTrajPerceiverConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        model_dim = paligemma_config.width

        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                **paligemma_config,
                embed_dtype=config.dtype,
                cache_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init")

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=model_dim,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # Projects [top_dino_emb; wrist_dino_emb; state] → [B, T, model_dim] for K/V
        traj_input_dim = config.traj_dino_emb_dim * 2 + config.state_dim
        traj_proj = nnx_bridge.ToNNX(nn.Dense(model_dim, use_bias=True))
        traj_proj.lazy_init(jnp.zeros((1, 1, traj_input_dim)), rngs=rngs)
        self.traj_proj = traj_proj

        # Projects query top-image DINOv2 emb → model_dim for perceiver Q seed.
        # Using DINOv2 (same space as K) rather than SigLIP for Q/K alignment.
        query_dino_proj = nnx_bridge.ToNNX(nn.Dense(model_dim, use_bias=True))
        query_dino_proj.lazy_init(jnp.zeros((1, config.traj_dino_emb_dim)), rngs=rngs)
        self.query_dino_proj = query_dino_proj

        perceiver = nnx_bridge.ToNNX(
            TrajPerceiverResampler(
                dim=model_dim,
                num_latents=config.num_latents,
                depth=6,
                dim_head=64,
                heads=8,
            )
        )
        # Lazy-init with dummy traj_tokens [1, 1, model_dim] and query_embed [1, model_dim]
        perceiver.lazy_init(
            jnp.zeros((1, 1, model_dim)),
            jnp.zeros((1, model_dim)),
            rngs=rngs,
        )
        self.perceiver = perceiver

        self.num_latents = config.num_latents
        self.max_token_len = config.max_token_len

    # ------------------------------------------------------------------
    # Core: embed query obs, compute mean-pooled embed for perceiver Q
    # ------------------------------------------------------------------

    def embed_inputs_and_query_embed(
        self,
        obs: _model.ObservationPrefixPostfix,
    ) -> tuple[at.Float[at.Array, "b s d"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"], at.Float[at.Array, "b d"]]:
        """Embed query obs images + text tokens; also return mean-pooled image embed for perceiver Q."""
        input_mask_parts = []
        ar_mask_parts = []
        token_parts = []
        patch_means = []

        for name in obs.images:
            patches, _ = self.PaliGemma.img(obs.images[name], train=False)  # [B, P, D]
            token_parts.append(patches)
            img_mask = einops.repeat(obs.image_masks[name], "b -> b p", p=patches.shape[1])
            input_mask_parts.append(img_mask)
            ar_mask_parts.append(0 * img_mask)
            # Weighted mean for perceiver Q (zero out masked/invalid images)
            patch_means.append(patches.mean(axis=1) * obs.image_masks[name].astype(jnp.float32)[:, None])

        # Mean-pool across all cameras → single query vector [B, D]
        query_embed = jnp.stack(patch_means, axis=0).mean(axis=0)

        assert obs.tokenized_prompt_prefix is not None
        if obs.tokenized_prompt_postfix is not None:
            text_tokens = jnp.concatenate([obs.tokenized_prompt_prefix, obs.tokenized_prompt_postfix], axis=1)
        else:
            text_tokens = obs.tokenized_prompt_prefix
        text_embeds = self.PaliGemma.llm(text_tokens, embed_only=True)

        token_parts.append(text_embeds)
        input_mask_parts.append(obs.tokenized_prompt_mask)
        ar_mask_parts.append(obs.token_ar_mask)

        return (
            jnp.concatenate(token_parts, axis=1),
            jnp.concatenate(input_mask_parts, axis=1),
            jnp.concatenate(ar_mask_parts, axis=1),
            query_embed,
        )

    def _build_obs_from_dict(self, obs_dict: dict) -> _model.ObservationPrefixPostfix:
        images = {
            k: (v.astype(jnp.float32) / 255.0 * 2.0 - 1.0) if v.dtype == jnp.uint8 else v
            for k, v in obs_dict["query_image"].items()
        }
        return _model.ObservationPrefixPostfix(
            images=images,
            image_masks=obs_dict["query_image_mask"],
            state=obs_dict["query_state"],
            tokenized_prompt_prefix=obs_dict.get("query_tokenized_prompt_prefix"),
            tokenized_prompt_postfix=obs_dict.get("query_tokenized_prompt_postfix"),
            tokenized_prompt_mask=obs_dict.get("query_tokenized_prompt_mask"),
            token_ar_mask=obs_dict.get("query_token_ar_mask"),
            token_loss_mask=obs_dict.get("query_token_loss_mask"),
        )

    def _build_traj_prefix(
        self,
        traj_state: at.Float[at.Array, "b t s"],
        traj_top_emb: at.Float[at.Array, "b t e"],
        traj_wrist_emb: at.Float[at.Array, "b t e"],
        traj_mask: at.Bool[at.Array, "b t"],
        query_dino_top_emb: at.Float[at.Array, "b e"],
    ) -> tuple[at.Float[at.Array, "b k d"], at.Bool[at.Array, "b k"], at.Int[at.Array, "b k"]]:
        # K/V: project trajectory (DINOv2 top + wrist + state) → model_dim
        traj_input = jnp.concatenate([traj_top_emb, traj_wrist_emb, traj_state], axis=-1)
        traj_tokens = self.traj_proj(traj_input)                         # [B, T, D]
        # Q seed: project query DINOv2 top emb → model_dim (same space as K)
        query_embed = self.query_dino_proj(query_dino_top_emb)           # [B, D]
        latents = self.perceiver(traj_tokens, query_embed, traj_mask=traj_mask)  # [B, K, D]
        batch_size = latents.shape[0]
        latent_mask = jnp.ones((batch_size, self.num_latents), dtype=jnp.bool_)
        latent_ar_mask = jnp.zeros((batch_size, self.num_latents), dtype=jnp.int32)
        return latents, latent_mask, latent_ar_mask

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        obs_dict: dict,
        actions: _model.Actions,
        *,
        train: bool = False,
        decode_indices: at.Int[at.Array, ""] = None,
    ) -> at.Float[at.Array, "*b"]:
        query_obs = self._build_obs_from_dict(obs_dict)
        query_obs = _model.preprocess_observation_prefix_postfix(
            rng, query_obs, train=train, image_keys=list(query_obs.images.keys())
        )

        query_embeddings, query_input_mask, query_ar_mask, query_embed = (
            self.embed_inputs_and_query_embed(query_obs)
        )

        traj_latents, latent_mask, latent_ar_mask = self._build_traj_prefix(
            obs_dict["traj_state"], obs_dict["traj_top_emb"], obs_dict["traj_wrist_emb"],
            obs_dict["traj_mask"], obs_dict["query_dino_top_emb"],
        )

        full_embeddings = jnp.concatenate([traj_latents, query_embeddings], axis=1)
        full_input_mask = jnp.concatenate([latent_mask, query_input_mask], axis=1)
        full_ar_mask = jnp.concatenate([latent_ar_mask, query_ar_mask], axis=1)
        full_attn_mask = make_attn_mask(full_input_mask, full_ar_mask)

        loss_mask = query_obs.token_loss_mask[:, 1:]
        targets = jax.nn.one_hot(
            jnp.concatenate(
                [query_obs.tokenized_prompt_prefix[:, 1:], query_obs.tokenized_prompt_postfix],
                axis=1,
            ),
            self.PaliGemma.llm.module.vocab_size,
        )

        pre_logits, _, _ = self.PaliGemma.llm(
            embedded_prefix=full_embeddings[:, :-1],
            mask=full_attn_mask[:, :-1, :-1],
            return_prelogits=True,
        )
        logits, _ = self.PaliGemma.llm(pre_logits=pre_logits[:, -targets.shape[1]:])

        logp = jax.nn.log_softmax(logits, axis=-1)
        token_pplx = jnp.sum(targets * logp, axis=-1)
        loss = -jnp.sum(token_pplx * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, -1), 1)
        return loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        obs_dict: dict,
        *,
        temperature: float = 0.0,
    ) -> _model.Actions:
        query_obs = self._build_obs_from_dict(obs_dict)
        query_obs = _model.preprocess_observation_prefix_postfix(
            None, query_obs, train=False, image_keys=list(query_obs.images.keys())
        )

        query_embeddings, query_input_mask, query_ar_mask, query_embed = (
            self.embed_inputs_and_query_embed(query_obs)
        )

        traj_latents, latent_mask, latent_ar_mask = self._build_traj_prefix(
            obs_dict["traj_state"], obs_dict["traj_top_emb"], obs_dict["traj_wrist_emb"],
            obs_dict["traj_mask"], obs_dict["query_dino_top_emb"],
        )

        prefix_embeddings = jnp.concatenate([traj_latents, query_embeddings], axis=1)
        prefix_input_mask = jnp.concatenate([latent_mask, query_input_mask], axis=1)
        prefix_ar_mask = jnp.concatenate([latent_ar_mask, query_ar_mask], axis=1)

        # max_decoding_steps = postfix length (FASTTokenizerRicl pads prefix to max_token_len // 2)
        max_decoding_steps = self.max_token_len // 2
        prefill_size = prefix_embeddings.shape[1]
        batch_size = prefix_embeddings.shape[0]

        prefix_attn_mask = make_attn_mask(prefix_input_mask, prefix_ar_mask)
        prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps)))

        prefill_len = jnp.sum(prefix_input_mask, axis=-1)
        prefix_start = prefill_size - prefill_len
        prefix_positions = jnp.cumsum(prefix_input_mask, axis=-1) - 1

        prefix_logits, kv_cache, _ = self.PaliGemma.llm(
            embedded_prefix=prefix_embeddings,
            mask=prefix_attn_mask,
            positions=prefix_positions,
            decode=True,
        )

        last_logit = prefix_logits[:, -1:]
        original_dtype = last_logit.dtype
        output_tokens = jnp.zeros((batch_size, max_decoding_steps))

        def step(carry):
            last_logit, output_tokens, cache, _, step_idx = carry

            if temperature > 0.0:
                token = jax.random.categorical(rng, last_logit / temperature, axis=-1)
            else:
                token = jnp.argmax(last_logit, axis=-1)

            output_tokens = put_along_last_axis(
                output_tokens,
                jnp.broadcast_to(step_idx, (token.shape[0], 1)),
                token,
            )

            has_eos = jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=-1)
            all_eos = jnp.all(has_eos)

            token_embedding = self.PaliGemma.llm(token, embed_only=True)
            positions = prefill_len[:, None] + step_idx + 1
            mask = jnp.logical_and(
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :] >= prefix_start[:, None, None],
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :]
                < jnp.broadcast_to(prefill_size + step_idx + 1, (prefix_start.shape[0], 1, 1)),
            )
            last_logit, kv_cache, _ = self.PaliGemma.llm(
                embedded_prefix=token_embedding,
                mask=mask,
                positions=positions,
                decode=True,
                kv_cache=cache,
            )
            return last_logit.astype(original_dtype), output_tokens, kv_cache, all_eos, step_idx + 1

        def cond(carry):
            _, _, _, all_eos, step_idx = carry
            return (~all_eos) & (step_idx < max_decoding_steps)

        _, output_tokens, _, _, _ = jax.lax.while_loop(
            cond, step, (last_logit, output_tokens, kv_cache, False, 0)
        )
        return output_tokens

    def get_perceiver_attn_weights(self, obs_dict: dict) -> jnp.ndarray:
        """Return perceiver attention weights for diagnostics.

        Returns avg-over-layers, avg-over-heads attention: [B, num_latents, T]
        where T = max_traj_len (padded positions have near-zero weight due to masking).
        """
        traj_input = jnp.concatenate(
            [obs_dict["traj_top_emb"], obs_dict["traj_wrist_emb"], obs_dict["traj_state"]], axis=-1
        )
        traj_tokens = self.traj_proj(traj_input)
        query_embed = self.query_dino_proj(obs_dict["query_dino_top_emb"])
        _, attn_weights = self.perceiver(
            traj_tokens, query_embed, traj_mask=obs_dict["traj_mask"], return_attn_weights=True
        )
        return attn_weights  # [B, num_latents, T]
