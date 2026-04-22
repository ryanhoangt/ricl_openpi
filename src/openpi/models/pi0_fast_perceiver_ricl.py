"""Pi0-FAST with Perceiver Resampler for RICL (Retrieval In-Context Learning).

Instead of brittle action interpolation, each retrieved (obs, action) sample is
compressed into `num_latents` fixed-size latent tokens via a PerceiverResampler.
These latents are prepended as a soft, bidirectional prefix before the query
observation, and actions are generated autoregressively from the query only.

Context layout at training / inference:
    [r0_latents × K | r1_latents × K | ... | rN_latents × K |
     q_images (patches) | q_prefix tokens | q_action tokens →]

Token budget: N × K retrieval tokens + ~max_token_len query tokens,
vs. N × max_token_len in the original Pi0FASTRicl.
"""

from __future__ import annotations

import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma_fast as _gemma
from openpi.models.perceiver_resampler import PerceiverResampler
import openpi.models.siglip as _siglip
from openpi.models.pi0_fast_ricl import make_attn_mask, put_along_last_axis
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")

PALIGEMMA_EOS_TOKEN = 1


@dataclasses.dataclass(frozen=True)
class Pi0FASTPerceiverRiclConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"

    # Model-specific defaults.
    action_dim: int = 32
    action_horizon: int = 32
    max_token_len: int = 250
    num_retrieved_observations: int = 4
    lamda: float = 10.0

    # Number of latent tokens the PerceiverResampler compresses each retrieved
    # sample into.  32 gives a ~10-16× compression of the image patches while
    # retaining enough capacity for (visual context + state + action) information.
    num_latents: int = 32

    # Keep True so that RiclLiberoDataset loads exp_lamda_distances into the batch.
    use_action_interpolation: bool = True

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_FAST

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0FASTPerceiverRicl":
        return Pi0FASTPerceiverRicl(self, rngs=nnx.Rngs(rng))

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
        action_spec = jax.ShapeDtypeStruct(
            [batch_size, self.action_horizon, self.action_dim], jnp.float32
        )
        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        if "lora" in self.paligemma_variant:
            return nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*lora.*")))
        return nnx.Nothing

    def get_freeze_filter_with_frozen_img_encoder(self) -> nnx.filterlib.Filter:
        """Freeze the image encoder (and LLM except LoRA if applicable).

        Use this for the RICL perceiver fine-tune: keep the perceiver + LLM head
        trainable, freeze the heavy vision backbone.
        """
        if "lora" in self.paligemma_variant:
            return nnx.Any(
                nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*lora.*"))),
                nnx_utils.PathRegex(".*img.*"),
            )
        else:
            return nnx.All(
                nnx_utils.PathRegex(".*img.*"), nnx.Not(nnx_utils.PathRegex(".*llm.*"))
            )


class Pi0FASTPerceiverRicl(_model.BaseModel):
    def __init__(self, config: Pi0FASTPerceiverRiclConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        paligemma_config = _gemma.get_config(config.paligemma_variant)

        # ---- PaliGemma (SigLip image encoder + Gemma LLM) ----
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
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # ---- PerceiverResampler ----
        # dim must equal paligemma_config.width (2048 for gemma_2b) so that the
        # compressed latents live in the same embedding space as the LLM tokens —
        # no additional linear projection is required.
        perceiver = nnx_bridge.ToNNX(
            PerceiverResampler(
                dim=paligemma_config.width,
                num_latents=config.num_latents,
                depth=6,
                dim_head=64,
                heads=8,
            )
        )
        perceiver.lazy_init(
            jnp.zeros((1, 1, paligemma_config.width), dtype=jnp.float32), rngs=rngs
        )
        self.perceiver = perceiver

        self.num_retrieved_observations = config.num_retrieved_observations
        self.num_latents = config.num_latents
        self.max_token_len = config.max_token_len

    # ------------------------------------------------------------------
    # Core helper: compress one retrieved (obs, action) sample
    # ------------------------------------------------------------------

    def embed_retrieved_sample(
        self,
        obs: _model.ObservationPrefixPostfix,
    ) -> at.Float[at.Array, "b k d"]:
        """Compress a retrieved (obs, action) sample into num_latents latents.

        Feeds all modalities — image patches, discretised state (via prefix
        tokens), and action tokens (postfix) — into the PerceiverResampler so
        that it can distil the full (context → action) association.

        Args:
            obs: Pre-processed observation with both prefix and postfix populated.

        Returns:
            Latents of shape [B, num_latents, D].
        """
        token_parts = []
        mask_parts = []

        # 1. Image patches.
        #    SigLip with num_classes=paligemma_config.width already projects
        #    patch features to LLM embedding width — no further projection needed.
        for name in obs.images:
            img_patches, _ = self.PaliGemma.img(obs.images[name], train=False)  # [B, P, D]
            token_parts.append(img_patches)
            # Per-image mask → per-patch mask.
            patch_mask = einops.repeat(
                obs.image_masks[name].astype(jnp.bool_), "b -> b p", p=img_patches.shape[1]
            )  # [B, P]
            mask_parts.append(patch_mask)

        # 2. Text tokens: prefix ("Task: …, State: …;\n") + postfix ("Action: … |").
        assert obs.tokenized_prompt_prefix is not None, "retrieved obs must have prefix"
        assert obs.tokenized_prompt_postfix is not None, "retrieved obs must have postfix (actions)"
        full_tokens = jnp.concatenate(
            [obs.tokenized_prompt_prefix, obs.tokenized_prompt_postfix], axis=1
        )  # [B, max_token_len]
        text_embeds = self.PaliGemma.llm(full_tokens, embed_only=True)  # [B, max_token_len, D]
        token_parts.append(text_embeds)
        mask_parts.append(obs.tokenized_prompt_mask.astype(jnp.bool_))  # [B, max_token_len]

        # 3. Concatenate and compress via PerceiverResampler.
        all_tokens = jnp.concatenate(token_parts, axis=1)  # [B, N_total, D]
        token_mask = jnp.concatenate(mask_parts, axis=1)   # [B, N_total]

        latents = self.perceiver(all_tokens, mask=token_mask)  # [B, num_latents, D]
        return latents

    # ------------------------------------------------------------------
    # Embed query observation into a flat (embeddings, input_mask, ar_mask)
    # ------------------------------------------------------------------

    def embed_inputs(
        self,
        obs: _model.ObservationPrefixPostfix,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Int[at.Array, "b s"],
    ]:
        """Embed query observation images + text tokens into a flat sequence.

        At training time obs has both prefix and postfix.
        At inference time obs has only prefix (postfix=None).
        """
        input_mask = []
        ar_mask = []
        token_embeddings = []

        for name in obs.images:
            image_token_embeddings, _ = self.PaliGemma.img(obs.images[name], train=False)
            token_embeddings.append(image_token_embeddings)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name], "b -> b s", s=image_token_embeddings.shape[1]
                )
            )
            ar_mask.append(0 * input_mask[-1])  # images attend bidirectionally

        assert obs.tokenized_prompt_prefix is not None
        assert obs.tokenized_prompt_mask is not None
        assert obs.token_ar_mask is not None

        if obs.tokenized_prompt_postfix is not None:
            tokenized_inputs_embeddings = self.PaliGemma.llm(
                jnp.concatenate(
                    [obs.tokenized_prompt_prefix, obs.tokenized_prompt_postfix], axis=1
                ),
                embed_only=True,
            )
        else:
            tokenized_inputs_embeddings = self.PaliGemma.llm(
                obs.tokenized_prompt_prefix, embed_only=True
            )

        token_embeddings.append(tokenized_inputs_embeddings)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask.append(obs.token_ar_mask)

        return (
            jnp.concatenate(token_embeddings, axis=1),
            jnp.concatenate(input_mask, axis=1),
            jnp.concatenate(ar_mask, axis=1),
        )

    # ------------------------------------------------------------------
    # Build retrieval prefix (shared between train and inference)
    # ------------------------------------------------------------------

    def _build_retrieval_prefix(
        self,
        ricl_observation: _model.RiclObservation,
        rng: at.KeyArrayLike | None,
        train: bool,
    ) -> tuple[
        at.Float[at.Array, "b n d"],
        at.Bool[at.Array, "b n"],
        at.Int[at.Array, "b n"],
    ]:
        """Compress all retrieved samples and return (latents, input_mask, ar_mask)."""
        retrieval_latent_parts = []

        for i in range(self.num_retrieved_observations):
            prefix = f"retrieved_{i}_"
            ret_obs = _model.extract_observation_from_ricl_observation(ricl_observation, prefix)
            ret_obs = _model.preprocess_observation_prefix_postfix(
                rng, ret_obs, train=train, image_keys=list(ret_obs.images.keys())
            )
            latents = self.embed_retrieved_sample(ret_obs)  # [B, K, D]
            retrieval_latent_parts.append(latents)

        retrieval_latents = jnp.concatenate(retrieval_latent_parts, axis=1)  # [B, N*K, D]
        batch_size, n_retrieval_tokens, _ = retrieval_latents.shape

        # Retrieval latents form a fully-bidirectional prefix (ar_mask=0, all valid).
        retrieval_input_mask = jnp.ones((batch_size, n_retrieval_tokens), dtype=jnp.bool_)
        retrieval_ar_mask = jnp.zeros((batch_size, n_retrieval_tokens), dtype=jnp.int32)

        return retrieval_latents, retrieval_input_mask, retrieval_ar_mask

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        ricl_observation: _model.RiclObservation,
        actions: _model.Actions,
        *,
        train: bool = False,
        decode_indices: at.Int[at.Array, ""] = None,
    ) -> at.Float[at.Array, "*b ah"]:
        # 1. Compress retrieved samples into latents.
        retrieval_latents, retrieval_input_mask, retrieval_ar_mask = (
            self._build_retrieval_prefix(ricl_observation, rng, train)
        )

        # 2. Embed query observation (images + prefix + postfix).
        query_obs = _model.extract_observation_from_ricl_observation(ricl_observation, "query_")
        query_obs = _model.preprocess_observation_prefix_postfix(
            rng, query_obs, train=train, image_keys=list(query_obs.images.keys())
        )
        query_embeddings, query_input_mask, query_ar_mask = self.embed_inputs(query_obs)

        # 3. Combine: [retrieval_latents | query_tokens].
        full_embeddings = jnp.concatenate([retrieval_latents, query_embeddings], axis=1)
        full_input_mask = jnp.concatenate([retrieval_input_mask, query_input_mask], axis=1)
        full_ar_mask = jnp.concatenate([retrieval_ar_mask, query_ar_mask], axis=1)
        full_attn_mask = make_attn_mask(full_input_mask, full_ar_mask)

        # 4. Loss targets: next-token prediction over query text tokens only.
        #    targets  = [query_prefix[1:] | query_postfix]   shape [B, max_token_len-1]
        #    loss_mask selects only the postfix (action) token positions.
        loss_mask = query_obs.token_loss_mask[:, 1:]
        targets = jax.nn.one_hot(
            jnp.concatenate(
                [
                    query_obs.tokenized_prompt_prefix[:, 1:],
                    query_obs.tokenized_prompt_postfix,
                ],
                axis=1,
            ),
            self.PaliGemma.llm.module.vocab_size,
        )

        # 5. Single forward pass — skip the last token since it predicts nothing here.
        pre_logits, _, _ = self.PaliGemma.llm(
            embedded_prefix=full_embeddings[:, :-1],
            mask=full_attn_mask[:, :-1, :-1],
            return_prelogits=True,
        )

        # Decode only the last targets.shape[1] pre-logits to keep the vocab
        # projection memory-efficient.
        logits, _ = self.PaliGemma.llm(
            pre_logits=pre_logits[:, -targets.shape[1]:],
        )

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
        ricl_observation: _model.RiclObservation,
        *,
        temperature: float = 0.0,
    ) -> _model.Actions:
        # 1. Compress retrieved samples.
        retrieval_latents, retrieval_input_mask, retrieval_ar_mask = (
            self._build_retrieval_prefix(ricl_observation, rng=None, train=False)
        )

        # 2. Embed query prefix (no postfix at inference time).
        query_obs = _model.extract_observation_from_ricl_observation(ricl_observation, "query_")
        query_obs = _model.preprocess_observation_prefix_postfix(
            None, query_obs, train=False, image_keys=list(query_obs.images.keys())
        )
        query_embeddings, query_input_mask, query_ar_mask = self.embed_inputs(query_obs)

        # 3. Build full prefix: [retrieval_latents | query_prefix].
        prefix_embeddings = jnp.concatenate([retrieval_latents, query_embeddings], axis=1)
        prefix_input_mask = jnp.concatenate([retrieval_input_mask, query_input_mask], axis=1)
        prefix_ar_mask = jnp.concatenate([retrieval_ar_mask, query_ar_mask], axis=1)

        # max_decoding_steps = postfix length as defined by FASTTokenizerRicl
        # (i.e., max_token_len // 2 action tokens to generate).
        max_decoding_steps = self.max_token_len // 2
        prefill_size = prefix_embeddings.shape[1]
        batch_size = prefix_embeddings.shape[0]

        prefix_attn_mask = make_attn_mask(prefix_input_mask, prefix_ar_mask)
        # Pad the KV-cache dimension to hold the generated tokens.
        prefix_attn_mask = jnp.pad(
            prefix_attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps))
        )

        prefill_len = jnp.sum(prefix_input_mask, axis=-1)
        prefix_start = prefill_size - prefill_len
        prefix_positions = jnp.cumsum(prefix_input_mask, axis=-1) - 1

        # 4. Prefill KV cache with the full prefix.
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

            # Sample or greedily decode one token.
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

            # Decode one step using the KV cache.
            token_embedding = self.PaliGemma.llm(token, embed_only=True)
            positions = prefill_len[:, None] + step_idx + 1
            mask = jnp.logical_and(
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :]
                >= prefix_start[:, None, None],
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :]
                < jnp.broadcast_to(
                    prefill_size + step_idx + 1, (prefix_start.shape[0], 1, 1)
                ),
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
