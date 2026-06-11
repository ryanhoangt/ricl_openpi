"""Implicit cross-context reasoning on top of RICL-Pi0-FAST.

Extends `Pi0FASTRicl` with:
  * K learnable *reasoning tokens* `R` inserted between the query prefix (state + language)
    and the query action postfix. Under the existing block-diagonal + causal attention, `R`
    attends to the full retrieved context and the query, and the autoregressive action tokens
    attend back to `R`.
  * An asymmetric SAM-mask *information bottleneck*: retrieved top-camera (`base_0_rgb`) patches
    are flagged with a learnable `seg_flag_embedding` where the retrieved object mask is active,
    and the reasoning-token hidden states `H_R` must reconstruct the (unflagged) query object mask
    via a small cross-attention projector. Supervised with focal + soft-dice loss at patch
    resolution. Total loss = action CE + lambda_scene * scene loss.

At inference the projector + SAM masks are dropped; only `R` is kept (≈K extra prefill tokens).
See plan: .claude/plans/radiant-wobbling-kahan.md
"""

import dataclasses
import logging

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_fast_ricl as _pi0_fast_ricl
import openpi.models.gemma_fast as _gemma
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

# Top camera image key (the only camera with SAM masks). Must match libero_policy.RiclLiberoInputs.
TOP_CAMERA_KEY = "base_0_rgb"


@dataclasses.dataclass(frozen=True)
class Pi0FASTReasoningRiclConfig(_pi0_fast_ricl.Pi0FASTRiclConfig):
    # Number of learnable reasoning tokens appended after the query prefix.
    num_reasoning_tokens: int = 16
    # Weight on the SAM-mask reconstruction (scene) loss.
    lambda_scene: float = 1.0
    # Hidden width of the segmentation projector (d_v).
    seg_hidden_dim: int = 512
    # Number of attention heads in the segmentation cross-attention projector.
    num_seg_heads: int = 8
    # Number of patches the SAM mask is reduced to (SigLIP So400m/14 @224 -> 16x16 = 256).
    num_seg_patches: int = 256
    # Ablation: if False, retrieved patches are NOT flagged (tests whether context is used).
    use_seg_flag: bool = True
    # Ablation: probability of dropping query top-camera patch embeddings for the reasoning
    # forward pass, forcing H_R toward the retrieved correspondence instead of the query shortcut.
    query_patch_dropout: float = 0.0
    # Focal loss hyperparameters.
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    # Joint geometric augmentation of (top image, SAM mask) in the training dataset. Decorrelates
    # query vs retrieved object positions so the model can't reconstruct the query mask by copying
    # the retrieved flag. Read by RiclReasoningLiberoDataset (train-only). Replaces the disabled
    # in-model geometric aug while keeping masks pixel-aligned with the augmented image.
    joint_mask_aug: bool = False
    aug_crop_scale: float = 0.95
    aug_rotate_deg: float = 5.0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0FASTReasoningRicl":
        return Pi0FASTReasoningRicl(self, rngs=nnx.Rngs(rng))


class SegProjector(nnx.Module):
    """Unpacks compressed reasoning-token hidden states into a per-patch mask logit grid.

    M_hat = MLP(CrossAttn(Q_sp, P(H_R), P(H_R))) -> (b, num_patches) logits (sigmoid applied in loss).
    """

    def __init__(self, *, llm_width: int, d_v: int, num_patches: int, num_heads: int, rngs: nnx.Rngs):
        self.num_patches = num_patches
        self.d_v = d_v
        # Linear projection P aligning LLM hidden dim -> visual dim d_v.
        self.proj_in = nnx.Linear(llm_width, d_v, rngs=rngs)
        # Learnable spatial query vectors Q_sp (num_patches, d_v).
        self.spatial_queries = nnx.Param(jax.random.normal(rngs.params(), (num_patches, d_v)) * 0.02)
        self.cross_attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=d_v,
            decode=False,
            rngs=rngs,
        )
        self.mlp_1 = nnx.Linear(d_v, d_v, rngs=rngs)
        self.mlp_2 = nnx.Linear(d_v, 1, rngs=rngs)

    def __call__(self, h_r: at.Float[at.Array, "b k w"]) -> at.Float[at.Array, "b p"]:
        b = h_r.shape[0]
        kv = self.proj_in(h_r.astype(jnp.float32))  # (b, K, d_v)
        q = jnp.broadcast_to(self.spatial_queries.value[None], (b, self.num_patches, self.d_v))
        attended = self.cross_attn(q, kv)  # (b, P, d_v)
        x = jax.nn.gelu(self.mlp_1(attended))
        logits = self.mlp_2(x)[..., 0]  # (b, P)
        return logits


class Pi0FASTReasoningRicl(_pi0_fast_ricl.Pi0FASTRicl):
    def __init__(self, config: Pi0FASTReasoningRiclConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        width = paligemma_config.width

        self.num_reasoning_tokens = config.num_reasoning_tokens
        self.lambda_scene = config.lambda_scene
        self.num_seg_patches = config.num_seg_patches
        self.use_seg_flag = config.use_seg_flag
        self.query_patch_dropout = config.query_patch_dropout
        self.focal_alpha = config.focal_alpha
        self.focal_gamma = config.focal_gamma

        # K learnable reasoning token embeddings (small random init).
        self.reasoning_tokens = nnx.Param(
            jax.random.normal(rngs.params(), (config.num_reasoning_tokens, width)) * 0.02
        )
        # Learnable flag embedding added to flagged retrieved top-camera patches.
        # Zero-initialised so it is a no-op at the start of training.
        self.seg_flag_embedding = nnx.Param(jnp.zeros((width,), dtype=jnp.float32))
        # Cross-attention projector that reconstructs the query mask from H_R.
        self.seg_projector = SegProjector(
            llm_width=width,
            d_v=config.seg_hidden_dim,
            num_patches=config.num_seg_patches,
            num_heads=config.num_seg_heads,
            rngs=rngs,
        )

    # ------------------------------------------------------------------ embedding helpers

    def _embed_observation(
        self,
        obs: _model.ObservationPrefixPostfix,
        *,
        flag_mask: at.Float[at.Array, "b p"] | None = None,
        append_reasoning: bool = False,
        patch_dropout_rng: at.KeyArrayLike | None = None,
    ):
        """Embed one observation block.

        * For retrieved blocks, `flag_mask` (per-patch fraction) adds `seg_flag_embedding` to the
          flagged `base_0_rgb` patches (no-op when None or when use_seg_flag is False).
        * For the query block, `append_reasoning=True` inserts the K reasoning tokens between the
          embedded prefix and postfix, updating the input/AR masks accordingly.

        Returns (token_embeddings (b, s, w), input_mask (b, s), ar_mask (b, s)).
        """
        input_mask = []
        ar_mask = []
        token_embeddings = []

        for name in obs.images:
            image_token_embeddings, _ = self.PaliGemma.img(obs.images[name], train=False)
            if name == TOP_CAMERA_KEY:
                num_patches = image_token_embeddings.shape[1]
                assert num_patches == self.num_seg_patches, (
                    f"top-camera patch count {num_patches} != num_seg_patches {self.num_seg_patches}"
                )
                if flag_mask is not None and self.use_seg_flag:
                    flag = flag_mask.astype(image_token_embeddings.dtype)  # (b, P)
                    image_token_embeddings = image_token_embeddings + (
                        flag[:, :, None] * self.seg_flag_embedding.value.astype(image_token_embeddings.dtype)[None, None, :]
                    )
                if append_reasoning and patch_dropout_rng is not None and self.query_patch_dropout > 0.0:
                    keep = jax.random.bernoulli(
                        patch_dropout_rng, p=1.0 - self.query_patch_dropout, shape=(image_token_embeddings.shape[0], num_patches)
                    ).astype(image_token_embeddings.dtype)
                    image_token_embeddings = image_token_embeddings * keep[:, :, None]
            token_embeddings.append(image_token_embeddings)
            input_mask.append(
                einops.repeat(obs.image_masks[name], "b -> b s", s=image_token_embeddings.shape[1])
            )
            ar_mask.append(0 * input_mask[-1])

        assert obs.tokenized_prompt_prefix is not None, "Tokenized prompt prefix is required"
        assert obs.tokenized_prompt_mask is not None, "Tokenized prompt mask is required"
        assert obs.token_ar_mask is not None, "Token auto-regressive mask is required"

        prefix_len = obs.tokenized_prompt_prefix.shape[1]
        prefix_embeddings = self.PaliGemma.llm(obs.tokenized_prompt_prefix, embed_only=True)
        prefix_input_mask = obs.tokenized_prompt_mask[:, :prefix_len]
        prefix_ar_mask = obs.token_ar_mask[:, :prefix_len]

        if append_reasoning:
            batch_size = prefix_embeddings.shape[0]
            reasoning = jnp.broadcast_to(
                self.reasoning_tokens.value.astype(prefix_embeddings.dtype)[None],
                (batch_size, self.num_reasoning_tokens, prefix_embeddings.shape[-1]),
            )
            reasoning_input_mask = jnp.ones((batch_size, self.num_reasoning_tokens), dtype=prefix_input_mask.dtype)
            reasoning_ar_mask = jnp.ones((batch_size, self.num_reasoning_tokens), dtype=prefix_ar_mask.dtype)

            token_embeddings.append(prefix_embeddings)
            input_mask.append(prefix_input_mask)
            ar_mask.append(prefix_ar_mask)
            token_embeddings.append(reasoning)
            input_mask.append(reasoning_input_mask)
            ar_mask.append(reasoning_ar_mask)

            if obs.tokenized_prompt_postfix is not None:
                postfix_embeddings = self.PaliGemma.llm(obs.tokenized_prompt_postfix, embed_only=True)
                token_embeddings.append(postfix_embeddings)
                input_mask.append(obs.tokenized_prompt_mask[:, prefix_len:])
                ar_mask.append(obs.token_ar_mask[:, prefix_len:])
        else:
            # Retrieved block: embed the full prefix+postfix prompt as in the base model.
            if obs.tokenized_prompt_postfix is not None:
                prompt_tokens = jnp.concatenate(
                    [obs.tokenized_prompt_prefix, obs.tokenized_prompt_postfix], axis=1
                )
            else:
                prompt_tokens = obs.tokenized_prompt_prefix
            prompt_embeddings = self.PaliGemma.llm(prompt_tokens, embed_only=True)
            token_embeddings.append(prompt_embeddings)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask.append(obs.token_ar_mask)

        return (
            jnp.concatenate(token_embeddings, axis=1),
            jnp.concatenate(input_mask, axis=1),
            jnp.concatenate(ar_mask, axis=1),
        )

    def combine_attn_masks_varlen(self, list_of_attn_masks, batch_size, block_lens):
        """Block-diagonal mask over (possibly unequal) per-block lengths, OR'd with a global tril."""
        seq_len = int(sum(block_lens))
        attn_mask = jnp.zeros((batch_size, seq_len, seq_len), dtype=bool)
        start = 0
        for i, bl in enumerate(block_lens):
            attn_mask = attn_mask.at[:, start : start + bl, start : start + bl].set(list_of_attn_masks[i])
            start += bl
        attn_mask = jnp.logical_or(attn_mask, jnp.tril(jnp.ones((batch_size, seq_len, seq_len), dtype=bool)))
        return attn_mask

    def _build_training_sequence(self, rng, ricl_observation, *, train):
        """Builds the full input sequence [retrieved blocks..., query block(+R+postfix)].

        Returns (input_token_embeddings, attn_mask, query_obs, postfix_len, first_targets).
        """
        num_observations = self.num_retrieved_observations + 1
        list_of_embeddings = []
        list_of_attn_masks = []
        block_lens = []
        first_targets = None
        query_obs = None
        dropout_rng = None
        if rng is not None and self.query_patch_dropout > 0.0:
            rng, dropout_rng = jax.random.split(rng)

        for i in range(num_observations):
            is_query = i == self.num_retrieved_observations
            prefix = "query_" if is_query else f"retrieved_{i}_"
            this_obs = _model.extract_observation_from_ricl_observation(ricl_observation, prefix)
            this_obs = _model.preprocess_observation_prefix_postfix(
                rng, this_obs, train=train, image_keys=list(this_obs.images.keys()), disable_geom_aug=True
            )

            flag_mask = None if is_query else getattr(ricl_observation, f"retrieved_{i}_flag_mask")
            emb, msk, ar = self._embed_observation(
                this_obs,
                flag_mask=flag_mask,
                append_reasoning=is_query,
                patch_dropout_rng=dropout_rng if is_query else None,
            )
            list_of_embeddings.append(emb)
            list_of_attn_masks.append(_pi0_fast_ricl.make_attn_mask(msk, ar))
            block_lens.append(emb.shape[1])

            if i == 0 and self.use_action_interpolation:
                first_targets = jax.nn.one_hot(
                    this_obs.tokenized_prompt_postfix, self.PaliGemma.llm.module.vocab_size
                )
            if is_query:
                query_obs = this_obs

        input_token_embeddings = jnp.concatenate(list_of_embeddings, axis=1)
        batch_size = input_token_embeddings.shape[0]
        attn_mask = self.combine_attn_masks_varlen(list_of_attn_masks, batch_size, block_lens)
        postfix_len = query_obs.tokenized_prompt_postfix.shape[1]
        return input_token_embeddings, attn_mask, query_obs, postfix_len, first_targets

    # ------------------------------------------------------------------ losses

    def _seg_loss(self, seg_logits, seg_target):
        """Focal + soft-dice loss between per-patch logits and soft target fractions. Returns (b,)."""
        eps = 1e-6
        p = jax.nn.sigmoid(seg_logits.astype(jnp.float32))  # (b, P)
        t = seg_target.astype(jnp.float32)

        p_c = jnp.clip(p, eps, 1.0 - eps)
        # Focal loss with soft targets.
        focal_pos = self.focal_alpha * jnp.power(1.0 - p_c, self.focal_gamma) * t * -jnp.log(p_c)
        focal_neg = (1.0 - self.focal_alpha) * jnp.power(p_c, self.focal_gamma) * (1.0 - t) * -jnp.log(1.0 - p_c)
        focal = jnp.mean(focal_pos + focal_neg, axis=-1)  # (b,)

        # Soft dice loss.
        inter = jnp.sum(p * t, axis=-1)
        dice = 1.0 - (2.0 * inter + eps) / (jnp.sum(p, axis=-1) + jnp.sum(t, axis=-1) + eps)  # (b,)
        return focal + dice

    def compute_loss_with_aux(
        self,
        rng: at.KeyArrayLike,
        ricl_observation: _model.RiclObservation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ):
        """Returns (per-example total loss (b,), aux dict with scalar components + seg_pred)."""
        input_token_embeddings, attn_mask, query_obs, postfix_len, first_targets = self._build_training_sequence(
            rng, ricl_observation, train=train
        )
        seq_len = input_token_embeddings.shape[1]

        # One forward pass; keep the last-layer hidden states (pre-logits).
        pre_logits, _, _ = self.PaliGemma.llm(
            embedded_prefix=input_token_embeddings[:, :-1],
            mask=attn_mask[:, :-1, :-1],
            return_prelogits=True,
        )

        # Action CE loss on the query postfix only.
        # pre_logits[:, -postfix_len:] are the hidden states that predict the postfix tokens.
        action_pre_logits = pre_logits[:, -postfix_len:]
        logits, _ = self.PaliGemma.llm(pre_logits=action_pre_logits)
        targets = jax.nn.one_hot(query_obs.tokenized_prompt_postfix, self.PaliGemma.llm.module.vocab_size)
        loss_mask = query_obs.token_loss_mask[:, -postfix_len:]

        if self.use_action_interpolation:
            exp_lamda = ricl_observation.exp_lamda_distances[:, -1:, :]  # (b, 1, 1)
            new_logits = exp_lamda * first_targets + (1.0 - exp_lamda) * jax.nn.softmax(logits, axis=-1)
            epsilon = 1e-9
            logp = jnp.log(jnp.clip(new_logits, epsilon, 1.0 - epsilon))
        else:
            logp = jax.nn.log_softmax(logits, axis=-1)

        token_pplx = jnp.sum(targets * logp, axis=-1)
        ce_loss = -jnp.sum(token_pplx * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, -1), 1)  # (b,)

        # Scene (SAM-mask) loss from the reasoning-token hidden states.
        r_start = seq_len - postfix_len - self.num_reasoning_tokens
        h_r = pre_logits[:, r_start : r_start + self.num_reasoning_tokens, :]  # (b, K, w)
        seg_logits = self.seg_projector(h_r)  # (b, P)
        seg_loss = self._seg_loss(seg_logits, ricl_observation.query_seg_target)  # (b,)

        total = ce_loss + self.lambda_scene * seg_loss
        aux = {
            "ce_loss": jnp.mean(ce_loss),
            "seg_loss": jnp.mean(seg_loss),
            "seg_pred": jax.nn.sigmoid(seg_logits.astype(jnp.float32)),
        }
        return total, aux

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
        loss, _ = self.compute_loss_with_aux(rng, ricl_observation, actions, train=train)
        return loss

    def predict_seg_mask(self, ricl_observation: _model.RiclObservation) -> at.Float[at.Array, "b p"]:
        """Inference-only reconstruction of the query mask (for visualization). Returns sigmoid probs."""
        input_token_embeddings, attn_mask, _, postfix_len, _ = self._build_training_sequence(
            None, ricl_observation, train=False
        )
        seq_len = input_token_embeddings.shape[1]
        pre_logits, _, _ = self.PaliGemma.llm(
            embedded_prefix=input_token_embeddings[:, :-1],
            mask=attn_mask[:, :-1, :-1],
            return_prelogits=True,
        )
        r_start = seq_len - postfix_len - self.num_reasoning_tokens
        h_r = pre_logits[:, r_start : r_start + self.num_reasoning_tokens, :]
        return jax.nn.sigmoid(self.seg_projector(h_r).astype(jnp.float32))

    # ------------------------------------------------------------------ inference

    @override
    def sample_actions_multiple_observations_processing(self, ricl_observation: _model.RiclObservation):
        """Like the base method, but flags retrieved top patches and appends reasoning tokens to the
        query block. The decode loop in `sample_actions` (inherited) is unchanged."""
        num_observations = self.num_retrieved_observations + 1
        list_of_prefix_token_embeddings = []
        list_of_prefix_attn_masks = []
        block_lens = []
        first_targets = None
        retrieval_block_len = None
        query_block_len = None
        max_decoding_steps = None
        query_prompt_len = None

        for i in range(num_observations):
            is_query = i == self.num_retrieved_observations
            prefix = "query_" if is_query else f"retrieved_{i}_"
            this_obs = _model.extract_observation_from_ricl_observation(ricl_observation, prefix)
            this_obs = _model.preprocess_observation_prefix_postfix(
                None, this_obs, train=False, image_keys=list(this_obs.images.keys()), disable_geom_aug=True
            )

            flag_mask = None if is_query else getattr(ricl_observation, f"retrieved_{i}_flag_mask")
            emb, msk, ar = self._embed_observation(this_obs, flag_mask=flag_mask, append_reasoning=is_query)
            list_of_prefix_token_embeddings.append(emb)
            list_of_prefix_attn_masks.append(_pi0_fast_ricl.make_attn_mask(msk, ar))
            block_lens.append(emb.shape[1])

            if i == 0:
                retrieval_prompt_len = (
                    this_obs.tokenized_prompt_prefix.shape[1] + this_obs.tokenized_prompt_postfix.shape[1]
                )
                retrieval_block_len = emb.shape[1]
                if self.use_action_interpolation:
                    first_targets = jax.nn.one_hot(
                        this_obs.tokenized_prompt_postfix, self.PaliGemma.llm.module.vocab_size
                    )
            if is_query:
                if this_obs.tokenized_prompt_postfix is not None:
                    query_prompt_len = (
                        this_obs.tokenized_prompt_prefix.shape[1] + this_obs.tokenized_prompt_postfix.shape[1]
                    )
                else:
                    query_prompt_len = this_obs.tokenized_prompt_prefix.shape[1]
                query_block_len = emb.shape[1]
                max_decoding_steps = retrieval_prompt_len - query_prompt_len

        prefix_token_embeddings = jnp.concatenate(list_of_prefix_token_embeddings, axis=1)
        batch_size, seq_len = prefix_token_embeddings.shape[0:2]
        prefix_attn_mask = self.combine_attn_masks_inference_time(
            list_of_prefix_attn_masks, batch_size, seq_len, num_observations, retrieval_block_len, query_block_len
        )
        prefix_mask = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)
        return (
            prefix_token_embeddings,
            prefix_attn_mask,
            first_targets,
            max_decoding_steps,
            query_prompt_len,
            batch_size,
            prefix_mask,
        )
