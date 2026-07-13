from collections.abc import Sequence
import logging
import pathlib
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_fast_ricl as _pi0_fast_ricl
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi.policies.utils import embed, embed_with_batches, load_dinov2, EMBED_DIM
import os
from autofaiss import build_index
import logging
from datetime import datetime
import json
from PIL import Image
logger = logging.getLogger()
BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._model = model

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        self._rng, sample_rng = jax.random.split(self._rng)
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), **self._sample_kwargs),
        }

        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        print(f'outputs: {outputs}')
        final_outputs = self._output_transform(outputs)
        logger.info(f'final_outputs: {final_outputs}')
        return final_outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata
    

def get_action_chunk_at_inference_time(actions, step_idx, action_horizon):
    num_steps = len(actions)
    action_dim = actions.shape[-1]
    action_chunk = []
    for i in range(action_horizon):
        if step_idx + i < num_steps:
            action_chunk.append(actions[step_idx + i])
        else:
            action_chunk.append(
                np.concatenate([np.zeros(action_dim - 1, dtype=np.float32), actions[-1, -1:]], axis=0)
            )  # combines 0 joint vels with last gripper pos
    action_chunk = np.stack(action_chunk, axis=0)
    assert action_chunk.shape == (action_horizon, action_dim), f"{action_chunk.shape=}"
    return action_chunk


def _build_seg_debug_panel(query_img_u8, nn_imgs_u8, nn_masks, pred_grid, sims, slots=None):
    """Debug panel: row of [NN_i top img + SAM mask (yellow), labeled] over [query | query+pred (red)].

    query_img_u8/nn_imgs_u8[i]: (H,W,3) uint8; nn_masks[i]: (H,W) bool or None; pred_grid: (g,g) in
    [0,1]; sims: per-observation exp(-lambda*dist) closeness; slots: optional (k,2) (ep,step) labels.
    Returns a single (2H, k*W, 3) uint8 image.
    """
    from PIL import Image, ImageDraw

    query_img_u8 = np.asarray(query_img_u8)
    h, w = query_img_u8.shape[:2]

    def overlay(rgb_u8, m, color, alpha=0.5):
        m = np.asarray(m, np.float32)
        if m.shape[:2] != (h, w):
            m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0
        out = rgb_u8.astype(np.float32) * (1 - alpha * m[..., None]) + np.asarray(color, np.float32) * (alpha * m[..., None])
        return out.clip(0, 255).astype(np.uint8)

    def label(arr, text):
        im = Image.fromarray(np.ascontiguousarray(arr))
        ImageDraw.Draw(im).text((3, 3), text, fill=(255, 255, 0))
        return np.asarray(im)

    nn_panels = []
    for i in range(len(nn_imgs_u8)):
        msk = np.zeros((h, w), np.float32) if nn_masks[i] is None else nn_masks[i]
        txt = f"NN{i} sim={sims[i]:.2f}" if i < len(sims) else f"NN{i}"
        if slots is not None and i < len(slots):
            txt += f" e{int(slots[i][0])}/{int(slots[i][1])}"
        nn_panels.append(label(overlay(np.asarray(nn_imgs_u8[i]), msk, (255, 255, 0)), txt))
    top = np.concatenate(nn_panels, axis=1)

    bottom = np.concatenate(
        [label(query_img_u8, "query"), label(overlay(query_img_u8, pred_grid, (255, 0, 0)), "query+pred")], axis=1
    )
    if bottom.shape[1] < top.shape[1]:
        bottom = np.concatenate([bottom, np.zeros((h, top.shape[1] - bottom.shape[1], 3), np.uint8)], axis=1)
    elif top.shape[1] < bottom.shape[1]:
        top = np.concatenate([top, np.zeros((h, bottom.shape[1] - top.shape[1], 3), np.uint8)], axis=1)
    return np.concatenate([top, bottom], axis=0)


class RiclPolicy(BasePolicy):
    def __init__(
        self,
        model: _pi0_fast_ricl.Pi0FASTRicl,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        demos_dir: str | None = None,
        use_action_interpolation: bool | None = None,
        lamda: float | None = None,
        action_horizon: int | None = None,
        max_distance_file: str = "assets/max_distance.json",
        ricl_step_offset: int = 0,
        record_debug: bool = False,
    ):
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        # Reasoning-RICL only: predict the query seg mask from the reasoning tokens for the debug video.
        self._record_debug = bool(record_debug)
        self._predict_seg_mask = (
            nnx_utils.module_jit(model.predict_seg_mask) if hasattr(model, "predict_seg_mask") else None
        )
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._model = model
        self._use_action_interpolation = use_action_interpolation
        self._lamda = lamda
        self._action_horizon = action_horizon
        # If > 0, retrieve only the top-1 NN and fill remaining slots with the same
        # trajectory at +ricl_step_offset, +2*ricl_step_offset, ... When a slot would
        # exceed the NN trajectory length, fall back to the next unused NN.
        self._ricl_step_offset = int(ricl_step_offset)
        # setup demos for retrieval
        print()
        logger.info(f'loading demos from {demos_dir}...')
        demo_entries = [(i, folder) for i, folder in enumerate(os.listdir(demos_dir)) if os.path.isdir(f"{demos_dir}/{folder}")]
        self._demos = {i: np.load(f"{demos_dir}/{folder}/processed_demo.npz") for i, folder in demo_entries}
        # Reasoning-RICL flags retrieved patches with offline SAM masks. The retrieved demos come from
        # this (labeled) priming buffer, so the flag is applied at inference too — matching training,
        # with zero live-SAM latency. Only the live query mask + projector are dropped at inference.
        self._attach_seg_masks = bool(getattr(model, "use_seg_flag", False))
        # >1 for the per-object (instance-level) reasoning model; 1 (or absent) means union masks.
        self._num_seg_objects = int(getattr(model, "num_seg_objects", 1))
        self._demo_sam_paths = {i: f"{demos_dir}/{folder}/sam_masks_top.npz" for i, folder in demo_entries}
        self._sam_union_cache: dict[int, np.ndarray] = {}
        self._sam_id_cache: dict[int, np.ndarray] = {}
        if self._attach_seg_masks:
            n_missing = sum(not os.path.exists(p) for p in self._demo_sam_paths.values())
            if n_missing:
                logger.warning(
                    f"{n_missing}/{len(self._demo_sam_paths)} retrieval demos lack sam_masks_top.npz; "
                    f"retrieved flags will be skipped for those (train/inference mismatch)."
                )
        self._all_indices = np.array([(ep_idx, step_idx) for ep_idx in list(self._demos.keys()) for step_idx in range(self._demos[ep_idx]["actions"].shape[0])])
        _all_embeddings = np.concatenate([self._demos[ep_idx]["top_image_embeddings"] for ep_idx in list(self._demos.keys())])
        assert _all_embeddings.shape == (len(self._all_indices), EMBED_DIM), f"{_all_embeddings.shape=}"
        self._knn_k = self._model.num_retrieved_observations
        print()
        logger.info(f'building retrieval index...')
        self._knn_index, knn_index_infos = build_index(embeddings=_all_embeddings, # Note: embeddings have to be float to avoid errors in autofaiss / embedding_reader!
                                            save_on_disk=False,
                                            min_nearest_neighbors_to_retrieve=max(self._knn_k + 5, 2 * self._knn_k), # default: 20; bumped to support ricl_step_offset fallback pool
                                            max_index_query_time_ms=10, # default: 10
                                            max_index_memory_usage="25G", # default: "16G"
                                            current_memory_available="50G", # default: "32G"
                                            metric_type='l2',
                                            nb_cores=8, # default: None # "The number of cores to use, by default will use all cores" as seen in https://criteo.github.io/autofaiss/getting_started/quantization.html#the-build-index-command
                                            )
        # setup the dinov2 model for embedding only
        logger.info('loading dinov2 for image embedding...')
        self._dinov2 = load_dinov2()
        self._max_dist = json.load(open(max_distance_file, 'r'))['distances']['max']
        print(f'self._max_dist: {self._max_dist} (from {max_distance_file}) [helpful to carefully check this value in case of any issues]')

    def _retrieved_seg_mask(self, ep_idx: int, step_idx: int) -> np.ndarray:
        """Retrieved SAM mask of a demo at a step, matching the model's flag layout:
          * union model (num_seg_objects == 1): (H, W) bool, OR over objects.
          * per-object model (num_seg_objects  > 1): (N, H, W) bool, scattered by global obj_id.
        Cached per demo."""
        if self._num_seg_objects > 1:
            if ep_idx not in self._sam_id_cache:
                d = np.load(self._demo_sam_paths[ep_idx])
                masks = d["masks"]  # (T, n_obj, H, W)
                obj_ids = d["obj_ids"]  # (n_obj,)
                t, _, h, w = masks.shape
                out = np.zeros((t, self._num_seg_objects, h, w), dtype=bool)
                for j, oid in enumerate(obj_ids):
                    oid = int(oid)
                    assert 0 <= oid < self._num_seg_objects, (
                        f"obj_id {oid} out of range [0, {self._num_seg_objects}) in {self._demo_sam_paths[ep_idx]}"
                    )
                    out[:, oid] = masks[:, j]
                self._sam_id_cache[ep_idx] = out  # (T, N, H, W) bool
            return self._sam_id_cache[ep_idx][step_idx]  # (N, H, W)
        if ep_idx not in self._sam_union_cache:
            d = np.load(self._demo_sam_paths[ep_idx])
            self._sam_union_cache[ep_idx] = np.any(d["masks"], axis=1)  # (T, H, W) bool
        return self._sam_union_cache[ep_idx][step_idx]

    def _ensure_query_keys(self, obs: dict) -> dict:
        if "query_top_image" not in obs:
            if "observation/image" in obs:
                obs["query_top_image"] = obs["observation/image"]
            elif "image" in obs:
                obs["query_top_image"] = obs["image"]
            elif "top_image" in obs:
                obs["query_top_image"] = obs["top_image"]

        if "query_wrist_image" not in obs:
            if "observation/wrist_image" in obs:
                obs["query_wrist_image"] = obs["observation/wrist_image"]
            elif "wrist_image" in obs:
                obs["query_wrist_image"] = obs["wrist_image"]

        if "query_state" not in obs:
            if "observation/state" in obs:
                obs["query_state"] = obs["observation/state"]
            elif "state" in obs:
                obs["query_state"] = obs["state"]

        if "query_prompt" not in obs and "prompt" in obs:
            obs["query_prompt"] = obs["prompt"]

        if "query_top_image" in obs and "query_right_image" not in obs:
            obs["query_right_image"] = np.zeros_like(obs["query_top_image"])

        return obs

    def retrieve(self, obs: dict) -> dict:
        more_obs = {"inference_time": True}
        obs = self._ensure_query_keys(obs)
        # embed
        query_embedding = embed(obs["query_top_image"], self._dinov2)
        assert query_embedding.shape == (1, EMBED_DIM), f"{query_embedding.shape=}"
        # retrieve a pool large enough to cover all fallback slots when ricl_step_offset is active
        pool_size = self._knn_k if self._ricl_step_offset == 0 else 2 * self._knn_k
        pool_size = min(pool_size, len(self._all_indices))
        _topk_distance, topk_indices = self._knn_index.search(query_embedding, pool_size)
        pool_indices = self._all_indices[topk_indices][0]  # (pool_size, 2)
        # Build the k slot list:
        #   - If ricl_step_offset == 0: original behavior (top-k NNs).
        #   - Else: slot 0 = top-1 NN at its step; slot j>=1 = top-1 NN at step + j*offset,
        #     falling back to the next unused NN in the pool when that step is out of range.
        slots = []  # list of (ep_idx, step_idx)
        if self._ricl_step_offset == 0:
            for ep_idx, step_idx in pool_indices[: self._knn_k]:
                slots.append((int(ep_idx), int(step_idx)))
        else:
            nn0_ep = int(pool_indices[0, 0])
            nn0_step = int(pool_indices[0, 1])
            nn0_traj_len = int(self._demos[nn0_ep]["actions"].shape[0])
            next_fallback = 1  # next unused NN in pool_indices (slot 0 consumed pool_indices[0])
            for j in range(self._knn_k):
                cand_step = nn0_step + j * self._ricl_step_offset
                if cand_step < nn0_traj_len:
                    slots.append((nn0_ep, cand_step))
                else:
                    if next_fallback < pool_indices.shape[0]:
                        fb_ep = int(pool_indices[next_fallback, 0])
                        fb_step = int(pool_indices[next_fallback, 1])
                        slots.append((fb_ep, fb_step))
                        next_fallback += 1
                    else:
                        # Pool exhausted; reuse the last fallback we picked (or top-1 if none).
                        slots.append(slots[-1] if slots else (nn0_ep, nn0_step))
            logger.info(f"ricl_step_offset={self._ricl_step_offset} slots={slots} (nn0_traj_len={nn0_traj_len})")
        assert len(slots) == self._knn_k
        more_obs["_debug_slots"] = np.array(slots, dtype=np.int32)  # (k, 2) (ep, step) for the debug panel
        # collect retrieved info
        for ct, (ep_idx, step_idx) in enumerate(slots):
            demo = self._demos[ep_idx]
            more_obs[f"retrieved_{ct}_state"] = demo["state"][step_idx]
            more_obs[f"retrieved_{ct}_wrist_image"] = demo["wrist_image"][step_idx]
            more_obs[f"retrieved_{ct}_top_image"] = demo["top_image"][step_idx]
            if "right_image" in demo:
                more_obs[f"retrieved_{ct}_right_image"] = demo["right_image"][step_idx]
            else:
                more_obs[f"retrieved_{ct}_right_image"] = np.zeros_like(more_obs[f"retrieved_{ct}_top_image"])
            more_obs[f"retrieved_{ct}_actions"] = get_action_chunk_at_inference_time(demo["actions"], step_idx, self._action_horizon)
            more_obs[f"retrieved_{ct}_prompt"] = demo["prompt"].item()
            # Flag this retrieved neighbor's object (offline SAM mask) to match training.
            if self._attach_seg_masks and os.path.exists(self._demo_sam_paths[ep_idx]):
                more_obs[f"retrieved_{ct}_seg_mask"] = self._retrieved_seg_mask(ep_idx, step_idx)
        # Compute exp_lamda_distances if use_action_interpolation
        if self._use_action_interpolation:
            first_ep, first_step = slots[0]
            first_embedding = self._demos[first_ep]["top_image_embeddings"][first_step]
            distances = [0.0] + [np.linalg.norm(self._demos[ep_idx]["top_image_embeddings"][step_idx:step_idx+1] - first_embedding) for ep_idx, step_idx in slots[1:]]
            distances.append(np.linalg.norm(query_embedding - first_embedding))
            distances = np.clip(np.array(distances), 0, self._max_dist) / self._max_dist
            print(f'distances: {distances}')
            more_obs["exp_lamda_distances"] = np.exp(-self._lamda * distances).reshape(-1, 1)
            print(f'exp_lamda_distances: {more_obs["exp_lamda_distances"]}')
        return {**obs, **more_obs}
    
    def save_obs(self, obs: dict, date: str, prefix: str):
        fol = f"obs_logs/{date}/{prefix}"
        os.makedirs(fol, exist_ok=True)
        current_datettime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # save all images in one png
        big_top_image = []
        big_right_image = []
        big_wrist_image = []    
        for ct in range(self._knn_k):
            big_top_image.append(obs[f"retrieved_{ct}_top_image"])
            big_right_image.append(obs[f"retrieved_{ct}_right_image"])
            big_wrist_image.append(obs[f"retrieved_{ct}_wrist_image"])
        big_top_image.append(obs["query_top_image"])
        big_right_image.append(obs["query_right_image"])
        big_wrist_image.append(obs["query_wrist_image"])
        final_image = np.concatenate((np.concatenate(big_top_image, axis=1), np.concatenate(big_right_image, axis=1), np.concatenate(big_wrist_image, axis=1)), axis=0)
        Image.fromarray(final_image).save(f"{fol}/{current_datettime}.png")
        # save everything else to json
        with open(f"{fol}/{current_datettime}.json", "w") as f:
            everything_else = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in obs.items() if "image" not in k}
            everything_else["final_image_shape"] = list(final_image.shape)
            json.dump(everything_else, f, indent=4)
        return current_datettime

    def save_tokenized_inputs(self, inputs: dict, current_datettime: str, date: str, prefix: str):
        fol = f"obs_logs/{date}/{prefix}"
        os.makedirs(fol, exist_ok=True)
        every_tokenized_input = {}
        for k, v in inputs.items():
            if "token" in k:
                if v is None:
                    every_tokenized_input[k] = v
                    continue
                assert isinstance(v, np.ndarray) and v.dtype in [np.bool_, np.int64], f"{k=}, {v.dtype=}"
                every_tokenized_input[k] = v.astype(np.int32).tolist()
        with open(f"{fol}/{current_datettime}_token_inputs.json", "w") as f:
            json.dump(every_tokenized_input, f, indent=4)

    @override
    def infer(self, obs: dict, debug: bool = True) -> dict:  # type: ignore[misc]
        # Remove the prefix from the obs; get date; below for saving folder only
        prefix = obs.pop("prefix", "temp")
        date = datetime.now().strftime("%m%d")
        # Retrieval
        print()
        logger.info(f'retrieving...')
        obs = self.retrieve(obs)
        # for debugging, save everything in obs
        if debug:
            logger.info(f'saving obs...')
            current_datettime = self.save_obs(obs, date, prefix)
        # Make a copy since transformations may modify the inputs in place.
        logger.info(f'transforming...')
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # for debugging, save tokenized inputs
        if debug:
            logger.info(f'saving tokenized inputs...')
            self.save_tokenized_inputs(inputs, current_datettime, date, prefix)
        # Make a batch and convert to jax.Array.
        logger.info(f'batching...')
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        self._rng, sample_rng = jax.random.split(self._rng)
        logger.info(f'sampling...')
        ricl_obs = _model.RiclObservation.from_dict(inputs, num_retrieved_observations=self._knn_k)
        outputs = {
            "query_state": inputs["query_state"],
            "query_actions": self._sample_actions(sample_rng, ricl_obs, **self._sample_kwargs),
        }

        # Optional debug panel: retrieved ctx (+ SAM masks) and the reasoning-token predicted query mask.
        debug_panel = None
        if self._record_debug and self._predict_seg_mask is not None:
            seg_pred = np.asarray(self._predict_seg_mask(ricl_obs))[0]  # (P,) or (N, P) per-object
            if seg_pred.ndim > 1:  # per-object: collapse channels to a single displayable mask
                seg_pred = seg_pred.max(axis=0)  # (P,)
            gs = int(round(seg_pred.shape[0] ** 0.5))
            # Retrieved SAM masks are (H, W) for union, (N, H, W) for per-object -> collapse to (H, W).
            retrieved_seg_masks = []
            for i in range(self._knn_k):
                m = obs.get(f"retrieved_{i}_seg_mask")
                if m is not None and np.asarray(m).ndim > 2:
                    m = np.asarray(m).max(axis=0)
                retrieved_seg_masks.append(m)
            debug_panel = _build_seg_debug_panel(
                obs["query_top_image"],
                [obs[f"retrieved_{i}_top_image"] for i in range(self._knn_k)],
                retrieved_seg_masks,
                seg_pred.reshape(gs, gs),
                np.asarray(obs["exp_lamda_distances"]).reshape(-1),
                obs.get("_debug_slots"),
            )

        # Unbatch and convert to np.ndarray.
        logger.info(f'unbatching...')
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        final_outputs = self._output_transform(outputs)
        print(f'final_outputs: {final_outputs}')
        if debug_panel is not None:
            final_outputs["debug_panel"] = debug_panel
        return final_outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class TrajPerceiverPolicy(BasePolicy):
    def __init__(
        self,
        model,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        demos_dir: str | None = None,
        max_traj_len: int = 300,
    ):
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._max_traj_len = max_traj_len

        logger.info(f"Loading reference trajectory from {demos_dir}...")
        folders = sorted(f for f in os.listdir(demos_dir) if os.path.isdir(f"{demos_dir}/{f}"))
        if not folders:
            raise ValueError(f"No demo subdirectories found in {demos_dir}")
        ref_npz = np.load(f"{demos_dir}/{folders[0]}/processed_demo.npz")
        self._traj_state, self._traj_top_emb, self._traj_wrist_emb, self._traj_mask = (
            self._preprocess_traj(ref_npz, max_traj_len)
        )
        logger.info(f"Loaded reference trajectory with {self._traj_mask.sum()} valid frames (max={max_traj_len})")

        logger.info("Loading DINOv2 for query embedding...")
        self._dinov2 = load_dinov2()

    @staticmethod
    def _preprocess_traj(ref_npz, max_traj_len: int):
        traj_state = ref_npz["state"].astype(np.float32)                          # [T, state_dim]
        traj_top_emb = ref_npz["top_image_embeddings"].astype(np.float32)         # [T, 49152]
        traj_wrist_emb = ref_npz["wrist_image_embeddings"].astype(np.float32)     # [T, 49152]
        T = traj_state.shape[0]

        if T > max_traj_len:
            idx = np.linspace(0, T - 1, max_traj_len, dtype=int)
            traj_state = traj_state[idx]
            traj_top_emb = traj_top_emb[idx]
            traj_wrist_emb = traj_wrist_emb[idx]
            traj_mask = np.ones(max_traj_len, dtype=bool)
        else:
            pad = max_traj_len - T
            traj_state = np.concatenate([traj_state, np.zeros((pad, traj_state.shape[1]), dtype=np.float32)], axis=0)
            traj_top_emb = np.concatenate([traj_top_emb, np.zeros((pad, 49152), dtype=np.float32)], axis=0)
            traj_wrist_emb = np.concatenate([traj_wrist_emb, np.zeros((pad, 49152), dtype=np.float32)], axis=0)
            traj_mask = np.array([True] * T + [False] * pad, dtype=bool)

        traj_top_emb = traj_top_emb.reshape(max_traj_len, 64, 768).mean(axis=1)
        traj_wrist_emb = traj_wrist_emb.reshape(max_traj_len, 64, 768).mean(axis=1)
        return traj_state, traj_top_emb, traj_wrist_emb, traj_mask

    def _ensure_query_keys(self, obs: dict) -> dict:
        if "query_top_image" not in obs:
            if "observation/image" in obs:
                obs["query_top_image"] = obs["observation/image"]
            elif "top_image" in obs:
                obs["query_top_image"] = obs["top_image"]
        if "query_wrist_image" not in obs:
            if "observation/wrist_image" in obs:
                obs["query_wrist_image"] = obs["observation/wrist_image"]
            elif "wrist_image" in obs:
                obs["query_wrist_image"] = obs["wrist_image"]
        if "query_state" not in obs:
            if "observation/state" in obs:
                obs["query_state"] = obs["observation/state"]
            elif "state" in obs:
                obs["query_state"] = obs["state"]
        if "query_prompt" not in obs and "prompt" in obs:
            obs["query_prompt"] = obs["prompt"]
        return obs

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        obs = self._ensure_query_keys(obs)
        obs["traj_state"] = self._traj_state
        obs["traj_top_emb"] = self._traj_top_emb
        obs["traj_wrist_emb"] = self._traj_wrist_emb
        obs["traj_mask"] = self._traj_mask
        # Compute DINOv2 query embedding (same space as trajectory K/V)
        raw_emb = embed(obs["query_top_image"], self._dinov2)  # [1, 49152]
        obs["query_dino_top_emb"] = raw_emb.reshape(64, 768).mean(axis=0).astype(np.float32)

        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        self._rng, sample_rng = jax.random.split(self._rng)
        actions = self._sample_actions(sample_rng, inputs, **self._sample_kwargs)

        outputs = {"query_actions": actions}
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        return self._output_transform(outputs)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
