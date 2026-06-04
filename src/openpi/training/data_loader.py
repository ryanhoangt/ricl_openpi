from collections.abc import Iterator, Sequence
import functools
import logging
import multiprocessing
import os
import typing
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.models.pi0_fast_ricl as _pi0_fast_ricl
import openpi.training.config as _config
import openpi.transforms as _transforms
import json

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples
    

def get_action_chunk(action_joint_vels, action_gripper_pos, step_idx, action_horizon):
    num_steps = len(action_joint_vels)
    assert action_joint_vels.shape == (num_steps, 7) and action_gripper_pos.shape == (num_steps, 1)
    action_chunk = []
    for i in range(action_horizon):
        if step_idx+i < num_steps:
            action_chunk.append(np.concatenate([action_joint_vels[step_idx+i], action_gripper_pos[step_idx+i]], axis=0))
        else:
            action_chunk.append(np.concatenate([np.zeros(action_joint_vels.shape[-1], dtype=np.float32), action_gripper_pos[-1]], axis=0))
    action_chunk = np.stack(action_chunk, axis=0)
    assert action_chunk.shape == (action_horizon, 8), f"{action_chunk.shape=}"
    return action_chunk



def get_action_chunk_from_actions(actions, step_idx, action_horizon):
    num_steps = len(actions)
    action_dim = actions.shape[-1]
    action_chunk = []
    for i in range(action_horizon):
        if step_idx + i < num_steps:
            action_chunk.append(actions[step_idx + i])
        else:
            action_chunk.append(np.zeros(action_dim, dtype=np.float32))
    action_chunk = np.stack(action_chunk, axis=0)
    assert action_chunk.shape == (action_horizon, action_dim), f"{action_chunk.shape=}"
    return action_chunk


class Pi0FastDroidFinetuneDataset(Dataset):
    def __init__(self, model_config: _pi0_fast_ricl.Pi0FASTRiclConfig, finetuning_collected_demos_dir: str | None):
        assert finetuning_collected_demos_dir is not None
        collected_demos_infos = {k: json.load(open(f"{finetuning_collected_demos_dir}/{k}.json")) for k in ['ep_idxs_to_fol', 'fols_to_ep_idxs', 'groups_to_ep_fols', 'groups_to_ep_idxs']}
        
        # files from the collected demos for training
        indices_files = [] 
        for group_name, ep_fols in collected_demos_infos["groups_to_ep_fols"].items():
            for ep_fol in ep_fols:
                indices_files.append(f"ricl_droid_preprocessing/{ep_fol}/indices_and_distances.npz")
        
        # actual loading...
        count_collected_demos = 0
        all_query_indices = []
        for file_idx, file_path in enumerate(indices_files):
            indices_and_dists = np.load(file_path)
            query_indices = indices_and_dists["query_indices"]
            num_steps = query_indices.shape[0]
            assert query_indices.shape == (num_steps, 2) and query_indices.dtype == np.int32
            expected_query_indices = np.array([[100000+file_idx, i] for i in range(num_steps)], dtype=np.int32)
            assert np.allclose(query_indices, expected_query_indices), f"{query_indices=}, {expected_query_indices=}"
            all_query_indices.append(query_indices)
            count_collected_demos += num_steps
        print(f"num states in collected demos given by count_collected_demos: {count_collected_demos}")
        all_query_indices = np.concatenate(all_query_indices, axis=0)
        len_dataset = all_query_indices.shape[0]
        print(f"len_dataset: {len_dataset}")
        assert len_dataset == count_collected_demos
        assert all_query_indices.shape == (len_dataset, 2) and all_query_indices.dtype == np.int32

        # load all data paths 
        all_ep_idxs = list(np.unique(all_query_indices[:, 0]))
        all_ep_data_paths = {ep_idx: 
                                    f"ricl_droid_preprocessing/{collected_demos_infos['ep_idxs_to_fol'][str(ep_idx)]}/processed_demo.npz"
                            for ep_idx in all_ep_idxs}
        common_prompt = " ".join(collected_demos_infos['ep_idxs_to_fol']['100000'].split("/")[1].split("_")[1:])
        print(f'num episodes: {len(all_ep_idxs)}')
        print(f"common_prompt: {common_prompt}")

        # save
        self.len_dataset = len_dataset
        self.all_ep_data_paths = all_ep_data_paths
        self.common_prompt = common_prompt
        self.all_query_indices = all_query_indices
        self.action_horizon = model_config.action_horizon

    def __getitem__(self, index: SupportsIndex) -> dict:
        query_ep_idx, query_step_idx = self.all_query_indices[index, :]
        ep_data = np.load(self.all_ep_data_paths[query_ep_idx])
        data = {'observation/exterior_image_1_left': ep_data['right_image'][query_step_idx],
                'observation/wrist_image_left': ep_data['wrist_image'][query_step_idx],
                'observation/joint_position': ep_data['state'][query_step_idx][:-1],
                'observation/gripper_position': ep_data['state'][query_step_idx][-1:],
                'actions': get_action_chunk(ep_data['actions'][:, :-1], ep_data['actions'][:, -1:], query_step_idx, self.action_horizon),
                'prompt': self.common_prompt}
        return data

    def __len__(self) -> int:
        return self.len_dataset


class RiclDroidDataset(Dataset):
    def __init__(self, model_config: _pi0_fast_ricl.Pi0FASTRiclConfig, finetuning_collected_demos_dir: str | None):
        # setup
        num_retrieved_observations = model_config.num_retrieved_observations
        knn_k = 100
        assert num_retrieved_observations <= knn_k
        embedding_type = "embeddings__wrist_image_left" # retrieval based on embeddings of wrist images
        indices_and_dists_fol = f"ricl_droid_preprocessing/droid_new_broken_up_indices_and_distances/chosenIDscene_id_numepisodes20_embtype{embedding_type}_knnk100"
        outer_dir = "ricl_droid_preprocessing/collected_demos_training" if finetuning_collected_demos_dir is None else finetuning_collected_demos_dir
        collected_demos_infos = {k: json.load(open(f"{outer_dir}/{k}.json")) for k in ['ep_idxs_to_fol', 'fols_to_ep_idxs', 'groups_to_ep_fols', 'groups_to_ep_idxs']}
        # load indices_and_dists
        all_retrieved_indices = []
        all_query_indices = []
        all_distances = []
        
        ## files from the droid dataset
        # indices_files = os.listdir(indices_and_dists_fol)
        # indices_files = [os.path.join(indices_and_dists_fol, f) for f in indices_files]
        indices_files = [] ## no files from droid dataset

        # files from the collected demos for training
        for group_name, ep_fols in collected_demos_infos["groups_to_ep_fols"].items():
            for ep_fol in ep_fols:
                indices_files.append(f"ricl_droid_preprocessing/{ep_fol}/indices_and_distances.npz")
        # actual loading...
        count_droid = 0
        count_collected_demos = 0
        for file_path in indices_files:
            indices_and_dists = np.load(file_path)
            query_indices, retrieved_indices = indices_and_dists["query_indices"], indices_and_dists["retrieved_indices"][:, :num_retrieved_observations, :]
            distances = np.concatenate((indices_and_dists["distances"][:, :num_retrieved_observations], indices_and_dists["distances"][:, -1:]), axis=1)
            num_steps = query_indices.shape[0]
            assert retrieved_indices.shape == (num_steps, num_retrieved_observations, 2) and retrieved_indices.dtype == np.int32 
            assert query_indices.shape == (num_steps, 2) and query_indices.dtype == np.int32
            all_retrieved_indices.append(retrieved_indices)
            all_query_indices.append(query_indices)
            all_distances.append(distances)
            if "collected_demos_training" in file_path or "collected_demos" in file_path:
                count_collected_demos += num_steps
            else:
                count_droid += num_steps
        print(f"count_droid: {count_droid}, count_collected_demos: {count_collected_demos}")
        all_retrieved_indices = np.concatenate(all_retrieved_indices, axis=0)
        all_query_indices = np.concatenate(all_query_indices, axis=0)
        all_distances = np.concatenate(all_distances, axis=0)
        len_dataset = all_retrieved_indices.shape[0]
        print(f"len_dataset: {len_dataset}")
        assert len_dataset == count_droid + count_collected_demos
        assert all_retrieved_indices.shape == (len_dataset, num_retrieved_observations, 2) and all_retrieved_indices.dtype == np.int32
        assert all_query_indices.shape == (len_dataset, 2) and all_query_indices.dtype == np.int32
        assert all_distances.shape == (len_dataset, num_retrieved_observations + 1) and all_distances.dtype == np.float64
        
        # normalize all_distances and convert to float32
        max_dist_value = json.load(open(f"assets/max_distance.json", 'r'))['distances']['max']
        if finetuning_collected_demos_dir is None:
            assert max_dist_value == np.max(all_distances), f"{max_dist_value=} from norm stats time does not match {np.max(all_distances)=} from dataset"
            print(f'max distance value: {max_dist_value}')
        all_distances = all_distances / max_dist_value
        all_distances = all_distances.astype(np.float32)

        # load all data paths 
        ds_name = f"droid_new"
        ds_fol = f"ricl_droid_preprocessing/{ds_name}_broken_up"
        all_ep_idxs = list(np.unique(all_retrieved_indices[:, :, 0])) + list(np.unique(all_query_indices[:, 0]))
        all_ep_data_paths = {ep_idx: 
                                    f"{ds_fol}/episode_{ep_idx}.npz" 
                                    if ep_idx < 100000 else 
                                    f"ricl_droid_preprocessing/{collected_demos_infos['ep_idxs_to_fol'][str(ep_idx)]}/processed_demo.npz"
                            for ep_idx in all_ep_idxs}
        all_ep_prompts = {ep_idx: 
                                    json.load(open(f"{ds_fol}/episode_{ep_idx}.json"))["language_instruction"]  
                                    if ep_idx < 100000 else 
                                    " ".join(collected_demos_infos['ep_idxs_to_fol'][str(ep_idx)].split("/")[1].split("_")[1:])
                            for ep_idx in all_ep_idxs}
        
        # if all episode prompts are the same, print the first prompt
        if all(all_ep_prompts[ep_idx] == all_ep_prompts[list(all_ep_prompts.keys())[0]] for ep_idx in all_ep_prompts):
            print(f"all {len(all_ep_prompts)} episode prompts are the same: {all_ep_prompts[list(all_ep_prompts.keys())[0]]}")

        # save
        self.len_dataset = len_dataset
        self.all_ep_data_paths = all_ep_data_paths
        self.all_ep_prompts = all_ep_prompts
        self.all_retrieved_indices = all_retrieved_indices
        self.all_query_indices = all_query_indices
        self.all_distances = all_distances
        self.use_action_interpolation = model_config.use_action_interpolation
        self.lamda = model_config.lamda
        self.action_horizon = model_config.action_horizon

    def __getitem__(self, index: SupportsIndex) -> dict:
        retrieved_indices = self.all_retrieved_indices[index, :, :]
        query_ep_idx, query_step_idx = self.all_query_indices[index, :]
        
        ep_idxs = list(np.unique(retrieved_indices[:, 0])) + [query_ep_idx]
        ep_data = {ep_idx: np.load(self.all_ep_data_paths[ep_idx]) for ep_idx in ep_idxs}
        data = {}
        random_ext_img = np.random.choice(["left", "right"])
        for ct, (ep_idx, step_idx) in enumerate(retrieved_indices):
            prefix = f"retrieved_{ct}_"
            if ep_idx < 100000:
                data[f"{prefix}top_image"] = ep_data[ep_idx]["observation__exterior_image_1_left"][step_idx]
                data[f"{prefix}right_image"] = ep_data[ep_idx]["observation__exterior_image_2_left"][step_idx]
                data[f"{prefix}wrist_image"] = ep_data[ep_idx]["observation__wrist_image_left"][step_idx]
                data[f"{prefix}state"] = np.concatenate([ep_data[ep_idx]["observation__joint_position"][step_idx], ep_data[ep_idx]["observation__gripper_position"][step_idx]], axis=0)
                data[f"{prefix}actions"] = get_action_chunk(ep_data[ep_idx]["action_dict__joint_velocity"], ep_data[ep_idx]["action_dict__gripper_position"], step_idx, self.action_horizon)
            else:
                data[f"{prefix}top_image"] = ep_data[ep_idx]["top_image"][step_idx]
                data[f"{prefix}right_image"] = ep_data[ep_idx]["right_image"][step_idx]
                data[f"{prefix}wrist_image"] = ep_data[ep_idx]["wrist_image"][step_idx]
                data[f"{prefix}state"] = ep_data[ep_idx]["state"][step_idx]
                data[f"{prefix}actions"] = get_action_chunk(ep_data[ep_idx]["actions"][:, :-1], ep_data[ep_idx]["actions"][:, -1:], step_idx, self.action_horizon)
            data[f"{prefix}prompt"] = self.all_ep_prompts[ep_idx]
        
        prefix = "query_"
        if query_ep_idx < 100000:
            data[f"{prefix}top_image"] = ep_data[query_ep_idx]["observation__exterior_image_1_left"][query_step_idx]
            data[f"{prefix}right_image"] = ep_data[query_ep_idx]["observation__exterior_image_2_left"][query_step_idx]
            data[f"{prefix}wrist_image"] = ep_data[query_ep_idx]["observation__wrist_image_left"][query_step_idx]
            data[f"{prefix}state"] = np.concatenate([ep_data[query_ep_idx]["observation__joint_position"][query_step_idx], ep_data[query_ep_idx]["observation__gripper_position"][query_step_idx]], axis=0)
            data[f"{prefix}actions"] = get_action_chunk(ep_data[query_ep_idx]["action_dict__joint_velocity"], ep_data[query_ep_idx]["action_dict__gripper_position"], query_step_idx, self.action_horizon)
        else:
            data[f"{prefix}top_image"] = ep_data[query_ep_idx]["top_image"][query_step_idx]
            data[f"{prefix}right_image"] = ep_data[query_ep_idx]["right_image"][query_step_idx]
            data[f"{prefix}wrist_image"] = ep_data[query_ep_idx]["wrist_image"][query_step_idx]
            data[f"{prefix}state"] = ep_data[query_ep_idx]["state"][query_step_idx]
            data[f"{prefix}actions"] = get_action_chunk(ep_data[query_ep_idx]["actions"][:, :-1], ep_data[query_ep_idx]["actions"][:, -1:], query_step_idx, self.action_horizon)
        data[f"{prefix}prompt"] = self.all_ep_prompts[query_ep_idx]

        if self.use_action_interpolation:
            # read distances
            distances = self.all_distances[index, :]
            # then compute exp(-lamda * distances)
            data["exp_lamda_distances"] = np.exp(-self.lamda * distances).reshape(-1, 1)

        return data

    def __len__(self) -> int:
        return self.len_dataset



def _resolve_demo_path(base_dir: str, demo_path: str) -> str:
    demo_path = os.path.expanduser(demo_path)
    if os.path.isabs(demo_path) and os.path.exists(demo_path):
        return demo_path
    if os.path.exists(demo_path):
        return demo_path
    return os.path.join(base_dir, demo_path)


class TrajPerceiverLiberoDataset(Dataset):
    """Dataset for trajectory-perceiver training.

    Each sample is a (query_timestep, reference_trajectory) pair.  The
    reference trajectory is the first episode of the same task group as the
    query, subsampled to max_traj_len if longer, zero-padded if shorter.
    """

    def __init__(self, model_config, finetuning_collected_demos_dir: str | None):
        self.action_horizon = model_config.action_horizon
        self.max_traj_len = model_config.max_traj_len

        outer_dir = (
            "ricl_libero_preprocessing/collected_demos_training"
            if finetuning_collected_demos_dir is None
            else finetuning_collected_demos_dir
        )
        collected_demos_infos = {
            k: json.load(open(f"{outer_dir}/{k}.json"))
            for k in ["ep_idxs_to_fol", "fols_to_ep_idxs", "groups_to_ep_fols", "groups_to_ep_idxs"]
        }

        # Build (ep_idx, step_idx, task_group) index and task→ref_ep mapping.
        all_query_indices = []   # list of (ep_idx, step_idx, group_name)
        task_to_ref_ep_path = {}  # group_name → path to reference processed_demo.npz
        all_ep_data_paths = {}   # ep_idx → path

        for group_name, ep_fols in collected_demos_infos["groups_to_ep_fols"].items():
            # First episode of this task is the reference.
            ref_ep_fol = ep_fols[0]
            ref_ep_path = os.path.join(_resolve_demo_path(outer_dir, ref_ep_fol), "processed_demo.npz")
            task_to_ref_ep_path[group_name] = ref_ep_path

            for ep_fol in ep_fols:
                ep_idx = collected_demos_infos["fols_to_ep_idxs"][ep_fol]
                ep_path = os.path.join(_resolve_demo_path(outer_dir, ep_fol), "processed_demo.npz")
                all_ep_data_paths[ep_idx] = ep_path

                ep_data = np.load(ep_path)
                T = ep_data["state"].shape[0]
                for step_idx in range(T):
                    all_query_indices.append((ep_idx, step_idx, group_name))

        self.all_query_indices = all_query_indices
        self.task_to_ref_ep_path = task_to_ref_ep_path
        self.all_ep_data_paths = all_ep_data_paths
        logging.info(f"TrajPerceiverLiberoDataset: {len(all_query_indices)} query timesteps across {len(task_to_ref_ep_path)} tasks")

    def __getitem__(self, index: SupportsIndex) -> dict:
        ep_idx, step_idx, group_name = self.all_query_indices[index]

        ep_data = np.load(self.all_ep_data_paths[ep_idx])
        # Mean-pool DINOv2 64PATCHES → 768-dim for query frame (used as perceiver Q seed)
        query_dino_top_emb = ep_data["top_image_embeddings"][step_idx].reshape(64, 768).mean(axis=0).astype(np.float32)
        data = {
            "query_top_image": ep_data["top_image"][step_idx],
            "query_wrist_image": ep_data["wrist_image"][step_idx],
            "query_state": ep_data["state"][step_idx],
            "query_dino_top_emb": query_dino_top_emb,
            "query_actions": get_action_chunk_from_actions(ep_data["actions"], step_idx, self.action_horizon),
            "query_prompt": ep_data["prompt"].item(),
        }

        # Load reference trajectory: state + DINOv2 embeddings (64PATCHES, vitb14).
        ref_data = np.load(self.task_to_ref_ep_path[group_name])
        traj_state = ref_data["state"].astype(np.float32)                          # [T, state_dim]
        traj_top_emb = ref_data["top_image_embeddings"].astype(np.float32)         # [T, 49152]
        traj_wrist_emb = ref_data["wrist_image_embeddings"].astype(np.float32)     # [T, 49152]
        T = traj_state.shape[0]

        if T > self.max_traj_len:
            indices = np.linspace(0, T - 1, self.max_traj_len, dtype=int)
            traj_state = traj_state[indices]
            traj_top_emb = traj_top_emb[indices]
            traj_wrist_emb = traj_wrist_emb[indices]
            traj_mask = np.ones(self.max_traj_len, dtype=bool)
        else:
            pad = self.max_traj_len - T
            traj_state = np.concatenate(
                [traj_state, np.zeros((pad, traj_state.shape[1]), dtype=np.float32)], axis=0
            )
            traj_top_emb = np.concatenate(
                [traj_top_emb, np.zeros((pad, 49152), dtype=np.float32)], axis=0
            )
            traj_wrist_emb = np.concatenate(
                [traj_wrist_emb, np.zeros((pad, 49152), dtype=np.float32)], axis=0
            )
            traj_mask = np.array([True] * T + [False] * pad, dtype=bool)

        # Mean-pool DINOv2 64PATCHES (64 × 768 = 49152) → 768-dim per frame per camera.
        traj_top_emb = traj_top_emb.reshape(self.max_traj_len, 64, 768).mean(axis=1)    # [max_traj_len, 768]
        traj_wrist_emb = traj_wrist_emb.reshape(self.max_traj_len, 64, 768).mean(axis=1)

        data["traj_state"] = traj_state         # [max_traj_len, state_dim]
        data["traj_top_emb"] = traj_top_emb     # [max_traj_len, 768]
        data["traj_wrist_emb"] = traj_wrist_emb # [max_traj_len, 768]
        data["traj_mask"] = traj_mask           # [max_traj_len]
        return data

    def __len__(self) -> int:
        return len(self.all_query_indices)


class RiclLiberoDataset(Dataset):
    def __init__(self, model_config: _pi0_fast_ricl.Pi0FASTRiclConfig, finetuning_collected_demos_dir: str | None):
        num_retrieved_observations = model_config.num_retrieved_observations
        ricl_step_offset = int(getattr(model_config, "ricl_step_offset", 0))
        knn_k = 100
        assert num_retrieved_observations <= knn_k
        # When ricl_step_offset > 0 we need a fallback pool larger than num_retrieved_observations.
        pool_size = num_retrieved_observations if ricl_step_offset == 0 else min(2 * num_retrieved_observations, knn_k)

        outer_dir = (
            "ricl_libero_preprocessing/collected_demos_training"
            if finetuning_collected_demos_dir is None
            else finetuning_collected_demos_dir
        )
        collected_demos_infos = {
            k: json.load(open(f"{outer_dir}/{k}.json"))
            for k in ["ep_idxs_to_fol", "fols_to_ep_idxs", "groups_to_ep_fols", "groups_to_ep_idxs"]
        }

        indices_files = []
        for ep_fols in collected_demos_infos["groups_to_ep_fols"].values():
            for ep_fol in ep_fols:
                ep_path = _resolve_demo_path(outer_dir, ep_fol)
                indices_files.append(os.path.join(ep_path, "indices_and_distances.npz"))

        all_pool_indices = []
        all_query_indices = []
        all_pool_distances = []

        for file_path in indices_files:
            indices_and_dists = np.load(file_path)
            query_indices = indices_and_dists["query_indices"]
            # Pool of candidates from the precomputed top-knn_k list. Distances row layout in the
            # precomputed file is [d(NN_0→NN_0)=0, d(NN_1→NN_0), ..., d(NN_{knn_k-1}→NN_0), d(query→NN_0)].
            pool_indices = indices_and_dists["retrieved_indices"][:, :pool_size, :]
            pool_distances = np.concatenate(
                (
                    indices_and_dists["distances"][:, :pool_size],
                    indices_and_dists["distances"][:, -1:],
                ),
                axis=1,
            )
            all_pool_indices.append(pool_indices)
            all_query_indices.append(query_indices)
            all_pool_distances.append(pool_distances)

        all_pool_indices = np.concatenate(all_pool_indices, axis=0)
        all_query_indices = np.concatenate(all_query_indices, axis=0)
        all_pool_distances = np.concatenate(all_pool_distances, axis=0)
        len_dataset = all_pool_indices.shape[0]

        assert all_pool_indices.shape == (len_dataset, pool_size, 2)
        assert all_query_indices.shape == (len_dataset, 2)
        assert all_pool_distances.shape == (len_dataset, pool_size + 1)

        max_dist_value = float(np.max(all_pool_distances)) if len(all_pool_distances) else 1.0
        if max_dist_value == 0:
            max_dist_value = 1.0
        logging.info(
            f"RiclLiberoDataset max_dist_value={max_dist_value:.6f} "
            f"— save this to a max_distance.json for inference!"
        )
        # Save max_distance alongside the training data for inference use
        max_dist_file = os.path.join(outer_dir, "max_distance.json")
        with open(max_dist_file, "w") as f:
            json.dump({"distances": {"max": max_dist_value}}, f, indent=2)
        logging.info(f"Saved max_distance to {max_dist_file}")

        all_pool_distances = (all_pool_distances / max_dist_value).astype(np.float32)

        all_ep_idxs = list(np.unique(all_pool_indices[:, :, 0])) + list(np.unique(all_query_indices[:, 0]))
        all_ep_data_paths = {
            ep_idx: os.path.join(
                _resolve_demo_path(outer_dir, collected_demos_infos["ep_idxs_to_fol"][str(ep_idx)]),
                "processed_demo.npz",
            )
            for ep_idx in all_ep_idxs
        }

        if ricl_step_offset == 0:
            # Original behavior: top-k NN slots with precomputed distances.
            all_retrieved_indices = all_pool_indices[:, :num_retrieved_observations, :]
            all_distances = np.concatenate(
                (all_pool_distances[:, :num_retrieved_observations], all_pool_distances[:, -1:]),
                axis=1,
            )
        else:
            # Offset scheme: slot 0 = top-1 NN at its step; slot j>=1 = same NN at step + j*offset,
            # falling back to the next unused pool entry on overflow. Distances measured to slot 0
            # (the anchor). For fallback slots we reuse the precomputed pool distance; for offset
            # slots we compute the embedding distance on the fly. We iterate by nn0_ep so each
            # demo's top_image_embeddings is loaded at most once (bounds peak memory).
            logging.info(
                f"RiclLiberoDataset: ricl_step_offset={ricl_step_offset} active; "
                f"pool_size={pool_size}, recomputing slots and distances"
            )
            offset_slots = np.zeros((len_dataset, num_retrieved_observations, 2), dtype=np.int32)
            offset_distances = np.zeros((len_dataset, num_retrieved_observations + 1), dtype=np.float32)

            samples_by_nn0_ep: dict[int, list[int]] = {}
            for i in range(len_dataset):
                samples_by_nn0_ep.setdefault(int(all_pool_indices[i, 0, 0]), []).append(i)

            for nn0_ep, sample_ids in samples_by_nn0_ep.items():
                emb_arr = np.load(all_ep_data_paths[nn0_ep])["top_image_embeddings"]
                nn0_traj_len = int(emb_arr.shape[0])
                for i in sample_ids:
                    nn0_step = int(all_pool_indices[i, 0, 1])
                    first_emb = emb_arr[nn0_step]
                    next_fallback = 1
                    for j in range(num_retrieved_observations):
                        cand_step = nn0_step + j * ricl_step_offset
                        if cand_step < nn0_traj_len:
                            offset_slots[i, j, 0] = nn0_ep
                            offset_slots[i, j, 1] = cand_step
                            if j == 0:
                                offset_distances[i, j] = 0.0
                            else:
                                d_raw = float(np.linalg.norm(emb_arr[cand_step] - first_emb))
                                d_clipped = min(max(d_raw, 0.0), max_dist_value)
                                offset_distances[i, j] = d_clipped / max_dist_value
                        else:
                            if next_fallback < pool_size:
                                offset_slots[i, j, 0] = int(all_pool_indices[i, next_fallback, 0])
                                offset_slots[i, j, 1] = int(all_pool_indices[i, next_fallback, 1])
                                offset_distances[i, j] = all_pool_distances[i, next_fallback]
                                next_fallback += 1
                            else:
                                offset_slots[i, j] = offset_slots[i, j - 1] if j > 0 else np.array([nn0_ep, nn0_step], dtype=np.int32)
                                offset_distances[i, j] = offset_distances[i, j - 1] if j > 0 else 0.0
                    offset_distances[i, -1] = all_pool_distances[i, -1]  # query → nn0, precomputed
                del emb_arr  # free per-demo embeddings before loading the next

            all_retrieved_indices = offset_slots
            all_distances = offset_distances

        # Episode paths may need to be re-derived if the slot set differs from the pool slice
        # (offset mode can pull additional episodes via fallback). The pool-based derivation
        # above is already a superset, so it's safe.

        self.len_dataset = len_dataset
        self.all_ep_data_paths = all_ep_data_paths
        self.all_retrieved_indices = all_retrieved_indices
        self.all_query_indices = all_query_indices
        self.all_distances = all_distances
        self.use_action_interpolation = model_config.use_action_interpolation
        self.lamda = model_config.lamda
        self.action_horizon = model_config.action_horizon
        self.ricl_step_offset = ricl_step_offset

    def __getitem__(self, index: SupportsIndex) -> dict:
        retrieved_indices = self.all_retrieved_indices[index, :, :]
        query_ep_idx, query_step_idx = self.all_query_indices[index, :]

        ep_idxs = list(np.unique(retrieved_indices[:, 0])) + [query_ep_idx]
        ep_data = {ep_idx: np.load(self.all_ep_data_paths[ep_idx]) for ep_idx in ep_idxs}
        data = {}

        for ct, (ep_idx, step_idx) in enumerate(retrieved_indices):
            prefix = f"retrieved_{ct}_"
            data[f"{prefix}top_image"] = ep_data[ep_idx]["top_image"][step_idx]
            data[f"{prefix}wrist_image"] = ep_data[ep_idx]["wrist_image"][step_idx]
            data[f"{prefix}state"] = ep_data[ep_idx]["state"][step_idx]
            data[f"{prefix}actions"] = get_action_chunk_from_actions(
                ep_data[ep_idx]["actions"], step_idx, self.action_horizon
            )
            data[f"{prefix}prompt"] = ep_data[ep_idx]["prompt"].item()

        prefix = "query_"
        data[f"{prefix}top_image"] = ep_data[query_ep_idx]["top_image"][query_step_idx]
        data[f"{prefix}wrist_image"] = ep_data[query_ep_idx]["wrist_image"][query_step_idx]
        data[f"{prefix}state"] = ep_data[query_ep_idx]["state"][query_step_idx]
        data[f"{prefix}actions"] = get_action_chunk_from_actions(
            ep_data[query_ep_idx]["actions"], query_step_idx, self.action_horizon
        )
        data[f"{prefix}prompt"] = ep_data[query_ep_idx]["prompt"].item()

        if self.use_action_interpolation:
            distances = self.all_distances[index, :]
            data["exp_lamda_distances"] = np.exp(-self.lamda * distances).reshape(-1, 1)

        return data

    def __len__(self) -> int:
        return self.len_dataset


@functools.lru_cache(maxsize=64)
def _load_union_masks(mask_path: str) -> np.ndarray:
    """Load `sam_masks_top.npz` and union over objects -> (T, H, W) bool. Cached per file."""
    d = np.load(mask_path)
    masks = d["masks"]  # (T, n_obj, H, W) bool
    if "frame_indices" in d:
        fi = d["frame_indices"]
        assert np.array_equal(fi, np.arange(len(fi))), f"frame_indices not contiguous in {mask_path}"
    return np.any(masks, axis=1)  # (T, H, W)


class RiclReasoningLiberoDataset(RiclLiberoDataset):
    """RICL LIBERO dataset that additionally serves union SAM masks for the query and each
    retrieved slot (top camera). Used by the implicit-reasoning config."""

    def __init__(self, model_config: _pi0_fast_ricl.Pi0FASTRiclConfig, finetuning_collected_demos_dir: str | None):
        super().__init__(model_config, finetuning_collected_demos_dir)
        # Require a SAM mask file for every episode in the buffer; fail loudly otherwise.
        self.all_ep_mask_paths = {}
        for ep_idx, demo_path in self.all_ep_data_paths.items():
            mask_path = os.path.join(os.path.dirname(demo_path), "sam_masks_top.npz")
            assert os.path.exists(mask_path), (
                f"Reasoning RICL requires SAM masks for every episode, but missing: {mask_path}"
            )
            self.all_ep_mask_paths[ep_idx] = mask_path

    def _union_mask(self, ep_idx, step_idx) -> np.ndarray:
        return _load_union_masks(self.all_ep_mask_paths[int(ep_idx)])[int(step_idx)]  # (H, W) bool

    def __getitem__(self, index: SupportsIndex) -> dict:
        data = super().__getitem__(index)
        retrieved_indices = self.all_retrieved_indices[index, :, :]
        query_ep_idx, query_step_idx = self.all_query_indices[index, :]
        for ct, (ep_idx, step_idx) in enumerate(retrieved_indices):
            data[f"retrieved_{ct}_seg_mask"] = self._union_mask(ep_idx, step_idx)
        data["query_seg_mask"] = self._union_mask(query_ep_idx, query_step_idx)
        return data


def create_dataset(data_config: _config.DataConfig, model_config: _model.BaseModelConfig) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, local_files_only=data_config.local_files_only)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(model_config.action_horizon)]
            for key in data_config.action_sequence_keys
        },
        local_files_only=data_config.local_files_only,
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
    """
    data_config = config.data.create(config.assets_dirs, config.model)

    if "traj_perceiver" in config.name and "libero" in config.name:
        dataset = TrajPerceiverLiberoDataset(config.model, config.finetuning_collected_demos_dir)
    elif "ricl" in config.name:
        if "libero" in config.name:
            if "reasoning" in config.name:
                dataset = RiclReasoningLiberoDataset(config.model, config.finetuning_collected_demos_dir)
            else:
                dataset = RiclLiberoDataset(config.model, config.finetuning_collected_demos_dir)
        else:
            dataset = RiclDroidDataset(config.model, config.finetuning_collected_demos_dir)
    elif "pi0_fast_droid___finetune_on_" in config.name:
        dataset = Pi0FastDroidFinetuneDataset(config.model, config.finetuning_collected_demos_dir)
    else:
        dataset = create_dataset(data_config, config.model)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=config.seed,
    )

    class DataLoaderImpl(DataLoader):
        def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
            self._data_config = data_config
            self._data_loader = data_loader

        def data_config(self) -> _config.DataConfig:
            return self._data_config

        def __iter__(self):
            for batch in self._data_loader:
                if "traj_perceiver" in config.name and "libero" in config.name:
                    # Convert images to float32 [-1, 1] if not already done by transforms.
                    for key in batch["query_image"]:
                        if batch["query_image"][key].dtype == jnp.uint8:
                            batch["query_image"][key] = batch["query_image"][key].astype(jnp.float32) / 255.0 * 2.0 - 1.0
                    yield batch, batch["query_actions"]
                elif "ricl" in config.name:
                    yield _model.RiclObservation.from_dict(batch, config.model.num_retrieved_observations), batch["query_actions"]
                else:
                    yield _model.Observation.from_dict(batch), batch["actions"]

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *x: np.stack(np.asarray(x), axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
