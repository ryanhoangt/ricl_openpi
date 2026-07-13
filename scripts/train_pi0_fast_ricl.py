import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import optax
import tqdm_loggable.auto as tqdm
import wandb

import numpy as np

import openpi.models.model as _model
import openpi.models.pi0_fast_reasoning_ricl as _pi0_fast_reasoning_ricl
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


def count_parameters(params, trainable_filter):
    """Counts total and trainable parameters."""
    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    
    trainable_params = sum(
        p.size for p in jax.tree_util.tree_leaves(params.filter(trainable_filter))
    )
    
    return total_params, trainable_params


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        # Count total and trainable parameters
        total_params, trainable_params = count_parameters(params, config.trainable_filter)
        print(f"Total Parameters: {total_params // 1e6}M")
        print(f"Trainable Parameters: {trainable_params // 1e6}M")
        print(f"Trainable Parameters %: {100 * trainable_params / total_params:.2f}%")

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


def create_decode_indices(config: _config.TrainConfig) -> jax.Array | None:
    # Only RICL models (with multiple retrieved observations) need explicit decode indices
    # to skip image tokens in the loss. Models without retrieval handle loss masking internally.
    if not hasattr(config.model, 'num_retrieved_observations'):
        return None
    image_token_len = 256*2 # number of image tokens times number of images
    prompt_token_len = config.model.max_token_len # max token len for each retrieved/query "prompt, state, action" prompt
    total_token_len = image_token_len + prompt_token_len
    decode_indices = []
    for i in range(config.model.num_retrieved_observations + 1):
        decode_indices.extend(list(range(i * total_token_len + image_token_len + 1, (i+1) * total_token_len)))
    decode_indices = jnp.asarray(decode_indices)
    print(f'decode_indices shape: {decode_indices.shape}')
    return decode_indices


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[Any, _model.Actions],
    decode_indices: jax.Array | None = None,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(model, rng, observation, actions):
        # Reasoning-RICL models expose an aux dict (ce / seg loss breakdown); others return loss only.
        if hasattr(model, "compute_loss_with_aux"):
            per_example_loss, aux = model.compute_loss_with_aux(rng, observation, actions, train=True)
            return jnp.mean(per_example_loss), {k: v for k, v in aux.items() if k != "seg_pred"}
        chunked_loss = model.compute_loss(rng, observation, actions, train=True, decode_indices=decode_indices)
        return jnp.mean(chunked_loss), {}

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        **aux,  # ce_loss / seg_loss scalars for reasoning-RICL; empty otherwise.
    }
    return new_state, info


# How often (in steps) to log reasoning-token mask reconstruction overlays to wandb.
SEG_VIZ_INTERVAL = 250
# Number of examples from the fixed viz batch to render each time.
SEG_VIZ_NUM_EXAMPLES = 4


def _soft_overlay(rgb_u8, grid, color, grid_size, alpha=0.6):
    """Blend a soft [0,1] patch grid over an RGB image as a colored overlay (viz only)."""
    h, w, _ = rgb_u8.shape
    full = np.kron(grid.reshape(grid_size, grid_size), np.ones((h // grid_size, w // grid_size)))[..., None]
    color = np.asarray(color, np.float32)
    out = rgb_u8.astype(np.float32) * (1.0 - alpha * full) + color * (alpha * full)
    return out.clip(0, 255).astype(np.uint8)


def _to_uint8(images_m1p1):
    """(n,H,W,3) in [-1,1] -> uint8 [0,255]."""
    return ((np.asarray(images_m1p1) + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)


def _build_seg_overlays(query_images, pred_patches, gt_patches, grid_size, nn_images_list=None, nn_flag_list=None):
    """One wandb.Image per example, panels left->right:
        [ NN0(flag) | NN1(flag) | ... | query | GT(green) | pred(red) ]

    Each NN panel is a retrieved neighbor's top image with its flagged object overlaid (yellow) —
    i.e. exactly what the model gets as in-context evidence. The query trio shows where the object
    IS (GT, green) vs where the reasoning tokens THINK it is (pred, red). Over training the red
    should converge onto the green. NN panels are included only when nn_* are provided.

    query_images / nn_images_list[j]: (n,H,W,3) in [-1,1]; pred/gt/nn_flag patches: (n,P) in [0,1].
    """
    images = []
    n, h, w, _ = query_images.shape
    rgb = _to_uint8(query_images)
    nn_rgb_list = None if nn_images_list is None else [_to_uint8(x) for x in nn_images_list]
    sep = np.full((h, 3, 3), 255, np.uint8)  # white separator columns
    for i in range(n):
        pred = np.asarray(pred_patches[i]).reshape(grid_size, grid_size)
        gt = np.asarray(gt_patches[i]).reshape(grid_size, grid_size)
        panels, caption = [], []
        if nn_rgb_list is not None and nn_flag_list is not None:
            for j, nn_rgb in enumerate(nn_rgb_list):
                flag = np.asarray(nn_flag_list[j][i]).reshape(grid_size, grid_size)
                panels += [_soft_overlay(nn_rgb[i], flag, (255, 255, 0), grid_size), sep]
                caption.append(f"NN{j}(flag)")
        panels += [rgb[i], sep, _soft_overlay(rgb[i], gt, (0, 255, 0), grid_size), sep,
                   _soft_overlay(rgb[i], pred, (255, 0, 0), grid_size)]
        caption += ["query", "GT(green)", "pred(red)"]
        images.append(wandb.Image(np.concatenate(panels, axis=1), caption=f"ex{i}: " + " | ".join(caption)))
    return images


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        num_workers=config.num_workers,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    decode_indices = create_decode_indices(config)

    ptrain_step = jax.jit(
        functools.partial(train_step, config, decode_indices=decode_indices),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    # Reasoning-RICL: set up periodic mask-reconstruction visualization on a fixed batch.
    is_reasoning = isinstance(config.model, _pi0_fast_reasoning_ricl.Pi0FASTReasoningRiclConfig)
    pseg_step = None
    viz_observation = None
    seg_grid_size = 0
    if is_reasoning:
        def _predict_seg(state, observation):
            model = nnx.merge(state.model_def, state.params)
            model.eval()
            return model.predict_seg_mask(observation)

        pseg_step = jax.jit(
            _predict_seg,
            in_shardings=(train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )
        viz_observation = batch[0]  # hold the first batch fixed so progress is comparable across steps
        seg_grid_size = int(round(config.model.num_seg_patches**0.5))

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            logging.info(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []

        if is_reasoning and (step % SEG_VIZ_INTERVAL == 0):
            with sharding.set_mesh(mesh):
                seg_pred = pseg_step(train_state, viz_observation)
            n = min(SEG_VIZ_NUM_EXAMPLES, seg_pred.shape[0])
            num_nn = config.model.num_retrieved_observations

            # Per-object model emits (b, N, P) for seg_pred / target / flags; collapse the object
            # axis (max over objects) to a single (b, P) mask for the overlay panels.
            def _collapse_objs(arr):
                arr = jax.device_get(arr)
                return arr.max(axis=1) if arr.ndim == 3 else arr

            nn_images_list = [
                jax.device_get(getattr(viz_observation, f"retrieved_{j}_images")["base_0_rgb"][:n])
                for j in range(num_nn)
            ]
            nn_flag_list = [
                _collapse_objs(getattr(viz_observation, f"retrieved_{j}_flag_mask")[:n]) for j in range(num_nn)
            ]
            overlays = _build_seg_overlays(
                jax.device_get(viz_observation.query_images["base_0_rgb"][:n]),
                _collapse_objs(seg_pred[:n]),
                _collapse_objs(viz_observation.query_seg_target[:n]),
                seg_grid_size,
                nn_images_list=nn_images_list,
                nn_flag_list=nn_flag_list,
            )
            wandb.log({"seg/overlay": overlays}, step=step)

        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
