import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    # Do not change this for your own dataset.
    action_dim: int

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # We only mask padding for pi0 model, not pi0-FAST. Do not change this for your own dataset.
        mask_padding = self.model_type == _model.ModelType.PI0

        # We pad the proprioceptive input to the action dimension of the model.
        # For pi0-FAST, we don't pad the state. For Libero, we don't need to differentiate
        # since the pi0-FAST action_dim = 7, which is < state_dim = 8, so pad is skipped.
        # Keep this for your own dataset, but if your dataset stores the proprioceptive input
        # in a different key than "observation/state", you should change it below.
        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Mask any non-existent images with False (if ``mask_padding`` is True).
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            # We are padding to the model action dim.
            # For pi0-FAST, this is a no-op (since action_dim = 7).
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :7])}


@dataclasses.dataclass(frozen=True)
class RiclLiberoInputs(transforms.DataTransformFn):
    """Prepare LIBERO observations for RICL training/inference."""

    action_dim: int
    num_retrieved_observations: int
    model_type: _model.ModelType = _model.ModelType.PI0_FAST

    def __call__(self, data: dict) -> dict:
        all_prefix = [f"retrieved_{i}_" for i in range(self.num_retrieved_observations)] + ["query_"]
        inputs_dicts = []

        for prefix in all_prefix:
            base_image = _parse_image(data[f"{prefix}top_image"])
            wrist_image = _parse_image(data[f"{prefix}wrist_image"])
            # Image key order MUST match LiberoInputs (used for SFT) so that
            # the SigLip positional slots are consistent across SFT → RICL:
            #   pos 0 = base_0_rgb (base camera)
            #   pos 1 = left_wrist_0_rgb (wrist camera)
            #   pos 2 = right_wrist_0_rgb (zeros / unused)
            # PI0_FAST does not mask padding images, so all masks are True.
            inputs_dicts.append(
                {
                    f"{prefix}state": data[f"{prefix}state"],
                    f"{prefix}image": {
                        "base_0_rgb": base_image,
                        "left_wrist_0_rgb": wrist_image,
                        "right_wrist_0_rgb": np.zeros_like(base_image),
                    },
                    f"{prefix}image_mask": {
                        "base_0_rgb": np.True_,
                        "left_wrist_0_rgb": np.True_,
                        "right_wrist_0_rgb": np.True_,
                    },
                }
            )

        inputs = {k: v for d in inputs_dicts for k, v in d.items()}

        for prefix in all_prefix[:-1]:
            inputs[f"{prefix}actions"] = data[f"{prefix}actions"]
        if "query_actions" in data:
            inputs["query_actions"] = data["query_actions"]

        for prefix in all_prefix:
            inputs[f"{prefix}prompt"] = data[f"{prefix}prompt"]

        if "exp_lamda_distances" in data:
            inputs["exp_lamda_distances"] = data["exp_lamda_distances"]
        if "inference_time" in data:
            inputs["inference_time"] = data["inference_time"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RiclLiberoOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["query_actions"])
        return {"actions": actions, "query_actions": actions}


@dataclasses.dataclass(frozen=True)
class TrajPerceiverLiberoInputs(transforms.DataTransformFn):
    """Prepare LIBERO observations for trajectory-perceiver training/inference.

    Processes query_top_image / query_wrist_image into the standard image dict
    format and passes traj_state / traj_mask through unchanged.
    """

    action_dim: int

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["query_top_image"])
        wrist_image = _parse_image(data["query_wrist_image"])

        inputs = {
            "query_state": data["query_state"],
            "query_image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "query_image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "query_prompt": data["query_prompt"],
            "traj_state": data["traj_state"],
            "traj_top_emb": data["traj_top_emb"],
            "traj_wrist_emb": data["traj_wrist_emb"],
            "traj_mask": data["traj_mask"],
        }
        if "query_actions" in data:
            inputs["query_actions"] = data["query_actions"]
        return inputs


@dataclasses.dataclass(frozen=True)
class TrajPerceiverLiberoOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["query_actions"])
        return {"actions": actions, "query_actions": actions}

