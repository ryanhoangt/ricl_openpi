import dataclasses
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import optax

import openpi.shared.array_typing as at


@runtime_checkable
class LRScheduleConfig(Protocol):
    def create(self) -> optax.Schedule: ...


@dataclasses.dataclass(frozen=True)
class CosineDecaySchedule(LRScheduleConfig):
    """Cosine decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 2.5e-5
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6

    def create(self) -> optax.Schedule:
        return optax.warmup_cosine_decay_schedule(
            init_value=self.peak_lr / (self.warmup_steps + 1),
            peak_value=self.peak_lr,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
            end_value=self.decay_lr,
        )


@dataclasses.dataclass(frozen=True)
class RsqrtDecaySchedule(LRScheduleConfig):
    """Inverse square root decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 5e-5
    timescale: float = 10_000

    def create(self) -> optax.Schedule:
        return optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=self.peak_lr / (self.warmup_steps + 1),
                    end_value=self.peak_lr,
                    transition_steps=self.warmup_steps,
                ),
                lambda step: self.peak_lr / jnp.sqrt((self.timescale + step) / self.timescale),
            ],
            [self.warmup_steps],
        )


@runtime_checkable
class OptimizerConfig(Protocol):
    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation: ...


@dataclasses.dataclass(frozen=True)
class AdamW(OptimizerConfig):
    """AdamW optimizer."""

    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        tx = optax.adamw(
            lr, b1=self.b1, b2=self.b2, eps=self.eps, weight_decay=self.weight_decay, mask=weight_decay_mask
        )

        return optax.chain(optax.clip_by_global_norm(self.clip_gradient_norm), tx)


@dataclasses.dataclass(frozen=True)
class SGD(OptimizerConfig):
    """SGD optimizer."""

    lr: float = 5e-5
    momentum: float = 0.9
    nesterov: bool = False

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        assert weight_decay_mask is None, "Weight decay is not supported for SGD"
        return optax.sgd(lr, momentum=self.momentum, nesterov=self.nesterov)


@dataclasses.dataclass(frozen=True)
class MultiGroupAdamW(OptimizerConfig):
    """AdamW with separate learning rate scales for different parameter groups.

    Intended for the Perceiver-RICL fine-tune where the perceiver is randomly
    initialised (needs a higher LR to move fast) while the LLM is pre-trained
    (needs a lower LR to avoid forgetting).

    The main `lr` schedule (from `lr_schedule` in TrainConfig) is used as the
    *perceiver / non-LLM* learning rate.  LLM params receive `lr * llm_lr_scale`.

    Param group assignment is based on whether the parameter path contains 'llm'.
    This matches both PaliGemma.llm weights and any LoRA adapters inside it.
    """

    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0
    # LLM params receive lr * llm_lr_scale; perceiver + other params receive full lr.
    llm_lr_scale: float = 0.1

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        def _scale(schedule, scale):
            """Return a new schedule equal to schedule * scale."""
            if callable(schedule):
                return lambda step: schedule(step) * scale
            return schedule * scale

        def _adamw(lr_for_group):
            return optax.adamw(
                lr_for_group,
                b1=self.b1,
                b2=self.b2,
                eps=self.eps,
                weight_decay=self.weight_decay,
                mask=weight_decay_mask,
            )

        transforms = {
            "llm": _adamw(_scale(lr, self.llm_lr_scale)),
            "other": _adamw(lr),
        }

        def param_labels_fn(params):
            """Return a pytree with same structure as params; each leaf is a group label."""
            def label(path, _):
                return "llm" if any("llm" in str(k) for k in path) else "other"
            return jax.tree_util.tree_map_with_path(label, params)

        return optax.chain(
            optax.clip_by_global_norm(self.clip_gradient_norm),
            optax.multi_transform(transforms, param_labels_fn),
        )


def create_optimizer(
    optimizer: OptimizerConfig, lr_schedule: LRScheduleConfig, weight_decay_mask: at.PyTree | None = None
) -> optax.GradientTransformation:
    lr = lr_schedule.create()
    return optimizer.create(lr, weight_decay_mask=weight_decay_mask)
