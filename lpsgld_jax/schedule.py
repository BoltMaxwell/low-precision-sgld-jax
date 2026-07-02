"""Learning-rate schedules (port of ``bnn/utils.py:adjust_learning_rate``).

* ``decay``  -- SWA-style piecewise: constant for the first half, linear ramp down to
  ``1%`` between 50%-90% of training, then constant ``1%``.
* ``cyclic`` -- ``M`` cosine cycles: ``factor = 0.5*(cos(pi * (t mod L)/L) + 1)``.

``build_schedule`` returns a jittable ``sched(step_id) -> lr`` for use inside
``lax.scan`` (``step_id`` is the global step ``epoch*num_batch + batch``).
"""

import jax.numpy as jnp


def build_schedule(lr_init, lr_type, *, total_steps, num_batch, epochs, M=1):
    if lr_type == "cyclic":
        cycle_len = total_steps // M

        def sched(step_id):
            cos_inner = jnp.pi * (step_id % cycle_len) / cycle_len
            return lr_init * 0.5 * (jnp.cos(cos_inner) + 1.0)

    elif lr_type == "decay":
        ratio = 0.01

        def sched(step_id):
            t = (step_id // num_batch) / epochs
            factor = jnp.where(
                t <= 0.5, 1.0,
                jnp.where(t <= 0.9, 1.0 - (1.0 - ratio) * (t - 0.5) / 0.4, ratio),
            )
            return lr_init * factor

    else:
        raise ValueError(f"unknown lr_type {lr_type!r}")

    return sched


def save_epochs(epochs, lr_type, M, num_savemodel, noise):
    """Set of epoch indices at which to snapshot a sample (mirrors bnn/train.py)."""
    if lr_type == "cyclic":
        per_cycle = max(num_savemodel // M, 1)
        cycle_len = epochs // M
        return {e for e in range(epochs) if (e % cycle_len) >= (cycle_len - per_cycle)}
    if noise:  # decay + SGLD: save the last num_savemodel epochs
        return set(range(max(epochs - num_savemodel, 0), epochs))
    return {epochs - 1}  # plain SGD: keep the final model
