"""Standard-Gaussian toy (JAX port of ``gaussian/gaussian.py``).

Low-precision SGLD sampling the standard normal (potential ``0.5*theta**2``, so the
gradient is ``theta``, estimated with noise ``sigma``). Compares three variants:

* ``f``     -- SGLDLP-F: full-precision accumulator, low-precision weight/grad.
* ``naive`` -- SGLDLP-L: quantize the whole update (biases the stationary variance).
* ``vc``    -- variance-corrected SGLDLP-L: matches N(0,1).

Run:
    python -m lpsgld_jax.gaussian                 # moment check for all three
    python -m lpsgld_jax.gaussian --plot          # also save figs/gaussian_jax.png
"""

import argparse
import functools

import jax
import jax.numpy as jnp
import numpy as np

from .quant import fixed_point_quantize as fq
from .vc import Q_vc

WL, FL = 8, 3
ALPHA = 2e-3
SIGMA = 0.1
VAR = 2 * ALPHA


def _sgrad(key, theta):
    """Stochastic gradient of 0.5*theta^2 (the standard normal potential)."""
    return theta + SIGMA * jax.random.normal(key, ())


def _step_f(carry, _):
    theta, key = carry
    key, kqt, kg, kqg, kn = jax.random.split(key, 5)
    g = _sgrad(kg, fq(theta, WL, FL, "stochastic", kqt))
    theta = theta - ALPHA * fq(g, WL, FL, "stochastic", kqg) + jnp.sqrt(VAR) * jax.random.normal(kn, ())
    return (theta, key), theta


def _step_naive(carry, _):
    theta, key = carry
    key, kg, kqg, kn, kq = jax.random.split(key, 5)
    g = fq(_sgrad(kg, theta), WL, FL, "stochastic", kqg)
    y = theta - ALPHA * g + jnp.sqrt(VAR) * jax.random.normal(kn, ())
    theta = fq(y, WL, FL, "stochastic", kq)
    return (theta, key), theta


def _step_vc(carry, _):
    theta, key = carry
    key, kg, kqg, kvc = jax.random.split(key, 4)
    g = fq(_sgrad(kg, theta), WL, FL, "stochastic", kqg)
    theta = Q_vc(kvc, theta - ALPHA * g, VAR, WL, FL)
    return (theta, key), theta


_STEPS = {"f": _step_f, "naive": _step_naive, "vc": _step_vc}


@functools.partial(jax.jit, static_argnums=(0, 2))
def run(variant, key, n_steps):
    (_, _), traj = jax.lax.scan(_STEPS[variant], (jnp.array(0.0), key), None, length=n_steps)
    return traj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    trajs = {}
    for i, variant in enumerate(("f", "naive", "vc")):
        traj = np.asarray(jax.block_until_ready(run(variant, jax.random.key(args.seed + i), args.steps)))
        trajs[variant] = traj
        burn = traj[len(traj) // 10:]  # drop 10% burn-in
        print(f"{variant:5s}: mean {burn.mean():+.4f}  std {burn.std():.4f}  "
              f"(target N(0,1): mean 0, std 1)")

    if args.plot:
        _plot(trajs)


def _plot(trajs):
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    os.makedirs("figs", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    xs = np.linspace(-4, 4, 400)
    for ax, variant in zip(axes, ("f", "naive", "vc")):
        burn = trajs[variant][len(trajs[variant]) // 10:]
        ax.hist(burn, bins=np.arange(-5, 5, 2.0 ** -FL), density=True, color="tab:red", alpha=0.7)
        ax.plot(xs, norm.pdf(xs), "k", lw=2, label="N(0,1)")
        ax.set_title(f"SGLD ({variant})")
        ax.set_xlim(-4, 4)
        ax.legend()
    fig.tight_layout()
    out = "figs/gaussian_jax.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
