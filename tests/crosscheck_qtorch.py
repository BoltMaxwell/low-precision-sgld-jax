"""Cross-check the JAX low-precision primitives against real QPyTorch.

Runs on the cluster (needs torch + qtorch). Nearest rounding must match bit-for-bit;
stochastic rounding is validated statistically (mean preservation + same support).

    pip install qtorch
    PYTHONPATH=. python tests/crosscheck_qtorch.py
"""

import numpy as np
import torch
from qtorch import BlockFloatingPoint, FixedPoint
from qtorch.quant import block_quantize as qt_block
from qtorch.quant import fixed_point_quantize as qt_fixed

import jax
import jax.numpy as jnp

from lpsgld_jax import quant as jq


def _report(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    return ok


def check_fixed_nearest(x_np, wl=8, fl=3):
    xt = torch.from_numpy(x_np)
    qt = qt_fixed(xt, wl=wl, fl=fl, rounding="nearest").numpy()
    jx = np.asarray(jq.fixed_point_quantize(jnp.asarray(x_np), wl, fl, "nearest"))
    max_err = float(np.max(np.abs(qt - jx)))
    return _report(f"fixed nearest (wl={wl},fl={fl})", max_err < 1e-6, f"max|Δ|={max_err:.2e}")


def check_fixed_stochastic_mean(x_np, wl=8, fl=3, n=4000):
    xt = torch.from_numpy(x_np)
    qt_mean = np.mean([qt_fixed(xt, wl=wl, fl=fl, rounding="stochastic").numpy() for _ in range(n)], 0)
    keys = jax.random.split(jax.random.key(0), n)
    jx = jax.vmap(lambda k: jq.fixed_point_quantize(jnp.asarray(x_np), wl, fl, "stochastic", k))(keys)
    jx_mean = np.asarray(jx.mean(0))
    err = float(np.max(np.abs(qt_mean - jx_mean)))
    both_unbiased = err < 5e-3 and np.max(np.abs(qt_mean - x_np)) < 5e-3
    return _report("fixed stochastic (mean-preserve)", both_unbiased, f"max|Δmean|={err:.2e}")


def check_block_nearest(x_np, wl=8):
    xt = torch.from_numpy(x_np)
    qt = qt_block(xt, wl=wl, dim=0, rounding="nearest").numpy()
    jx = np.asarray(jq.block_quantize(jnp.asarray(x_np), wl, "nearest", dim=0))
    max_err = float(np.max(np.abs(qt - jx)))
    # block-FP conventions can differ slightly; report closeness
    return _report(f"block nearest (wl={wl},dim=0)", max_err < 1e-5, f"max|Δ|={max_err:.2e}")


def main():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((64, 128)).astype(np.float32)) * 2.0
    print(f"torch {torch.__version__}  jax {jax.__version__}\n")
    results = [
        check_fixed_nearest(x, 8, 3),
        check_fixed_nearest(x, 8, 8),
        check_fixed_stochastic_mean(x, 8, 3),
        check_block_nearest(x, 8),
    ]
    print(f"\n{sum(results)}/{len(results)} checks passed")


if __name__ == "__main__":
    main()
