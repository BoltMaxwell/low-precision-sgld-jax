"""Tests for block-FP variance-corrected quantization (Q_vc_block).

    PYTHONPATH=. python tests/test_vc_block.py
"""

import jax
import jax.numpy as jnp
import numpy as np

from lpsgld_jax.vc import Q_vc_block, _block_D_FL


def _multiscale_mu():
    rows = jnp.array([0.003, 0.03, 0.3, 3.0])  # per-row magnitudes span 3 orders
    return rows[:, None] * jax.random.normal(jax.random.key(0), (4, 64))


def test_block_grid_adapts_per_row():
    D, _ = _block_D_FL(_multiscale_mu(), 8)
    d = np.asarray(D[:, 0])
    # bigger-magnitude rows get a coarser (larger) step
    assert np.all(np.diff(d) > 0), d
    # small row is much finer than a global fixed-point fl=8 grid (2**-8)
    assert d[0] < 2**-8 < d[-1]
    print(f"[ok] block grid adapts per row: steps {d}")


def test_block_vc_unbiased():
    mu = _multiscale_mu()
    keys = jax.random.split(jax.random.key(1), 6000)
    qs = jax.vmap(lambda k: Q_vc_block(k, mu, 1e-4, 8))(keys)
    assert bool(jnp.isfinite(qs).all())
    # unbiased relative to the per-row scale (max step is 0.0625 here)
    assert float(jnp.max(jnp.abs(qs.mean(0) - mu))) < 3e-3
    print("[ok] block VC is mean-preserving")


if __name__ == "__main__":
    test_block_grid_adapts_per_row()
    test_block_vc_unbiased()
    print("\nblock VC tests passed")
