"""Unit tests for the JAX low-precision primitives (run: PYTHONPATH=. python tests/test_quant.py)."""

import jax
import jax.numpy as jnp
import numpy as np

from lpsgld_jax.quant import (
    FixedPoint,
    block_quantize,
    fixed_point_quantize,
    float_quantize,
    make_quantizer,
    straight_through,
)


def test_fixed_point_nearest_on_grid_and_clamped():
    wl, fl = 8, 3
    step = 2.0 ** -fl
    x = jnp.linspace(-20, 20, 501)
    q = fixed_point_quantize(x, wl, fl, "nearest")
    # every output is an integer multiple of the step
    assert np.allclose(np.asarray(q) / step, np.round(np.asarray(q) / step), atol=1e-4)
    # clamped to the representable range
    assert float(q.min()) >= -(2.0 ** (wl - fl - 1)) - 1e-6
    assert float(q.max()) <= 2.0 ** (wl - fl - 1) - step + 1e-6
    print("[ok] fixed-point nearest: on-grid + clamped")


def test_fixed_point_stochastic_unbiased():
    wl, fl = 8, 3
    x = jnp.array([0.31, -1.27, 2.05, 0.5 * 2**-fl])  # non-grid values
    keys = jax.random.split(jax.random.key(0), 20000)
    qs = jax.vmap(lambda k: fixed_point_quantize(x, wl, fl, "stochastic", k))(keys)
    mean = np.asarray(qs.mean(0))
    assert np.allclose(mean, np.asarray(x), atol=2e-3), mean - np.asarray(x)
    # outputs land only on the two surrounding grid points
    step = 2.0 ** -fl
    lo = np.floor(np.asarray(x) / step) * step
    within = (np.asarray(qs) >= lo - 1e-6) & (np.asarray(qs) <= lo + step + 1e-6)
    assert within.all()
    print("[ok] fixed-point stochastic: unbiased E[q(x)]=x, neighbours only")


def test_block_quantize_representable_and_zero():
    x = jax.random.normal(jax.random.key(1), (16, 32)) * 3.0
    q = block_quantize(x, 8, "nearest", dim=0)
    assert q.shape == x.shape and bool(jnp.isfinite(q).all())
    # a max-magnitude entry per row should be near-exact (mantissa has 8 bits)
    assert float(jnp.max(jnp.abs(q - x)) / jnp.max(jnp.abs(x))) < 0.05
    # all-zero input passes through unchanged
    z = jnp.zeros((4, 4))
    assert np.allclose(np.asarray(block_quantize(z, 8, "nearest")), 0.0)
    print("[ok] block-FP: representable, zero-safe")


def test_block_stochastic_unbiased():
    x = jax.random.normal(jax.random.key(2), (8, 8))
    keys = jax.random.split(jax.random.key(3), 8000)
    qs = jax.vmap(lambda k: block_quantize(x, 6, "stochastic", k, dim=0))(keys)
    assert np.allclose(np.asarray(qs.mean(0)), np.asarray(x), atol=5e-3)
    print("[ok] block-FP stochastic: unbiased")


def test_float_nearest_idempotent():
    x = jax.random.normal(jax.random.key(4), (100,))
    q = float_quantize(x, exp=5, man=2, rounding="nearest")
    q2 = float_quantize(q, exp=5, man=2, rounding="nearest")
    assert np.allclose(np.asarray(q), np.asarray(q2), atol=0)  # quantizing twice is stable
    print("[ok] float nearest: idempotent")


def test_float_stochastic_unbiased():
    x = jnp.array([0.371, 1.62, -0.913, 3.14])
    keys = jax.random.split(jax.random.key(5), 20000)
    qs = jax.vmap(lambda k: float_quantize(x, 5, 2, "stochastic", k))(keys)
    assert np.allclose(np.asarray(qs.mean(0)), np.asarray(x), atol=5e-3)
    print("[ok] float stochastic: unbiased")


def test_straight_through_gradient():
    f = lambda x: jnp.sum(straight_through(x, fixed_point_quantize(x, 8, 3, "nearest")) ** 2)
    x = jnp.array([0.3, 1.7, -2.1])
    g = jax.grad(f)(x)
    q = fixed_point_quantize(x, 8, 3, "nearest")
    assert np.allclose(np.asarray(g), np.asarray(2 * q), atol=1e-5)  # d/dx of q^2 via ST = 2q
    print("[ok] straight-through gradient is identity")


def test_make_quantizer_matches():
    x = jax.random.normal(jax.random.key(6), (10,))
    q = make_quantizer(FixedPoint(8, 3), "nearest")(x)
    assert np.allclose(np.asarray(q), np.asarray(fixed_point_quantize(x, 8, 3, "nearest")))
    print("[ok] make_quantizer dispatch")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall quant tests passed")
