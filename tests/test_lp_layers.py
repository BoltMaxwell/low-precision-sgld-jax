"""Tests for fully low-precision layers: activation (fwd) + error (bwd) quantization.

    JAX_PLATFORMS=cpu PYTHONPATH=. python tests/test_lp_layers.py
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from lpsgld_jax.models.resnet_low import make_resnet18lp
from lpsgld_jax.quant import block_quantize, lp_quant


def test_lp_quant_forward_and_backward():
    x = jax.random.normal(jax.random.key(0), (3, 8)) * 2.0
    key = jax.random.key(1)
    # forward quantizes the activation
    y = lp_quant(x, key, 4, 4, "stochastic")
    assert float(jnp.max(jnp.abs(y - x))) > 0
    # backward quantizes the incoming gradient (error) to wl_err bits
    upstream = jnp.arange(24.0).reshape(3, 8)
    g = jax.grad(lambda z: jnp.sum(lp_quant(z, key, 4, 4, "stochastic") * upstream))(x)
    expected = block_quantize(upstream, 4, "stochastic", jax.random.split(key)[1], dim=None)
    assert np.allclose(np.asarray(g), np.asarray(expected))
    print("[ok] lp_quant: forward activation quant + backward error quant")


def test_lp_quant_full_precision_passthrough():
    x = jax.random.normal(jax.random.key(2), (4,))
    y = lp_quant(x, jax.random.key(0), -1, -1, "stochastic")  # -1 = full precision
    assert np.allclose(np.asarray(y), np.asarray(x))
    print("[ok] lp_quant: wl=-1 is a passthrough")


def test_lp_resnet_forward_and_grad():
    model, state = make_resnet18lp(jax.random.key(0), num_classes=10, wl_act=8, wl_err=8)
    params, static = eqx.partition(model, eqx.is_inexact_array)
    x = jax.random.normal(jax.random.key(1), (8, 3, 32, 32))
    keys = jax.random.split(jax.random.key(2), 8)

    def loss(p):
        m = eqx.combine(p, static)
        logits, _ = jax.vmap(m, axis_name="batch", in_axes=(0, None, 0), out_axes=(0, None))(x, state, keys)
        return jnp.mean((logits - 1.0) ** 2)

    g = eqx.filter_grad(loss)(params)
    leaves = jax.tree_util.tree_leaves(g)
    assert all(bool(jnp.isfinite(v).all()) for v in leaves)
    assert sum(float(jnp.sum(jnp.abs(v))) for v in leaves) > 0  # gradient flows through quant
    print("[ok] ResNet18LP: forward + gradient through activation/error quantization")


if __name__ == "__main__":
    test_lp_quant_forward_and_backward()
    test_lp_quant_full_precision_passthrough()
    test_lp_resnet_forward_and_grad()
    print("\nlow-precision layer tests passed")
