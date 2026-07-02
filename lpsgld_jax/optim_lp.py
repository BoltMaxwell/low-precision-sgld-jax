"""Low-precision SGLD update (port of ``bnn/optim.py`` OptimLP, core scope).

Three variants (fixed-point number format; block-FP weight/grad is a future extension,
block-FP VC is deferred with the low-precision layers):

* ``sgldlp_f`` -- full-precision gradient accumulator, low-precision weights/grads.
  The forward pass uses quantized weights; the update runs on the full-precision
  accumulator with Gaussian Langevin noise.
* ``naive``   -- low-precision accumulator; quantize the whole noisy update (biases
  the stationary variance -- see the Gaussian toy).
* ``vc``      -- low-precision accumulator with variance-corrected quantization
  (:func:`lpsgld_jax.vc.Q_vc`): injects the Langevin noise *and* quantizes in one
  variance-preserving step.

Each returns ``(forward_quant, update)``:
  * ``forward_quant(key, params)`` -> the weights to use in the forward pass.
  * ``update(key, params, grads, lr)`` -> new params (the accumulator).
Quantization is applied per pytree leaf with an independent stochastic-rounding key.
"""

import jax
import jax.numpy as jnp

from .quant import fixed_point_quantize
from .vc import Q_vc


def _tree_map_keyed(fn, tree, key):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    return jax.tree_util.tree_unflatten(treedef, [fn(x, k) for x, k in zip(leaves, keys)])


def _tree_gaussian(key, tree, std):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    noise = [std * jax.random.normal(k, x.shape, x.dtype) for x, k in zip(leaves, keys)]
    return jax.tree_util.tree_unflatten(treedef, noise)


def make_lp_update(variant, wl, fl, *, weight_decay, temperature, datasize, noise=True):
    def qfix(x, k):
        return fixed_point_quantize(x, wl, fl, "stochastic", k)

    def forward_quant(key, params):
        # SGLDLP-F keeps a full-precision accumulator, so quantize for the forward
        # pass; the -L variants already hold low-precision weights.
        if variant == "sgldlp_f":
            return _tree_map_keyed(qfix, params, key)
        return params

    def update(key, params, grads, lr):
        kg, kn, kw = jax.random.split(key, 3)
        qgrads = _tree_map_keyed(qfix, grads, kg)
        var = 2.0 * lr * temperature / datasize if noise else 0.0

        def gd(p, g):  # gradient + weight-decay step
            return p - lr * (g + weight_decay * p)

        if variant == "sgldlp_f":
            stepped = jax.tree_util.tree_map(gd, params, qgrads)
            if noise:
                gn = _tree_gaussian(kn, params, jnp.sqrt(var))
                stepped = jax.tree_util.tree_map(lambda p, n: p + n, stepped, gn)
            return stepped  # full-precision accumulator
        if variant == "naive":
            stepped = jax.tree_util.tree_map(gd, params, qgrads)
            if noise:
                gn = _tree_gaussian(kn, params, jnp.sqrt(var))
                stepped = jax.tree_util.tree_map(lambda p, n: p + n, stepped, gn)
            return _tree_map_keyed(qfix, stepped, kw)  # low-precision accumulator
        if variant == "vc":
            mu = jax.tree_util.tree_map(gd, params, qgrads)
            if not noise:
                return _tree_map_keyed(qfix, mu, kw)
            return _tree_map_keyed(lambda x, k: Q_vc(k, x, var, wl, fl), mu, kw)
        raise ValueError(f"unknown variant {variant!r}")

    return forward_quant, update
