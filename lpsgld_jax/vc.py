"""Variance-corrected (VC) quantization -- the paper's core contribution.

A naive low-precision SGLD step adds Gaussian noise and then rounds, which distorts
the per-step noise variance. VC quantization instead produces a *quantized* sample
whose discrete distribution has exactly the target Langevin variance ``var``, so
low-precision SGLD keeps the correct stationary distribution.

Fixed-point path (``Q_vc``), ported from ``gaussian/gaussian.py:34-75`` and
``bnn/optim.py``. Two regimes for step ``D = 2**-fl`` and ``var_fix = D**2/4``
(the variance stochastic rounding already contributes at the half-step):

* ``var > var_fix`` : add the deficit as Gaussian noise, nearest-round, then add a
  sign-correlated discrete ``{+D,-D,0}`` correction (``_sample_mu``).
* ``var <= var_fix``: stochastically round (which itself has variance ``var_s``),
  then top up to ``var`` with discrete ``{+D,-D,0}`` noise (``_sample``).

All branches are computed and selected with ``jnp.where`` so ``var`` may be traced
(e.g. a cyclical learning rate).
"""

import jax
import jax.numpy as jnp

from .quant import fixed_point_quantize


def _sample(key, var, D):
    """Discrete noise in {+D, -D, 0} with variance ``var`` (P(+D)=P(-D)=var/(2 D^2))."""
    p1 = var / (2 * D**2)
    u = jax.random.uniform(key, jnp.shape(var))
    return jnp.where(u < p1, D, jnp.where(u < 2 * p1, -D, 0.0))


def _sample_mu(key, mu, var_fix, D):
    """Mean-correcting discrete noise for the residual regime (``mu = |residual|``)."""
    p1 = (var_fix + mu**2 + mu * D) / (2 * D**2)
    p2 = (var_fix + mu**2 - mu * D) / (2 * D**2)
    u = jax.random.uniform(key, jnp.shape(mu))
    return jnp.where(u < p1, D, jnp.where(u < p1 + p2, -D, 0.0))


def Q_vc(key, mu, var, wl, fl):
    """Variance-corrected fixed-point quantization of ``mu`` targeting variance ``var``."""
    D = 2.0 ** (-fl)
    var_fix = D**2 / 4.0
    kg, ksmu, kqmu, ks = jax.random.split(key, 4)

    # regime A: var > var_fix
    x = mu + jnp.sqrt(jnp.maximum(var - var_fix, 0.0)) * jax.random.normal(kg, jnp.shape(mu))
    quant_x = fixed_point_quantize(x, wl, fl, "nearest")
    res_a = x - quant_x
    theta_a = quant_x + jnp.sign(res_a) * _sample_mu(ksmu, jnp.abs(res_a), var_fix, D)

    # regime B: var <= var_fix
    quant_mu = fixed_point_quantize(mu, wl, fl, "stochastic", kqmu)
    res_b = mu - quant_mu
    p1 = jnp.abs(res_b) / D
    var_s = (1.0 - p1) * res_b**2 + p1 * (-res_b + jnp.sign(res_b) * D) ** 2
    theta_b = quant_mu + _sample(ks, jnp.maximum(var - var_s, 0.0), D)

    theta = jnp.where(var > var_fix, theta_a, theta_b)
    t_max = 2.0 ** (wl - fl - 1) - D
    return jnp.clip(theta, -(2.0 ** (wl - fl - 1)), t_max)
