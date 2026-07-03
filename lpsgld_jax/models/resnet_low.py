"""Fully low-precision ResNet18 in Equinox (port of ``models/resnet_low.py``).

Inserts an :class:`LPQuant` at each activation site (2 in the stem, 3 per BasicBlock).
Each quantizes the forward activation to ``wl_act`` bits and, via ``custom_vjp``, the
backward gradient (error) to ``wl_err`` bits -- so the whole network runs low-precision,
not just the weights/grads (which the optimizer handles).

A single ``key`` is threaded through the forward; each quant site folds in its unique
``qid`` for independent stochastic rounding. Weights are quantized by the optimizer, so
this model is used together with ``optim_lp`` (which quantizes weights/grads).

    model, state = make_resnet18lp(key, num_classes=10, wl_act=8, wl_err=8)
    logits, state = jax.vmap(model, axis_name="batch", in_axes=(0, None, 0),
                             out_axes=(0, None))(x, state, keys)
"""

from typing import List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from ..quant import lp_quant
from .resnet import _bn


class LPQuant(eqx.Module):
    wl_act: int = eqx.field(static=True)
    wl_err: int = eqx.field(static=True)
    rounding: str = eqx.field(static=True)
    qid: int = eqx.field(static=True)

    def __call__(self, x, key):
        return lp_quant(x, jax.random.fold_in(key, self.qid),
                        self.wl_act, self.wl_err, self.rounding)


class _Counter:
    """Mints LPQuant modules with unique ``qid``s during construction."""

    def __init__(self, wl_act, wl_err, rounding):
        self.cfg = (wl_act, wl_err, rounding)
        self.n = 0

    def make(self):
        q = LPQuant(*self.cfg, qid=self.n)
        self.n += 1
        return q


class BasicBlockLP(eqx.Module):
    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    conv2: eqx.nn.Conv2d
    bn2: eqx.nn.BatchNorm
    sc_conv: Optional[eqx.nn.Conv2d]
    sc_bn: Optional[eqx.nn.BatchNorm]
    q1: LPQuant
    q2: LPQuant
    q3: LPQuant
    expansion = 1

    def __init__(self, in_planes, planes, stride, counter, key):
        k1, k2, ks = jax.random.split(key, 3)
        self.conv1 = eqx.nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, use_bias=False, key=k1)
        self.bn1 = _bn(planes)
        self.conv2 = eqx.nn.Conv2d(planes, planes, 3, stride=1, padding=1, use_bias=False, key=k2)
        self.bn2 = _bn(planes)
        if stride != 1 or in_planes != planes:
            self.sc_conv = eqx.nn.Conv2d(in_planes, planes, 1, stride=stride, use_bias=False, key=ks)
            self.sc_bn = _bn(planes)
        else:
            self.sc_conv = self.sc_bn = None
        self.q1, self.q2, self.q3 = counter.make(), counter.make(), counter.make()

    def __call__(self, x, state, key):
        h = self.q1(self.conv1(x), key)
        h, state = self.bn1(h, state)
        h = self.q2(jax.nn.relu(h), key)
        h = self.q3(self.conv2(h), key)
        h, state = self.bn2(h, state)
        if self.sc_conv is not None:
            sc, state = self.sc_bn(self.sc_conv(x), state)
        else:
            sc = x
        return jax.nn.relu(h + sc), state


class ResNetLP(eqx.Module):
    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    q_stem1: LPQuant
    q_stem2: LPQuant
    blocks: List[BasicBlockLP]
    linear: eqx.nn.Linear

    def __init__(self, num_blocks, num_classes, wl_act, wl_err, rounding, key):
        counter = _Counter(wl_act, wl_err, rounding)
        keys = jax.random.split(key, 6)
        self.conv1 = eqx.nn.Conv2d(3, 64, 3, stride=1, padding=1, use_bias=False, key=keys[0])
        self.bn1 = _bn(64)
        self.q_stem1, self.q_stem2 = counter.make(), counter.make()
        blocks, in_planes = [], 64
        for planes, n, stride, lk in zip((64, 128, 256, 512), num_blocks, (1, 2, 2, 2), keys[1:5]):
            for s, bk in zip([stride] + [1] * (n - 1), jax.random.split(lk, n)):
                blocks.append(BasicBlockLP(in_planes, planes, s, counter, bk))
                in_planes = planes
        self.blocks = blocks
        self.linear = eqx.nn.Linear(512, num_classes, key=keys[5])

    def __call__(self, x, state, key):
        h = self.q_stem1(self.conv1(x), key)
        h, state = self.bn1(h, state)
        h = self.q_stem2(jax.nn.relu(h), key)
        for block in self.blocks:
            h, state = block(h, state, key)
        h = jnp.mean(h, axis=(1, 2))
        return self.linear(h), state


def make_resnet18lp(key, num_classes=10, wl_act=8, wl_err=8, rounding="stochastic"):
    return eqx.nn.make_with_state(ResNetLP)(
        [2, 2, 2, 2], num_classes, wl_act, wl_err, rounding, key=key)
