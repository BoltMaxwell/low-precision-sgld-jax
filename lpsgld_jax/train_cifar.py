"""Low-precision SGLD training on CIFAR-10/100 (port of ``bnn/train.py``, core scope).

One ``lax.scan`` per epoch (BN state threaded through the carry). The forward pass uses
quantized weights (:func:`optim_lp.forward_quant`); the LP-SGLD step quantizes the
gradient, takes the weight-decay step, and injects Langevin noise -- as Gaussian noise
(``sgldlp_f``/``naive``) or variance-corrected quantization (``vc``).

Deferred (documented): fully low-precision layers (activation/error quantization) and
block-FP VC; core scope uses fixed-point weights/grads/accumulator.

    python -m lpsgld_jax.train_cifar --data_path DATA --dir runs/vc \
        --variant vc --dataset cifar10 --lr_type cyclic
"""

import argparse
import os
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .data import load_cifar, make_augment
from .models.resnet import make_resnet18
from .optim_lp import make_lp_update
from .schedule import build_schedule, save_epochs


def make_loss_fn(static):
    def loss_fn(params, state, x, y):
        model = eqx.combine(params, static)
        batched = jax.vmap(model, axis_name="batch", in_axes=(0, None), out_axes=(0, None))
        logits, new_state = batched(x, state)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        return loss, new_state

    return eqx.filter_value_and_grad(loss_fn, has_aux=True)


def build_epoch_step(static, augment, schedule_fn, forward_quant, update):
    grad_fn = make_loss_fn(static)

    def step(carry, inp):
        params, state, key = carry
        step_id, x_raw, y = inp
        key, k_aug, k_fwd, k_upd = jax.random.split(key, 4)
        x = augment(k_aug, x_raw)
        fwd_params = forward_quant(k_fwd, params)
        (loss, state), grads = grad_fn(fwd_params, state, x, y)
        params = update(k_upd, params, grads, schedule_fn(step_id))
        return (params, state, key), loss

    return step


def _batch_epoch(rng, x, y, batch_size):
    n = (x.shape[0] // batch_size) * batch_size
    perm = np.asarray(jax.random.permutation(rng, x.shape[0]))[:n]
    return x[perm].reshape(-1, batch_size, *x.shape[1:]), y[perm].reshape(-1, batch_size)


def train(
    data,
    *,
    variant="vc",
    dataset="cifar10",
    epochs=245,
    batch_size=128,
    lr_init=0.5,
    weight_decay=5e-4,
    temperature=0.001,
    wl=8,
    fl=8,
    number="fixed",
    lr_type="cyclic",
    M=7,
    num_savemodel=35,
    noise=True,
    seed=1,
    save_dir=None,
    log_every=5,
):
    key = jax.random.key(seed)
    key, model_key = jax.random.split(key)
    model, state = make_resnet18(model_key, num_classes=data["num_classes"])
    params, static = eqx.partition(model, eqx.is_inexact_array)

    x_train, y_train = data["x_train"], data["y_train"]
    datasize = x_train.shape[0]
    num_batch = datasize // batch_size
    total_steps = epochs * num_batch

    augment = make_augment("cifar100" if data["num_classes"] == 100 else "cifar10")
    schedule_fn = build_schedule(lr_init, lr_type, total_steps=total_steps,
                                 num_batch=num_batch, epochs=epochs, M=M)
    forward_quant, update = make_lp_update(variant, wl, fl, weight_decay=weight_decay,
                                           temperature=temperature, datasize=datasize,
                                           noise=noise, number=number)
    step = build_epoch_step(static, augment, schedule_fn, forward_quant, update)
    run_epoch = eqx.filter_jit(partial(jax.lax.scan, step))
    to_save = save_epochs(epochs, lr_type, M, num_savemodel, noise)

    samples = []
    for epoch in range(epochs):
        key, shuffle_key, epoch_key = jax.random.split(key, 3)
        xb, yb = _batch_epoch(shuffle_key, x_train, y_train, batch_size)
        step_ids = epoch * num_batch + jnp.arange(xb.shape[0])
        carry, losses = run_epoch((params, state, epoch_key),
                                  (step_ids, jnp.asarray(xb), jnp.asarray(yb)))
        params, state, _ = carry
        if epoch % log_every == 0:
            print(f"epoch {epoch:3d}  loss {float(jnp.mean(losses)):.4f}")
        if epoch in to_save:
            model_s = eqx.combine(params, static)
            samples.append((model_s, state))
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                eqx.tree_serialise_leaves(
                    os.path.join(save_dir, f"sample_{len(samples) - 1:03d}.eqx"), (model_s, state))
    return samples


def main():
    p = argparse.ArgumentParser(description="Low-precision SGLD on CIFAR (JAX/Equinox)")
    p.add_argument("--data_path", required=True)
    p.add_argument("--dir", dest="save_dir", required=True)
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--variant", default="vc", choices=["sgldlp_f", "naive", "vc"])
    p.add_argument("--epochs", type=int, default=245)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--temperature", type=float, default=0.001)
    p.add_argument("--wl", type=int, default=8)
    p.add_argument("--fl", type=int, default=8)
    p.add_argument("--number", default="fixed", choices=["fixed", "block"])
    p.add_argument("--lr_type", default="cyclic", choices=["cyclic", "decay"])
    p.add_argument("--M", type=int, default=7)
    p.add_argument("--num_savemodel", type=int, default=35)
    p.add_argument("--no_noise", action="store_true", help="plain SGD (no Langevin noise)")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    data = load_cifar(args.data_path, args.dataset)
    train(data, variant=args.variant, dataset=args.dataset, epochs=args.epochs,
          batch_size=args.batch_size, lr_init=args.lr, weight_decay=args.wd,
          temperature=args.temperature, wl=args.wl, fl=args.fl, number=args.number,
          lr_type=args.lr_type,
          M=args.M, num_savemodel=args.num_savemodel, noise=not args.no_noise,
          seed=args.seed, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
