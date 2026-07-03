"""Bayesian model averaging over saved low-precision SGLD samples (port of
``bnn/ens.py``): softmax-average the predictions, report BMA accuracy and expected
calibration error (ECE).

    python -m lpsgld_jax.ensemble --data_path DATA --dir runs/vc --dataset cifar10
"""

import argparse
import glob
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .data import load_cifar
from .models.resnet import make_resnet18
from .models.resnet_low import make_resnet18lp


@eqx.filter_jit
def _probs(model, state, x):
    model = eqx.nn.inference_mode(model)
    batched = jax.vmap(model, axis_name="batch", in_axes=(0, None), out_axes=(0, None))
    logits, _ = batched(x, state)
    return jax.nn.softmax(logits, axis=-1)


@eqx.filter_jit
def _probs_lp(model, state, x, akeys):
    model = eqx.nn.inference_mode(model)
    batched = jax.vmap(model, axis_name="batch", in_axes=(0, None, 0), out_axes=(0, None))
    logits, _ = batched(x, state, akeys)
    return jax.nn.softmax(logits, axis=-1)


def predict_mean(model, state, x_test, batch_size=200, lp_layers=False):
    out = []
    for i in range(0, x_test.shape[0], batch_size):
        xb = jnp.asarray(x_test[i:i + batch_size])
        if lp_layers:  # fixed keys -> reproducible eval-time activation quantization
            probs = _probs_lp(model, state, xb, jax.random.split(jax.random.key(i), xb.shape[0]))
        else:
            probs = _probs(model, state, xb)
        out.append(np.asarray(probs))
    return np.concatenate(out, axis=0)


def expected_calibration_error(y_true, y_prob, num_bins=10):
    """ECE (port of bnn/ens.py:expected_calibration_error)."""
    pred = y_prob.argmax(-1)
    correct = (pred == y_true).astype(np.float32)
    conf = y_prob.max(-1)
    bins = np.digitize(conf, np.linspace(0, 1.0, num_bins), right=True)
    o = 0.0
    for b in range(num_bins):
        mask = bins == b
        if mask.any():
            o += np.abs(np.sum(correct[mask] - conf[mask]))
    return o / y_prob.shape[0]


def evaluate(samples, x_test, y_test, batch_size=200, lp_layers=False):
    y_test = np.asarray(y_test)
    bma = np.zeros((x_test.shape[0], int(y_test.max()) + 1), dtype=np.float64)
    per_sample = []
    for model, state in samples:
        probs = predict_mean(model, state, x_test, batch_size, lp_layers)
        per_sample.append(float((probs.argmax(1) == y_test).mean()))
        bma += probs
    bma /= len(samples)
    acc = float((bma.argmax(1) == y_test).mean())
    ece = expected_calibration_error(y_test, bma)
    return acc, ece, per_sample


def load_samples(save_dir, num_classes, lp_layers=False, wl=8):
    paths = sorted(glob.glob(os.path.join(save_dir, "sample_*.eqx")))
    if not paths:
        raise FileNotFoundError(f"no sample_*.eqx files in {save_dir}")
    samples = []
    for path in paths:
        if lp_layers:
            skeleton = make_resnet18lp(jax.random.key(0), num_classes=num_classes, wl_act=wl, wl_err=wl)
        else:
            skeleton = make_resnet18(jax.random.key(0), num_classes=num_classes)
        samples.append(eqx.tree_deserialise_leaves(path, skeleton))
    return samples


def main():
    p = argparse.ArgumentParser(description="Low-precision SGLD ensemble evaluation")
    p.add_argument("--data_path", required=True)
    p.add_argument("--dir", dest="save_dir", required=True)
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--lp_layers", action="store_true")
    p.add_argument("--wl", type=int, default=8)
    args = p.parse_args()

    data = load_cifar(args.data_path, args.dataset)
    samples = load_samples(args.save_dir, data["num_classes"], args.lp_layers, args.wl)
    acc, ece, per_sample = evaluate(samples, data["x_test"], data["y_test"], lp_layers=args.lp_layers)
    print(f"loaded {len(samples)} samples")
    print(f"per-sample acc: mean {np.mean(per_sample):.4f}  "
          f"range [{min(per_sample):.4f}, {max(per_sample):.4f}]")
    print(f"BMA accuracy: {acc:.4f}   error%: {(1 - acc) * 100:.2f}   ECE%: {ece * 100:.2f}")


if __name__ == "__main__":
    main()
