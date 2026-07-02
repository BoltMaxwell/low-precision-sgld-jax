# low-precision-sgld-jax

A [JAX](https://github.com/jax-ml/jax) reimplementation of **Low-Precision SGLD** for
Bayesian deep learning — the PyTorch original rebuilt on
[Equinox](https://github.com/patrick-kidger/equinox) (models), with the
**QPyTorch dependency replaced by a small pure-JAX quantization module**.

This is a reimplementation written with the assistance of
**[Claude Code](https://claude.com/claude-code)** (Anthropic). It ports the method — it
does not introduce it. All credit for the algorithm and the original code goes to the
authors:

> Ruqi Zhang, Andrew Gordon Wilson, Christopher De Sa.
> **Low-Precision Stochastic Gradient Langevin Dynamics.** ICML 2022.
> [paper](https://arxiv.org/pdf/2206.09909.pdf) · [original code](https://github.com/ruqizhang/low-precision-sgld)

```bibtex
@article{zhang2022lpsgld,
  title={Low-Precision Stochastic Gradient Langevin Dynamics},
  author={Zhang, Ruqi and Wilson, Andrew Gordon and De Sa, Christopher},
  journal={International Conference on Machine Learning},
  year={2022}
}
```

This repo contains **only the JAX reimplementation**. For the original PyTorch sources
(reference training scripts, the model zoo, the fully low-precision layers), see the
original repo: **https://github.com/ruqizhang/low-precision-sgld**.

## The QPyTorch replacement

The original depends on [QPyTorch](https://github.com/Tiiiger/QPyTorch) for
low-precision arithmetic simulation (fixed-point / block-floating-point / float with
nearest & stochastic rounding). Rather than port QPyTorch's CUDA kernels or adopt a
heavier JAX quantization library (AQT/Qwix, which target HW-accelerated INT8/FP8 QAT and
don't expose arbitrary fixed-point simulation), the whole dependency collapses to
**`lpsgld_jax/quant.py`** (~150 lines of elementwise JAX). The paper's actual novelty —
variance-corrected (VC) quantization — was always plain tensor math and lives in
`lpsgld_jax/vc.py`.

`quant.py` is validated **bit-for-bit against real QPyTorch** on GPU
(`tests/crosscheck_qtorch.py`): nearest rounding matches exactly (`max|Δ|=0`), stochastic
rounding matches by mean-preservation.

## The idea

Low-precision SGD is common; low-precision *sampling* is not. Naively rounding a noisy
SGLD update distorts the per-step noise variance and biases the stationary distribution.
**Variance-corrected (VC) quantization** produces a quantized sample whose discrete
distribution has exactly the target Langevin variance, so 8-bit SGLD matches
full-precision SGLD. Three variants:

| variant | accumulator | weights/grads |
|---|---|---|
| `sgldlp_f` (SGLDLP-F) | full precision | low precision |
| `naive`   (SGLDLP-L)  | low precision  | low precision (biased) |
| `vc`      (SGLDLP-L)  | low precision  | low precision, variance-corrected |

## Layout

```
lpsgld_jax/
  quant.py       fixed-point / block-FP / float quantize (nearest+stochastic) -- replaces QPyTorch
  vc.py          variance-corrected quantization (the paper's contribution)
  optim_lp.py    low-precision SGLD update (3 variants)
  schedule.py    decay + cyclic (M-cycle cosine) LR
  models/resnet.py, data.py, download.py   Equinox ResNet18 + CIFAR loader (reused from csgmcmc-jax)
  train_cifar.py, ensemble.py, gaussian.py
tests/           quant unit tests, QPyTorch cross-check, gaussian + CIFAR smoke
```

## Install & run

```bash
pip install -e .            # jax, equinox, optax, numpy
pip install -e '.[plot]'    # + matplotlib/scipy for figures

python -m lpsgld_jax.gaussian --plot                       # toy: reproduces figs/gaussian_jax.png
python -m lpsgld_jax.download --dataset cifar10 --root data
python -m lpsgld_jax.train_cifar --data_path data --dir runs/vc --variant vc --lr_type cyclic
python -m lpsgld_jax.ensemble  --data_path data --dir runs/vc
```

## Results

**Gaussian toy** (sampling N(0,1) at 8-bit). Naive low-precision SGLD inflates the
variance (std **1.30**); VC corrects it (std **1.00**) — reproducing the paper's figure:

![gaussian](figs/gaussian_jax.png)

**CIFAR-10 / ResNet18**, 8-bit, 245 epochs, ensemble of samples (H100):

| variant | BMA accuracy | error % | ECE % |
|---|---|---|---|
| SGLDLP-F (full-precision accumulator) | **94.79%** | 5.21 | 1.45 |
| VC SGLDLP-L | 93.32% | 6.68 | **1.44** |
| naive SGLDLP-L | 92.89% | 7.11 | 2.21 |

8-bit SGLD reaches ~95% (near full-precision SGLD). **VC beats naive** on both accuracy
and — the point of the method — **calibration** (ECE 1.44 vs 2.21%): correcting the
quantization variance yields better-calibrated uncertainty.

## Scope / deferred

Core scope is weight/grad/accumulator quantization with **fixed-point** VC. Deferred and
documented as known gaps (see the [original repo](https://github.com/ruqizhang/low-precision-sgld)):
- **Fully low-precision layers** — activation + backward/error quantization
  (`models/resnet_low.py`), which need `jax.custom_vjp`.
- **Block-FP VC** (`fp_Q_vc`) — the paper's CIFAR headline uses block-FP VC; here CIFAR VC
  uses fixed-point VC, which explains the ~1.5% gap to SGLDLP-F.

## Tests

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. python tests/test_quant.py      # quant primitives
JAX_PLATFORMS=cpu PYTHONPATH=. python tests/test_gaussian.py   # VC corrects naive bias
JAX_PLATFORMS=cpu PYTHONPATH=. python tests/test_cifar_smoke.py
PYTHONPATH=. python tests/crosscheck_qtorch.py                 # needs torch+qtorch (GPU)
```
