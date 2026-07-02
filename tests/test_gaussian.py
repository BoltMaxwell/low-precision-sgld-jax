"""Regression test for the Gaussian toy: VC corrects the naive variance bias.

    PYTHONPATH=. python tests/test_gaussian.py
"""

import jax
import numpy as np

from lpsgld_jax.gaussian import run


def _std(variant, steps=400_000, seed=0):
    traj = np.asarray(run(variant, jax.random.key(seed), steps))
    return traj[len(traj) // 10:].std()


def test_vc_matches_naive_is_biased():
    std_f = _std("f")
    std_naive = _std("naive")
    std_vc = _std("vc")
    print(f"std  f={std_f:.3f}  naive={std_naive:.3f}  vc={std_vc:.3f}  (target 1.0)")
    # naive low-precision SGLD inflates the stationary variance
    assert std_naive > 1.12, std_naive
    # full-precision-accumulator and variance-corrected both track N(0,1)
    assert 0.9 < std_f < 1.1, std_f
    assert 0.9 < std_vc < 1.1, std_vc
    # VC is meaningfully closer to 1 than naive
    assert abs(std_vc - 1.0) < abs(std_naive - 1.0) - 0.1
    print("[ok] VC corrects the naive variance bias")


if __name__ == "__main__":
    test_vc_matches_naive_is_biased()
    print("\ngaussian toy test passed")
