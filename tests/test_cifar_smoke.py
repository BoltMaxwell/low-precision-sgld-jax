"""End-to-end smoke test of LP-SGLD CIFAR train -> ensemble on tiny synthetic data.
Slow on CPU (~1-2 min: ResNet18 compiles per variant).

    JAX_PLATFORMS=cpu PYTHONPATH=. python tests/test_cifar_smoke.py
"""

import numpy as np

from lpsgld_jax.ensemble import evaluate, load_samples
from lpsgld_jax.train_cifar import train


def _fake(seed=0, n_train=64, n_test=32, num_classes=10):
    rng = np.random.default_rng(seed)
    return {
        "x_train": rng.standard_normal((n_train, 3, 32, 32)).astype(np.float32),
        "y_train": rng.integers(0, num_classes, n_train).astype(np.int32),
        "x_test": rng.standard_normal((n_test, 3, 32, 32)).astype(np.float32),
        "y_test": rng.integers(0, num_classes, n_test).astype(np.int32),
        "num_classes": num_classes,
    }


def test_all_variants_run():
    data = _fake()
    for variant in ("sgldlp_f", "naive", "vc"):
        samples = train(data, variant=variant, epochs=1, batch_size=32, lr_type="cyclic",
                        M=1, num_savemodel=1, seed=1, log_every=99)
        assert len(samples) == 1
        acc, ece, _ = evaluate(samples, data["x_test"], data["y_test"])
        assert 0.0 <= acc <= 1.0 and 0.0 <= ece <= 1.0
        print(f"[ok] {variant}: acc={acc:.3f} ece={ece:.3f}")


def test_serialise_roundtrip(tmp_path=None):
    import tempfile

    data = _fake(seed=2)
    d = tempfile.mkdtemp() if tmp_path is None else str(tmp_path)
    train(data, variant="vc", epochs=2, batch_size=32, lr_type="cyclic", M=1,
          num_savemodel=2, seed=2, save_dir=d, log_every=99)
    samples = load_samples(d, num_classes=10)
    assert len(samples) == 2
    acc, ece, _ = evaluate(samples, data["x_test"], data["y_test"])
    assert 0.0 <= acc <= 1.0
    print(f"[ok] serialise->load->evaluate: acc={acc:.3f}")


if __name__ == "__main__":
    test_all_variants_run()
    test_serialise_roundtrip()
    print("\nCIFAR smoke tests passed")
