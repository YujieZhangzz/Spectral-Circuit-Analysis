"""Subspace overlap metrics for cross-task analysis.

Main metric, for U_a (d, p) and U_b (d, q) with orthonormal columns:

    overlap(U_a, U_b) = ||U_a^T U_b||_F^2 / min(p, q)

i.e. the mean squared cosine of the principal angles; symmetric, in [0, 1],
and 1 iff one subspace contains the other. sigma_1(U_a^T U_b) (cosine of the
smallest principal angle) is provided as a supplementary measure.

For random subspaces the expected overlap is about p*q / (d * min(p, q)),
which is tiny for d ~ 1e4-1e5. Monte Carlo baselines are provided for small d.
"""

from __future__ import annotations
import numpy as np


# Core overlap metrics

def subspace_overlap(U_a: np.ndarray, U_b: np.ndarray) -> float:
    """
    Normalized projection overlap between two subspaces.

    Args:
        U_a: (d, p) ndarray with orthonormal columns
        U_b: (d, q) ndarray with orthonormal columns

    Returns:
        overlap in [0, 1]. Returns 0.0 if either subspace is empty.
    """
    if U_a.size == 0 or U_b.size == 0:
        return 0.0
    p = U_a.shape[1]
    q = U_b.shape[1]
    if p == 0 or q == 0:
        return 0.0
    M = U_a.T @ U_b                          # (p, q)
    return float((M ** 2).sum() / min(p, q))


def top_principal_angle(U_a: np.ndarray, U_b: np.ndarray) -> float:
    """
    Cosine of smallest principal angle (largest singular value of U_a^T U_b).

    Returns a value in [0, 1]. 1 means the two subspaces share at least
    one direction exactly.
    """
    if U_a.size == 0 or U_b.size == 0:
        return 0.0
    if U_a.shape[1] == 0 or U_b.shape[1] == 0:
        return 0.0
    M = U_a.T @ U_b
    s = np.linalg.svd(M, compute_uv=False)
    return float(s[0])


def all_principal_angles(U_a: np.ndarray, U_b: np.ndarray) -> np.ndarray:
    """
    All principal angle cosines (sorted descending). Length = min(p, q).

    Useful for diagnostics: a single high cosine + many low cosines means
    the subspaces share one strong direction; many medium cosines means
    they share a higher-dimensional structure.
    """
    if U_a.size == 0 or U_b.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if U_a.shape[1] == 0 or U_b.shape[1] == 0:
        return np.zeros((0,), dtype=np.float32)
    M = U_a.T @ U_b
    s = np.linalg.svd(M, compute_uv=False)
    return s.astype(np.float32)


# Random baseline (Monte Carlo)

def random_orthonormal(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a uniformly random k-dimensional subspace of R^d via QR."""
    A = rng.standard_normal((d, k)).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    return Q


def random_baseline_overlap(
    d: int,
    p: int,
    q: int,
    n_samples: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """
    Monte Carlo distribution of overlap between random (d, p) and (d, q)
    orthonormal subspaces.

    Returns:
        (n_samples,) ndarray of overlap values. Use percentiles for
        significance testing.

    Performance:
        For d ~ 10^5 and n_samples=200 this takes a few seconds per
        (p, q) pair; use BaselineCache to share results across units.
    """
    if p == 0 or q == 0:
        return np.zeros((n_samples,), dtype=np.float32)
    rng = np.random.default_rng(seed)
    out = np.empty(n_samples, dtype=np.float32)
    for i in range(n_samples):
        Ua = random_orthonormal(d, p, rng)
        Ub = random_orthonormal(d, q, rng)
        out[i] = subspace_overlap(Ua, Ub)
    return out


def baseline_threshold(
    d: int,
    p: int,
    q: int,
    n_samples: int = 200,
    quantile: float = 0.99,
    seed: int = 0,
) -> float:
    """
    Single-number significance threshold: the (quantile)th percentile of
    the random baseline distribution. An observed overlap above this
    value is significant at the (1 - quantile) level.
    """
    samples = random_baseline_overlap(d, p, q, n_samples=n_samples, seed=seed)
    return float(np.quantile(samples, quantile))


# Baseline cache - many units share the same (d, p, q) shape

class BaselineCache:
    """
    Cache random baseline thresholds keyed by (d, p, q). The Monte Carlo
    distribution depends only on shape, not on the specific subspace, so
    units with identical shape can share the same threshold.

    A run has thousands of overlap computations but only a few distinct
    (d, p, q) shapes.

    Fast mode: for d >= fast_threshold the random overlap concentrates
    tightly around its mean p*q / (d * min(p, q)), so instead of Monte
    Carlo we use

        threshold = safety_factor * p * q / (d * min(p, q))

    With safety_factor = 5 this is well above the 99th percentile for
    large d, and gives the same significance decisions in practice.
    """

    def __init__(self, n_samples: int = 200, quantile: float = 0.99,
                 seed: int = 0, verbose: bool = False,
                 fast_threshold: int = 10_000, safety_factor: float = 5.0):
        self.n_samples = n_samples
        self.quantile = quantile
        self.seed = seed
        self.verbose = verbose
        self.fast_threshold = fast_threshold
        self.safety_factor = safety_factor
        self._cache: dict = {}

    def get(self, d: int, p: int, q: int) -> float:
        # Symmetric in (p, q): canonicalize the cache key
        key = (int(d), int(min(p, q)), int(max(p, q)))
        if key not in self._cache:
            if key[0] >= self.fast_threshold:
                # Analytical shortcut for large d
                if key[1] == 0 or key[2] == 0:
                    self._cache[key] = 0.0
                else:
                    mean = key[1] * key[2] / (key[0] * key[1])
                    self._cache[key] = self.safety_factor * mean
                if self.verbose:
                    print(f"  baseline (fast): d={key[0]} "
                          f"p={key[1]} q={key[2]} -> {self._cache[key]:.2e}")
            else:
                if self.verbose:
                    print(f"  baseline (MC): d={key[0]} "
                          f"p={key[1]} q={key[2]} (computing)")
                self._cache[key] = baseline_threshold(
                    d=key[0], p=key[1], q=key[2],
                    n_samples=self.n_samples,
                    quantile=self.quantile,
                    seed=self.seed,
                )
        return self._cache[key]

    def __len__(self):
        return len(self._cache)


# Self-test on synthetic data

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d = 1000

    # Test 1: identical subspaces should have overlap = 1
    A = rng.standard_normal((d, 5))
    Q, _ = np.linalg.qr(A)
    print(f"identical:    overlap = {subspace_overlap(Q, Q):.4f}  (expect 1.0)")

    # Test 2: orthogonal subspaces should have overlap ~ 0
    A1 = rng.standard_normal((d, 5))
    A2 = rng.standard_normal((d, 5))
    Q1, _ = np.linalg.qr(A1)
    A2 = A2 - Q1 @ (Q1.T @ A2)               # project out Q1
    Q2, _ = np.linalg.qr(A2)
    print(f"orthogonal:   overlap = {subspace_overlap(Q1, Q2):.4f}  (expect ~0)")

    # Test 3: shared subspace of 2 dimensions out of 5
    A_shared = rng.standard_normal((d, 2))
    Q_shared, _ = np.linalg.qr(A_shared)
    A1 = np.hstack([Q_shared, rng.standard_normal((d, 3))])
    A2 = np.hstack([Q_shared, rng.standard_normal((d, 3))])
    Q1, _ = np.linalg.qr(A1)
    Q2, _ = np.linalg.qr(A2)
    overlap = subspace_overlap(Q1, Q2)
    print(f"2/5 shared:   overlap = {overlap:.4f}  (expect ~0.4 = 2/5)")

    # Test 4: random baseline distribution
    print()
    print("Random baseline (d=1000, p=q=5, 200 samples):")
    samples = random_baseline_overlap(d=1000, p=5, q=5, n_samples=200, seed=0)
    print(f"  mean   = {samples.mean():.4f}")
    print(f"  median = {np.median(samples):.4f}")
    print(f"  99%    = {np.quantile(samples, 0.99):.4f}")
    print(f"  max    = {samples.max():.4f}")
    print(f"  asymptotic ~ p*q / (d * min(p,q)) = {5*5 / (1000*5):.4f}")

    # Test 5: BaselineCache reuse
    print()
    cache = BaselineCache(n_samples=100, quantile=0.99, verbose=True)
    t1 = cache.get(d=1000, p=5, q=5)
    t2 = cache.get(d=1000, p=5, q=5)            # cached
    t3 = cache.get(d=1000, p=5, q=8)            # new
    t4 = cache.get(d=1000, p=8, q=5)            # symmetric, same as t3
    print(f"  thresholds: {t1:.4f} {t2:.4f} {t3:.4f} {t4:.4f}")
    print(f"  cache size: {len(cache)} (expect 2)")
    assert t1 == t2, "cache hit failed"
    assert t3 == t4, "symmetry failed"
    print("OK")