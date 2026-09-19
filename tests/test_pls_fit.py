import unittest

import numpy as np

from scripts.pls_fit import pls1_gram


def dense_pls1(C, s, n_components):
    """The reference recursion: deflate the full covariance after every component."""
    C, s = np.array(C, float), np.array(s, float)
    weights, loadings, target_loadings = [], [], []
    for _ in range(n_components):
        w = s / np.linalg.norm(s)
        Cw = C @ w
        variance, covariance = float(w @ Cw), float(w @ s)
        loading = Cw / variance
        weights.append(w), loadings.append(loading), target_loadings.append(covariance / variance)
        C = C - np.outer(loading, Cw) - np.outer(Cw, loading) + variance * np.outer(loading, loading)
        s = s - loading * covariance
    return np.column_stack(weights), np.column_stack(loadings), np.array(target_loadings)


class Pls1GramTests(unittest.TestCase):
    def test_implicit_deflation_matches_dense_deflation(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 40)) @ rng.normal(size=(40, 40))
        y = X @ rng.normal(size=40) + rng.normal(size=300)
        X, y = X - X.mean(0), y - y.mean()
        C, s = X.T @ X / len(X), X.T @ y / len(X)
        for fast, dense in zip(pls1_gram(C, s, 12), dense_pls1(C, s, 12)):
            np.testing.assert_allclose(fast, dense, rtol=1e-8, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
