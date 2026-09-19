"""Centered, unscaled single-target PLS fitted from streamed weighted cross-products."""
import numpy as np

from .weighted_pls_statistics import WeightedStatistics


def pls1_gram(C, s, n_components):
    """Single-target PLS with regression deflation, from cross-products alone.

    Regression deflation takes C_(k+1) = C_k - v_k p_k p_k' (v_k the score variance, p_k the
    loading), so C_k = C - P D P' with D = diag(v). C_k is only ever needed times the new
    weight vector, so it is never formed: each component costs one product with C.
    """
    C = np.asarray(C, dtype=np.float64)
    s = np.array(s, dtype=np.float64, copy=True)
    features = C.shape[0]
    if C.shape != (features, features) or s.shape != (features,):
        raise ValueError('Invalid feature covariance or cross-covariance.')
    if not 1 <= n_components <= features:
        raise ValueError('Invalid component count.')
    if not (np.isfinite(C).all() and np.isfinite(s).all()):
        raise ValueError('PLS cross-products must be finite.')
    weights = np.zeros((features, n_components))
    loadings = np.zeros_like(weights)
    target_loadings = np.zeros(n_components)
    score_variances = np.zeros(n_components)
    for component in range(n_components):
        norm = np.linalg.norm(s)
        if not np.isfinite(norm) or norm <= 0:
            raise RuntimeError(f'Cross-covariance vanished before component {component + 1}.')
        w = s / norm
        previous = loadings[:, :component]
        Cw = C @ w - previous @ (score_variances[:component] * (previous.T @ w))   # C_k w
        score_variance = float(w @ Cw)
        if not np.isfinite(score_variance) or score_variance <= 0:
            raise RuntimeError(f'Component {component + 1} has nonpositive score variance.')
        covariance = float(w @ s)
        loading = Cw / score_variance
        weights[:, component] = w
        loadings[:, component] = loading
        target_loadings[component] = covariance / score_variance
        score_variances[component] = score_variance
        s = s - loading * covariance
    return weights, loadings, target_loadings


class WeightedPls:
    """PLS on activation batches; rows score as (batch - x_mean) @ base_rotations."""

    def __init__(self, n_components, target):
        self.n_components = n_components
        self.target = target

    def fit(self, batches_by_file, frames_by_file, feature_count):
        """Fit from {file: frame} with target and pls_fit_weight, reading batches per file."""
        statistics = WeightedStatistics(feature_count)
        for rel, frame in frames_by_file.items():
            if not frame['horizon_type'].eq('horizon_free').all():
                raise ValueError('Only horizon-free rows may enter the PLS fit.')
            target = frame[self.target].to_numpy(dtype=np.float64)
            fit_weight = frame['pls_fit_weight'].to_numpy(dtype=np.float64)
            if not np.isfinite(target).all() or not np.isfinite(fit_weight).all() or (fit_weight <= 0).any():
                raise ValueError(f'Invalid PLS target or weight: {rel}')
            for start, stop, block in batches_by_file(rel):
                statistics.add(block, target[start:stop], fit_weight[start:stop])
        self.rows, self.mass = statistics.rows, statistics.mass

        C, cross_covariance, y_variance, self.x_mean, self.y_mean = statistics.centred()
        del statistics
        if not (np.isfinite(C).all() and np.isfinite(cross_covariance).all()) or y_variance <= 0:
            raise ValueError('Invalid training covariance or nonpositive target variance.')
        self.y_variance = y_variance

        self.weights, self.loadings, self.target_loadings = pls1_gram(
            C, cross_covariance, self.n_components)
        loading_weights = self.loadings.T @ self.weights
        if (not np.isfinite(loading_weights).all()
                or np.linalg.matrix_rank(loading_weights) != self.n_components):
            raise RuntimeError('The PLS rotation system is numerically singular.')
        self.base_rotations = self.weights @ np.linalg.pinv(loading_weights)
        self.score_covariance = self.base_rotations.T @ C @ self.base_rotations
        self.component_variances = np.diag(self.score_covariance)
        if not np.isfinite(self.base_rotations).all() or (self.component_variances <= 0).any():
            raise RuntimeError('The fit did not produce finite, positive-variance components.')

        # Share of total feature variance reconstructed by each component; PLS scores are
        # mutually uncorrelated, so shares add across components.
        self.total_x_variance = float(np.trace(C))
        self.x_variance_share = self.component_variances * (self.loadings ** 2).sum(axis=0) / self.total_x_variance

        # Cumulative in-sample R-squared from the first k components.
        coefficients = self.base_rotations * self.target_loadings
        self.cumulative_r2 = []
        for k in range(1, self.n_components + 1):
            beta = coefficients[:, :k].sum(axis=1)
            residual = y_variance - 2 * beta @ cross_covariance + beta @ C @ beta
            self.cumulative_r2.append(1 - residual / y_variance)
        return self
