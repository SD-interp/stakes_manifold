"""Between-class PCA of merged stakes classes and arc length along its centroid spline.

BCPC rows carry equal total weight per source file. Median class centers, the
BCPC1-3 anchor selection, and the full BCPC1-7 penalized cubic spline follow the
original seven-component target; `bcpc_arc_length` starts at the end nearest very_low.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.polynomial import polynomial as poly
from scipy.integrate import quad
from scipy.interpolate import BSpline, splev
from scipy.linalg import eigh


@dataclass(frozen=True)
class BcpcSettings:
    n_components: int = 7
    centroid_method: str = 'median'
    anchor_weight: float = 1.0
    internal_knots: int = 2
    end_curvature_penalty: float = 0.0002
    degree: int = 3
    curve_points: int = 500

    def __post_init__(self):
        if self.centroid_method not in ('mean', 'median'):
            raise ValueError('centroid_method must be "mean" or "median".')
        if not np.isfinite(self.anchor_weight) or self.anchor_weight <= 0:
            raise ValueError('anchor_weight must be finite and positive.')
        if not np.isfinite(self.end_curvature_penalty) or self.end_curvature_penalty < 0:
            raise ValueError('end_curvature_penalty must be finite and nonnegative.')
        if self.degree != 3:
            raise ValueError('The BCPC spline is cubic (degree 3).')
        if not isinstance(self.curve_points, int) or self.curve_points < 2:
            raise ValueError('curve_points must be an integer of at least two.')


def weighted_center(values, row_weights=None, method='median'):
    if method == 'mean':
        return np.average(values, axis=0, weights=row_weights)
    if row_weights is None:
        return np.median(values, axis=0)

    # Weighted coordinate-wise median, preserving the equal-file row weights.
    center = np.empty(values.shape[1])
    for column in range(values.shape[1]):
        order = np.argsort(values[:, column], kind='stable')
        sorted_values = values[order, column]
        cumulative = np.cumsum(row_weights[order])
        half_mass = cumulative[-1] / 2
        index = int(np.searchsorted(cumulative, half_mass, side='left'))
        # Recognize half-mass ties despite floating-point accumulation error.
        if index > 0 and np.isclose(cumulative[index - 1], half_mass, rtol=1e-12, atol=0):
            index -= 1
        if index + 1 < len(values) and np.isclose(cumulative[index], half_mass, rtol=1e-12, atol=0):
            center[column] = (sorted_values[index] + sorted_values[index + 1]) / 2
        else:
            center[column] = sorted_values[index]
    return center


def fit_between_class_pca(X, weights, labels, n_components):
    """Weighted between-class scatter eigenvectors; returns (projection, class_index, scores)."""
    classes, class_index = np.unique(labels, return_inverse=True)
    if len(classes) < 2:
        raise ValueError('At least two merged stakes classes are required.')

    class_mass = np.bincount(class_index, weights=weights)
    mean = np.average(X, axis=0, weights=weights)
    class_means = np.stack([
        np.average(X[class_index == c], axis=0, weights=weights[class_index == c])
        for c in range(len(classes))
    ])
    centered_means = class_means - mean
    between_scatter = centered_means.T @ (class_mass[:, None] * centered_means)
    between_scatter = (between_scatter + between_scatter.T) / 2

    # Ordinary symmetric eigensystem: the metric is Euclidean, not within-class scatter.
    max_components = min(n_components, len(classes) - 1, X.shape[1])
    if max_components < n_components:
        raise ValueError(f'The merged classes support at most {max_components} BCPC directions; '
                         f'{n_components} need at least {n_components + 1} classes.')
    eigenvalues, components = eigh(
        between_scatter,
        subset_by_index=(X.shape[1] - max_components, X.shape[1] - 1),
    )
    eigenvalues, components = eigenvalues[::-1], components[:, ::-1]
    tolerance = np.finfo(np.float64).eps * X.shape[1] * max(float(eigenvalues[0]), 0.0)
    keep = eigenvalues > tolerance
    eigenvalues, components = eigenvalues[keep], components[:, keep]
    if components.shape[1] != n_components:
        raise ValueError(f'This analysis requires {n_components} positive BCPC directions.')

    # Fix arbitrary signs for a consistent orientation.
    pivots = np.argmax(np.abs(components), axis=0)
    components *= np.sign(components[pivots, np.arange(components.shape[1])])

    scores = (X - mean) @ components
    projection = dict(mean=mean, components=components, eigenvalues=eigenvalues,
                      classes=classes, class_mass=class_mass, class_means=class_means)
    return projection, class_index, scores


class BcpcArcLength:
    """Fit BCPC scores and the oriented closest-point arc length of their centroid spline."""

    def __init__(self, settings=None):
        self.settings = settings or BcpcSettings()

    def fit(self, X, weights, rows, stakes_column='stakes'):
        """Fit on aligned rows and return them with BCPC scores and arc-length columns."""
        projection, class_index, scores = fit_between_class_pca(
            X, weights, rows[stakes_column].to_numpy(), self.settings.n_components)
        self.fit_curve(projection, scores, weights, class_index)
        projected_rows = pd.concat([rows, pd.DataFrame(scores, columns=self.score_columns)], axis=1)
        return self.add_arc_length_columns(projected_rows, scores)

    def fit_curve(self, projection, scores, weights, class_index):
        """Fit centers, spline and orientation to already computed BCPC scores.

        `projection` is a `fit_between_class_pca` dictionary whose `classes` are indexed by
        `class_index`; `scores` are the fitting rows' coordinates in its components.
        """
        self.projection = projection
        self.score_columns = [f'BCPC{i + 1}' for i in range(scores.shape[1])]
        self._fit_centers(scores, weights, class_index)
        self._fit_spline()
        self._orient()
        self.spline_curve['bcpc_arc_length'] = [self.arc_length_at(u)
                                                for u in self.spline_curve['parameter']]
        return self

    def class_table(self):
        return pd.DataFrame({'class': self.projection['classes'],
                             'weight': self.projection['class_mass']})

    def variance_table(self):
        return pd.DataFrame({'component': self.score_columns,
                             'between_class_variance': self.projection['eigenvalues']})

    def transform(self, block):
        return (block - self.projection['mean']) @ self.projection['components']

    def _fit_centers(self, scores, weights, class_index):
        if scores.shape[1] < 3:
            raise ValueError('This analysis requires at least three projection components.')
        method = self.settings.centroid_method
        classes = self.projection['classes']
        points_3d = scores[:, :3]
        self.centroids = pd.DataFrame(
            np.stack([
                weighted_center(scores[class_index == c], weights[class_index == c], method)
                for c in range(len(classes))
            ]),
            index=pd.Index(classes, name='stakes'),
            columns=self.score_columns,
        )
        radius_squared = np.sum(points_3d ** 2, axis=1)
        self.anchor_indices = {}
        for name, mask in (
            ('negative_anchor', points_3d[:, 0] < 0),
            ('positive_anchor', points_3d[:, 0] > 0),
        ):
            candidates = np.flatnonzero(mask)
            if len(candidates) < 10:
                raise ValueError(f'{name} needs at least ten points; found {len(candidates)}.')
            order = np.argsort(-radius_squared[candidates], kind='stable')
            self.anchor_indices[name] = candidates[order[:10]]
        self.anchors = pd.DataFrame(
            {name: weighted_center(scores[indices], method=method)
             for name, indices in self.anchor_indices.items()},
            index=self.score_columns,
        ).T
        self.anchors.index.name = 'anchor'

    def _fit_spline(self):
        s = self.settings
        score_columns = self.score_columns
        spline_points = pd.concat([
            self.anchors.loc[['negative_anchor']],
            self.centroids.sort_values(score_columns[0], kind='stable'),
            self.anchors.loc[['positive_anchor']],
        ])
        control_scores = spline_points[score_columns].to_numpy()
        fit_weights = np.ones(len(spline_points))
        fit_weights[[0, -1]] = s.anchor_weight

        max_internal_knots = len(control_scores) - s.degree - 1
        if (not isinstance(s.internal_knots, int) or isinstance(s.internal_knots, bool)
                or not 0 <= s.internal_knots <= max_internal_knots):
            raise ValueError(f'internal_knots must be an integer from 0 to {max_internal_knots}.')

        chord_lengths = np.linalg.norm(np.diff(control_scores, axis=0), axis=1)
        if np.any(chord_lengths == 0):
            raise ValueError('Consecutive spline points coincide; chord-length parameters must be distinct.')
        parameter = np.r_[0.0, np.cumsum(chord_lengths)]
        parameter /= parameter[-1]

        internal_knots = np.quantile(
            parameter, np.linspace(0.0, 1.0, s.internal_knots + 2)[1:-1]
        )
        knot_vector = np.r_[
            np.repeat(parameter[0], s.degree + 1),
            internal_knots,
            np.repeat(parameter[-1], s.degree + 1),
        ]

        # Identity coefficients expose every B-spline basis function as one column.
        n_basis = len(knot_vector) - s.degree - 1
        basis = BSpline(knot_vector, np.eye(n_basis), s.degree)
        sqrt_weights = np.sqrt(fit_weights)[:, None]
        design = sqrt_weights * basis(parameter)
        target = sqrt_weights * control_scores

        if s.end_curvature_penalty > 0:
            second_derivative = basis.derivative(2)
            penalty_blocks = []
            for left, right in (
                (parameter[0], parameter[1]),
                (parameter[-2], parameter[-1]),
            ):
                # Split at every internal knot so the second derivative is linear
                # within each integration span. Its squared norm is quadratic.
                breaks = np.r_[left, internal_knots[
                    (internal_knots > left) & (internal_knots < right)
                ], right]
                for a, b in zip(breaks[:-1], breaks[1:]):
                    half_width = (b - a) / 2
                    nodes = (a + b) / 2 + half_width * np.array([-1.0, 1.0]) / np.sqrt(3)
                    penalty_blocks.append(
                        np.sqrt(s.end_curvature_penalty * half_width) * second_derivative(nodes)
                    )
            penalty_design = np.vstack(penalty_blocks)
            design = np.vstack([design, penalty_design])
            target = np.vstack([target, np.zeros((len(penalty_design), len(score_columns)))])

        coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
        if rank < n_basis:
            raise ValueError('Spline fit is rank deficient; reduce internal_knots.')
        spline_tck = (knot_vector, coefficients.T, s.degree)
        curve_parameter = np.linspace(0.0, 1.0, s.curve_points)
        self.spline_curve = pd.DataFrame(
            np.asarray(splev(curve_parameter, spline_tck)).T,
            columns=score_columns,
        )
        self.spline_curve.insert(0, 'parameter', curve_parameter)

        fitted_scores = np.asarray(splev(parameter, spline_tck)).T
        spline_points['parameter'] = parameter
        spline_points['fit_weight'] = fit_weights
        spline_points['residual_distance'] = np.linalg.norm(control_scores - fitted_scores, axis=1)
        self.weighted_residual_sum = float(
            np.sum(fit_weights * spline_points['residual_distance'].to_numpy() ** 2))
        self.spline_points = spline_points
        self.curve = BSpline(spline_tck[0], np.asarray(spline_tck[1]).T, spline_tck[2])

    def _orient(self):
        if 'very_low' not in self.centroids.index:
            raise ValueError('The very_low class is required to orient arc length.')
        score_columns = self.score_columns
        anchor_names = ['negative_anchor', 'positive_anchor']
        anchor_distances = np.linalg.norm(
            self.anchors.loc[anchor_names, score_columns].to_numpy()
            - self.centroids.loc['very_low', score_columns].to_numpy(), axis=1,
        )
        if np.isclose(anchor_distances[0], anchor_distances[1], rtol=1e-12, atol=1e-12):
            raise ValueError('The anchors are equally close to very_low; the zero end is ambiguous.')
        self.zero_anchor = anchor_names[int(np.argmin(anchor_distances))]
        self.reverse_arc_length = self.zero_anchor == 'positive_anchor'

        curve = self.curve
        self._curve_derivative = curve.derivative()
        self._span_edges = np.unique(curve.t[curve.k: -curve.k])
        span_lengths = np.array([
            quad(self._speed, left, right, epsabs=1e-8, epsrel=1e-8)[0]
            for left, right in zip(self._span_edges[:-1], self._span_edges[1:])
        ])
        self._span_count = len(span_lengths)
        self._cumulative_lengths = np.r_[0.0, np.cumsum(span_lengths)]
        self.total_arc_length = float(self._cumulative_lengths[-1])

        # Cubic coefficients in local t=(u-left)/(right-left), keeping root finding
        # on [0, 1] even when knot spans have very different lengths.
        self._span_polynomials = []
        for left, right in zip(self._span_edges[:-1], self._span_edges[1:]):
            width = right - left
            local_coefficients = np.stack([
                curve(left),
                curve.derivative(1)(left) * width,
                curve.derivative(2)(left) * width ** 2 / 2,
                curve.derivative(3)(left) * width ** 3 / 6,
            ])
            self._span_polynomials.append((left, width, local_coefficients))

    def _speed(self, u):
        return float(np.linalg.norm(self._curve_derivative(u)))

    def arc_length_at(self, u):
        edges = self._span_edges
        u = float(np.clip(u, edges[0], edges[-1]))
        span = min(np.searchsorted(edges, u, side='right') - 1, self._span_count - 1)
        length = self._cumulative_lengths[span] + quad(
            self._speed, edges[span], u, epsabs=1e-8, epsrel=1e-8,
        )[0]
        oriented = self.total_arc_length - length if self.reverse_arc_length else length
        return float(np.clip(oriented, 0.0, self.total_arc_length))

    def closest_parameter(self, point):
        candidates = list(self._span_edges)
        for left, width, local_coefficients in self._span_polynomials:
            # (r(t)-point) dot r'(t) is half the derivative of squared distance.
            stationary_polynomial = np.zeros(6)
            for axis in range(len(self.score_columns)):
                residual = local_coefficients[:, axis].copy()
                residual[0] -= point[axis]
                term = poly.polymul(residual, poly.polyder(local_coefficients[:, axis]))
                stationary_polynomial[:len(term)] += term
            for root in poly.polyroots(stationary_polynomial):
                if abs(root.imag) <= 1e-8 and -1e-10 <= root.real <= 1 + 1e-10:
                    candidates.append(left + width * np.clip(root.real, 0.0, 1.0))
        candidates = np.unique(candidates)
        if self.reverse_arc_length:
            candidates = candidates[::-1]  # Exact distance ties prefer smaller arc length.
        squared_distances = np.sum((self.curve(candidates) - point) ** 2, axis=1)
        return float(candidates[np.argmin(squared_distances)])

    def batch_arc_length(self, points, grid_points=4097, newton_steps=8, chunk_rows=2048):
        """Oriented closest-point arc length of many points at once.

        Agrees with `add_arc_length_columns` to floating-point accuracy: each point's
        nearest node on a dense parameter grid seeds Newton iterations on
        (C(u) - x) . C'(u) = 0, kept within one grid spacing of that node. Arc length is
        a cumulative Gauss-Legendre table over the grid (knots included, so the speed is
        smooth on every interval) plus the same quadrature from the node to u.
        """
        curve, first, second = self.curve, self._curve_derivative, self._curve_derivative.derivative()
        low, high = self._span_edges[0], self._span_edges[-1]
        grid = np.union1d(np.linspace(low, high, grid_points), self._span_edges)
        nodes, gauss_weights = np.polynomial.legendre.leggauss(8)

        def length_from(start, stop):
            """Arc length from start to stop, elementwise, by 8-point Gauss-Legendre."""
            half = (stop - start) / 2
            u = (start + stop)[:, None] / 2 + half[:, None] * nodes[None, :]
            speed = np.linalg.norm(first(u.ravel()), axis=1).reshape(u.shape)
            return half * (speed @ gauss_weights)

        cumulative = np.r_[0.0, np.cumsum(length_from(grid[:-1], grid[1:]))]
        grid_points_xyz = curve(grid)
        grid_norms = np.sum(grid_points_xyz ** 2, axis=1)

        points = np.asarray(points, dtype=np.float64)
        lengths = np.empty(len(points))
        for start in range(0, len(points), chunk_rows):
            x = points[start:start + chunk_rows]
            nearest = np.argmin(grid_norms[None, :] - 2 * x @ grid_points_xyz.T, axis=1)
            lower = grid[np.maximum(nearest - 1, 0)]
            upper = grid[np.minimum(nearest + 1, len(grid) - 1)]
            u = grid[nearest]
            for _ in range(newton_steps):
                residual = curve(u) - x
                d1, d2 = first(u), second(u)
                gradient = np.sum(residual * d1, axis=1)
                curvature = np.sum(d1 * d1, axis=1) + np.sum(residual * d2, axis=1)
                step = np.where(curvature > 0, gradient / np.where(curvature > 0, curvature, 1.0), 0.0)
                u = np.clip(u - step, lower, upper)
            # Keep the grid node if refinement did not improve on it.
            refined_distance = np.sum((curve(u) - x) ** 2, axis=1)
            node_distance = np.sum((grid_points_xyz[nearest] - x) ** 2, axis=1)
            u = np.where(refined_distance <= node_distance, u, grid[nearest])

            interval = np.clip(np.searchsorted(grid, u, side='right') - 1, 0, len(grid) - 2)
            length = cumulative[interval] + length_from(grid[interval], u)
            if self.reverse_arc_length:
                length = self.total_arc_length - length
            lengths[start:start + len(x)] = np.clip(length, 0.0, self.total_arc_length)
        return lengths

    def add_arc_length_columns(self, frame, point_coordinates):
        """Apply the already fitted spline to aligned rows; does not modify the fit."""
        if len(frame) != len(point_coordinates):
            raise ValueError('Row metadata and projection coordinates are misaligned.')
        closest_parameters = np.array([
            self.closest_parameter(point) for point in point_coordinates
        ])
        closest_points = self.curve(closest_parameters)
        frame['spline_parameter'] = closest_parameters
        frame['bcpc_arc_length'] = [self.arc_length_at(u) for u in closest_parameters]
        frame['distance_to_spline'] = np.linalg.norm(point_coordinates - closest_points, axis=1)
        for axis, name in enumerate(self.score_columns):
            frame[f'closest_{name}'] = closest_points[:, axis]
        return frame
