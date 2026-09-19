import json
from dataclasses import asdict
import unittest

import numpy as np

from scripts import geometric_surface as gs

CONFIG = gs.SurfaceConfig(shape_knots=(3, 3), u_knots=(3, 3), shape_bending=1e-6, u_bending=1e-8,
                          fit_iterations=80, search_grid=33, trajectory_steps=48)


def fit(points, target, u_zero, config=CONFIG):
    weights = np.full(len(points), 1.0 / len(points))
    shape, st, _, diagnostics = gs.fit_shape(points, weights, gs.initial_chart(points, target, weights),
                                             config)
    u_map = gs.fit_scalar(st, target, weights, config)
    coordinates = gs.OrthogonalCoordinates.build(shape, u_map, u_zero, points.mean(0), st, weights, config)
    return coordinates, diagnostics


class PlaneTests(unittest.TestCase):
    """A flat sheet where u, v and both lengths are known exactly."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        x, y = rng.uniform(0, 2, 900), rng.uniform(0, 3, 900)
        cls.x, cls.y = x, y
        cls.points = np.column_stack([x, y, 0.01 * rng.normal(size=900)])
        cls.coordinates, cls.diagnostics = fit(cls.points, 5 * x, u_zero=2.5)
        cls.mapped = cls.coordinates.map_points(cls.points)

    def test_shape_is_the_plane(self):
        self.assertLess(self.diagnostics['weighted_rms_distance'], 0.012)
        self.assertLess(self.diagnostics['residual_tangent_cosine_max'], 1e-4)

    def test_u_matches_target(self):
        np.testing.assert_allclose(self.mapped['surface_u'], 5 * self.x, atol=0.02)

    def test_parallel_length_runs_along_x_from_the_zero_level(self):
        inside = self.mapped['coordinate_status'] == 'interior'
        self.assertGreater(inside.mean(), 0.9)
        np.testing.assert_allclose(self.mapped['arc_length_parallel'][inside], self.x[inside] - 0.5, atol=0.01)

    def test_v_is_length_along_y(self):
        v = self.mapped['surface_v']
        sign = np.sign(np.corrcoef(v, self.y)[0, 1])
        centre = self.y - self.points[:, 1].mean()
        np.testing.assert_allclose(sign * v, centre, atol=0.02)
        np.testing.assert_array_equal(self.mapped['arc_length_orthogonal'], v)


class CurvedSheetTests(unittest.TestCase):
    """A curved sheet in five dimensions."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(1)
        s, t = rng.uniform(0, 1, 800), rng.uniform(0, 1, 800)
        clean = np.column_stack([s, t, 0.8 * s ** 2, 0.5 * np.sin(2 * t), 0.3 * s * t])
        cls.points = clean + 0.01 * rng.normal(size=clean.shape)
        cls.target = 3 * s + 0.5 * t ** 2
        cls.coordinates, cls.diagnostics = fit(cls.points, cls.target, u_zero=1.5)

    def test_closest_points_are_orthogonal_projections(self):
        self.assertLess(self.diagnostics['weighted_rms_distance'], 0.03)
        self.assertLess(self.diagnostics['residual_tangent_cosine_max'], 1e-5)

    def test_chart_table_matches_direct_integration(self):
        st = gs.closest_points(self.coordinates.shape, self.points, config=CONFIG)[0]
        table = self.coordinates.coordinates(st)
        direct = self.coordinates.integrate_coordinates(st, steps=256)
        for column in ('surface_u', 'surface_v', 'arc_length_parallel'):
            np.testing.assert_allclose(table[column], direct[column], atol=1e-4)

    def test_net_and_chart_table_agree(self):
        check = self.coordinates.table_diagnostics()
        self.assertGreater(check['net_nodes_checked'], 1000)
        for key in ('round_trip_max_abs_u', 'round_trip_max_abs_v', 'round_trip_max_abs_parallel'):
            self.assertLess(check[key], 1e-4, key)
        self.assertEqual(check['chart_table_invalid_share'], 0.0)
        self.assertGreater(check['chart_min_u_gradient'], check['u_gradient_floor'])

    def test_u_and_v_directions_are_orthogonal(self):
        self.assertLess(self.coordinates.orthogonality_check(), 1e-9)

    def test_u_approximates_the_target(self):
        mapped = self.coordinates.map_points(self.points)
        # The target is exact but the points are noisy: 0.01 of noise along s moves 3 s by 0.03.
        self.assertLess(np.sqrt(np.mean((mapped['surface_u'] - self.target) ** 2)), 0.035)
        side = np.sign(mapped['arc_length_parallel'])
        np.testing.assert_array_equal(side, np.sign(mapped['surface_u'] - 1.5))

    def test_saved_arrays_reproduce_the_mapping(self):
        arrays = {key: np.asarray(value) for key, value in self.coordinates.arrays().items()}
        config = json.loads(json.dumps(asdict(self.coordinates.config)))
        restored = gs.OrthogonalCoordinates.from_arrays(arrays, config)
        before, after = self.coordinates.map_points(self.points), restored.map_points(self.points)
        for column in gs.MAP_COLUMNS:
            np.testing.assert_array_equal(before[column], after[column])


class CriticalLineTests(unittest.TestCase):
    """A plane whose target (x - 1.2)^2 has no gradient on x = 1.2: u is no coordinate there,
    and beyond it the level set u = u_zero has a second branch that does not set v."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(3)
        cls.x, y = rng.uniform(0, 2, 900), rng.uniform(0, 3, 900)
        points = np.column_stack([cls.x, y, 0.01 * rng.normal(size=900)])
        cls.coordinates, _ = fit(points, (cls.x - 1.2) ** 2, u_zero=0.09)   # branches at x = 0.9, 1.5
        cls.mapped = cls.coordinates.map_points(points)

    def test_rows_past_the_critical_line_are_invalid(self):
        status = self.mapped['coordinate_status']
        self.assertTrue(np.all(status[self.x > 1.25] == 'invalid_u'))
        self.assertTrue(np.all(status[np.abs(self.x - 1.2) < 0.02] == 'invalid_u'))
        self.assertGreater(np.mean(status[(self.x > 0.05) & (self.x < 1.0)] != 'invalid_u'), 0.95)


class ConvexShapeTests(unittest.TestCase):
    """A noisy cap in four dimensions: convex along one normal, a saddle along the other."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(4)
        x, y = rng.uniform(-1, 1, 1500), rng.uniform(-1, 1, 1500)
        clean = np.column_stack([2 * x, 2 * y, 0.5 * (x ** 2 + y ** 2), 0.3 * (x ** 2 - y ** 2)])
        points = clean + 0.01 * rng.normal(size=clean.shape)
        weights = np.full(len(points), 1 / len(points))
        cls.shape, cls.st, _, cls.diagnostics = gs.fit_shape(points, weights, None, CONFIG)

    def test_fits_the_cap_to_noise_level(self):
        self.assertLess(self.diagnostics['weighted_rms_distance'], 0.03)
        self.assertTrue(self.diagnostics['converged'])

    def test_height_along_the_convex_direction_is_convex(self):
        self.assertGreater(self.diagnostics['convex_height_min_eigen_ratio'], -0.06)
        # The convex direction is the cap's normal (dimension 3), not the saddle's.
        self.assertGreater(abs(self.shape.frame[2, 2]), 0.95)

    def test_hull_covers_the_closest_points(self):
        self.assertTrue(self.shape.inside_hull(self.st).all())
        self.assertFalse(self.shape.inside_hull(np.array([[-0.5, -0.5], [1.5, 0.5]])).any())

    def test_convex_solve_holds_a_concave_target_convex(self):
        knots = gs.open_knots(3, 3)
        shell = gs.TensorSpline(knots, knots, 3, np.zeros((7, 7, 1)))
        st = np.random.default_rng(5).uniform(0, 1, (400, 2))
        design = shell.design(st)
        lower = np.linalg.cholesky(design.T @ design / len(st) + 1e-6 * np.eye(49))
        right = design.T @ -((st - 0.5) ** 2).sum(1) / len(st)
        constraints = gs._convexity_rows(shell, (1.0, 1.0), CONFIG)
        coefficients = gs._convex_solve(lower, right, constraints)
        self.assertGreater((constraints @ coefficients).min(), -1e-9)


class TangentSolveTests(unittest.TestCase):
    def test_cellwise_solve_matches_dense_normal_equations(self):
        rng = np.random.default_rng(2)
        m, D = 500, 4
        st = rng.uniform(0, 1, (m, 2))
        st[:3] = [[0, 0], [1, 1], [1, 0.5]]
        points = rng.normal(size=(m, D))
        weights = rng.uniform(0.5, 1, m)
        weights /= weights.sum()
        knots_s, knots_t = gs.open_knots(4, 3), gs.open_knots(2, 3)
        shell = gs.TensorSpline(knots_s, knots_t, 3, np.zeros((8, 6, 1)))
        tangents = np.linalg.qr(rng.normal(size=(m, D, 2)))[0]
        penalty = gs.bending_matrix(knots_s, knots_t, 3)
        design = shell.design(st)
        n = design.shape[1]
        system = np.kron((design.T * weights) @ design + 1e-3 * penalty, np.eye(D))
        target = points.copy()
        for k in range(2):
            t = tangents[:, :, k]
            paired = (design[:, :, None] * t[:, None, :]).reshape(m, n * D)
            system -= 0.8 * (paired.T * weights) @ paired
            target -= 0.8 * (points * t).sum(1)[:, None] * t
        dense = np.linalg.solve(system, ((design.T * weights) @ target).ravel()).reshape(n, D)
        cellwise = gs._tangent_solve(shell, st, weights, points, tangents, penalty, 1e-3, 0.2)
        np.testing.assert_allclose(cellwise, dense, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
