"""A geometric least-squares surface with orthogonal (u, v) coordinates, in any dimension.

Three stages, each fitted on its own:

1. shape: a surface S(s, t) over the chart [0, 1]^2, fitted by weighted orthogonal-distance
   least squares. The fit alternates between each row's closest point on the surface (damped
   Newton steps) and a penalised linear least-squares solve for the coefficients. A
   thin-plate bending energy keeps the surface from chasing noise. The chart (s, t) is only a
   working parameterisation. Two shape models:

   - 'convex' (default): a smooth convex cap, `ConvexSurface`. It is a graph of cubic tensor
     B-spline heights over the rows' weighted principal plane, and its height along one
     normal direction (the rows' mean-curvature direction) is held convex, so the surface
     lies on the boundary of its own convex hull. Every height shares one design matrix, so
     each step is one Cholesky solve plus a small constrained solve for the convex height.
     The patch is trimmed to the convex hull of the rows' closest points for drawing.
   - 'bspline': a general parametric tensor-product cubic B-spline with a fixed chart and no
     convexity. It can follow finer structure but is much slower to fit.
2. u: a smooth scalar tensor spline h(s, t), fitted by penalised least squares to a target
   value at each row's closest point. It is learnt after the shape and does not change it.
3. v: the lines of constant v are the orthogonal trajectories of the lines of constant u,
   that is, the integral curves of u's gradient over the surface, so S_u . S_v = 0 exactly.
   v is surface length along the level curve u = u_zero, measured from where that curve
   passes closest to a given reference point.

   u is a valid coordinate only where its surface gradient stays well away from zero: each
   row's |grad u| is checked at its point and along its line of constant v, and that line
   must reach the traced level curve u = u_zero.

The (s, t) <-> (u, v) correspondence is cached as two tables when the coordinates are
built (`OrthogonalCoordinates`). A point is mapped by finding its closest point on the
surface, reading u there from h, and reading v and the signed `arc_length_parallel` from
the cached chart table.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.interpolate import BSpline, RectBivariateSpline, make_lsq_spline
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import nnls
from scipy.spatial import ConvexHull, Delaunay, cKDTree

MAP_COLUMNS = ('surface_s', 'surface_t', 'surface_u', 'surface_v', 'arc_length_parallel',
               'arc_length_orthogonal', 'surface_distance', 'u_gradient', 'path_u_gradient',
               'coordinate_status')
STATUS = ('interior', 'invalid_u', 'patch_boundary', 'path_extrapolated')
SHAPE_MODELS = ('convex', 'bspline')


@dataclass(frozen=True)
class SurfaceConfig:
    degree: int = 3                    # B-spline degree of u = h(s, t) and of the shape
    shape_model: str = 'convex'
    shape_knots: tuple = (6, 4)        # interior knots of the shape along s and t
    shape_bending: float = 1e-3        # bending-energy weight, relative to the data term
    # Shape-fit weights grow by this share per RMS distance from the centroid (0 = unchanged;
    # 0.2 gives a row at the RMS distance 1.2 times the weight of one at the centroid).
    outer_weight: float = 0.0
    # 'convex': the chart box is the rows' extent in the principal plane, widened by this
    # share of its size on every side.
    box_padding: float = 0.02
    convex_directions: int = 8         # directions d in which d' Hess(f) d >= 0 is imposed
    convex_points: int = 4             # constraint points per knot span, per chart axis
    # 'bspline': share of each residual's tangential part kept in the control-point step
    # (1 = plain point-distance alternation, which converges slowly; smaller values
    # approximate the squared distance to the surface and converge faster).
    tangent_weight: float = 0.1
    u_knots: tuple = (6, 4)            # interior knots of u = h(s, t)
    u_bending: float = 1e-4
    fit_iterations: int = 300
    fit_tolerance: float = 1e-4        # stop when the objective improves by less than this share
    search_grid: int = 65              # coarse closest-point grid per chart axis
    search_every: int = 10             # shape-fit iterations between grid searches (warm starts between)
    newton_steps: int = 30
    newton_tolerance: float = 1e-6     # chart step below which a closest point has converged
    trajectory_steps: int = 64         # RK4 steps from a chart-table node's u to u_zero
    grid_points: int = 129             # nodes per chart axis of the (s, t) -> (u, v) table
    net_points: tuple = (65, 65)       # u levels and v lines of the (u, v) -> (s, t) net
    net_steps: int = 8                 # RK4 steps between consecutive u levels of the net
    # u stops being a valid coordinate where |grad u| on the surface falls below this share
    # of its weighted median at the fitting rows' closest points.
    min_gradient_share: float = 0.1
    zero_curve_tolerance: float = 1e-3  # chart distance by which a traced path may miss u = u_zero
    zero_curve_step: float = 1 / 1024  # chart step when tracing the level curve u = u_zero
    chart_margin: float = 1.0          # how far past the chart edge (tangent extension) the
                                       # level curve u = u_zero is traced
    chunk_rows: int = 4096

    def __post_init__(self):
        for name in ('shape_knots', 'u_knots'):
            knots = getattr(self, name)
            if len(knots) != 2 or any(int(k) != k or k < 0 for k in knots):
                raise ValueError(f'{name} must be two nonnegative integers.')
        if self.degree < 2:
            raise ValueError('The Newton projection needs second derivatives: degree >= 2.')
        if self.shape_model not in SHAPE_MODELS:
            raise ValueError(f'shape_model must be one of {SHAPE_MODELS}.')
        if self.outer_weight < 0:
            raise ValueError('outer_weight must be >= 0.')
        if self.box_padding < 0 or self.convex_directions < 3 or self.convex_points < 1:
            raise ValueError('box_padding must be >= 0, convex_directions >= 3, convex_points >= 1.')


def open_knots(interior, degree):
    """Clamped knot vector on [0, 1] with uniform interior knots."""
    return np.r_[np.zeros(degree + 1), np.linspace(0.0, 1.0, interior + 2)[1:-1], np.ones(degree + 1)]


class TensorSpline:
    """S(s, t) = sum_jk c_jk N_j(s) M_k(t), with c_jk in R^D.

    Beyond the chart [0, 1]^2 each axis's basis continues linearly from its edge value and
    slope, so the surface extends along its edge tangents (C1) instead of along the end
    cubics, which diverge quickly.
    """

    def __init__(self, knots_s, knots_t, degree, coefficients):
        self.knots_s, self.knots_t = np.asarray(knots_s, float), np.asarray(knots_t, float)
        self.degree = int(degree)
        self.ns, self.nt = len(self.knots_s) - degree - 1, len(self.knots_t) - degree - 1
        self.coefficients = np.asarray(coefficients, float).reshape(self.ns, self.nt, -1)
        self._bases = (BSpline(self.knots_s, np.eye(self.ns), degree),
                       BSpline(self.knots_t, np.eye(self.nt), degree))

    @property
    def dimension(self):
        return self.coefficients.shape[2]

    def basis(self, st, order=0):
        """Basis values and derivatives up to `order` along s and t: two lists of (m, n)."""
        st = np.atleast_2d(np.asarray(st, float))
        return [_linear_extension(spline, st[:, axis], order) for axis, spline in enumerate(self._bases)]

    @staticmethod
    def _kron(N, M):
        return (N[:, :, None] * M[:, None, :]).reshape(len(N), -1)

    def design(self, st):
        """(m, ns * nt) row-wise Kronecker design matrix of the tensor basis."""
        (N,), (M,) = self.basis(st)
        return self._kron(N, M)

    def jet(self, st, order=0):
        """{(i, j): d^(i+j) S / ds^i dt^j} for i + j <= order, each (m, D)."""
        N, M = self.basis(st, order)
        flat = self.coefficients.reshape(self.ns * self.nt, -1)
        return {(i, j): self._kron(N[i], M[j]) @ flat
                for i in range(order + 1) for j in range(order + 1 - i)}

    def __call__(self, st):
        return self.jet(st)[(0, 0)]

    def arrays(self, prefix):
        return {f'{prefix}_knots_s': self.knots_s, f'{prefix}_knots_t': self.knots_t,
                f'{prefix}_degree': np.asarray(self.degree), f'{prefix}_coefficients': self.coefficients}

    @classmethod
    def from_arrays(cls, arrays, prefix):
        return cls(arrays[f'{prefix}_knots_s'], arrays[f'{prefix}_knots_t'],
                   int(arrays[f'{prefix}_degree']), arrays[f'{prefix}_coefficients'])


class ConvexSurface:
    """S(s, t) = origin + x e_1 + y e_2 + sum_k h_k(s, t) n_k: B-spline heights over a plane.

    `frame` is an orthonormal basis (D, D): its first two columns e_1, e_2 span the base
    plane, the rest n_k are normal to it. (x, y) = box[0] + box[1] * (s, t) are plane
    coordinates, so the chart [0, 1]^2 covers the box. The heights h_k are a TensorSpline;
    beyond the chart they continue linearly, like the plane.

    `fit_shape` holds the first height f = h_1 convex in (x, y). The surface then lies on the
    boundary of its own convex hull: at every point the hyperplane with normal
    n_1 - f_x e_1 - f_y e_2 supports it. `hull`, if set, holds the chart vertices of the convex
    hull of the fitted rows' closest points, the part of the surface the data covers.
    """

    def __init__(self, origin, frame, box, heights, hull=None):
        self.origin, self.frame = np.asarray(origin, float), np.asarray(frame, float)
        self.box = np.asarray(box, float)          # rows: low corner (x, y), size (x, y)
        self.heights = heights
        self.hull = None if hull is None or len(hull) == 0 else np.asarray(hull, float)

    @property
    def dimension(self):
        return len(self.origin)

    @property
    def coefficients(self):
        return self.heights.coefficients

    def jet(self, st, order=0):
        """{(i, j): d^(i+j) S / ds^i dt^j} for i + j <= order, each (m, D)."""
        st = np.atleast_2d(np.asarray(st, float))
        jet = {key: value @ self.frame[:, 2:].T for key, value in self.heights.jet(st, order).items()}
        jet[(0, 0)] = jet[(0, 0)] + self.origin + (self.box[0] + self.box[1] * st) @ self.frame[:, :2].T
        if order >= 1:
            jet[(1, 0)] = jet[(1, 0)] + self.box[1, 0] * self.frame[:, 0]
            jet[(0, 1)] = jet[(0, 1)] + self.box[1, 1] * self.frame[:, 1]
        return jet

    def __call__(self, st):
        return self.jet(st)[(0, 0)]

    def plane_chart(self, points):
        """Chart position of each point's orthogonal projection onto the base plane."""
        return ((np.asarray(points, float) - self.origin) @ self.frame[:, :2] - self.box[0]) / self.box[1]

    def convex_hessian(self, st):
        """Hessian of the convex height f in plane units, (m, 2, 2)."""
        jet = self.heights.jet(st, order=2)
        (sx, sy), f = self.box[1], {key: value[:, 0] for key, value in jet.items()}
        xy = f[(1, 1)] / (sx * sy)
        return np.stack([np.stack([f[(2, 0)] / sx ** 2, xy], -1), np.stack([xy, f[(0, 2)] / sy ** 2], -1)], -2)

    def inside_hull(self, st):
        """Whether chart points lie inside `hull` (inside the chart when no hull is set)."""
        st = np.atleast_2d(np.asarray(st, float))
        if self.hull is None:
            return np.all((st >= 0) & (st <= 1), axis=1)
        return Delaunay(self.hull).find_simplex(st) >= 0

    def arrays(self, prefix):
        return {f'{prefix}_origin': self.origin, f'{prefix}_frame': self.frame, f'{prefix}_box': self.box,
                f'{prefix}_hull': np.zeros((0, 2)) if self.hull is None else self.hull,
                **self.heights.arrays(f'{prefix}_heights')}

    @classmethod
    def from_arrays(cls, arrays, prefix):
        return cls(arrays[f'{prefix}_origin'], arrays[f'{prefix}_frame'], arrays[f'{prefix}_box'],
                   TensorSpline.from_arrays(arrays, f'{prefix}_heights'), arrays[f'{prefix}_hull'])


def shape_from_arrays(arrays, prefix):
    """The saved shape under `prefix`, whichever model it is."""
    if f'{prefix}_frame' in arrays:
        return ConvexSurface.from_arrays(arrays, prefix)
    return TensorSpline.from_arrays(arrays, prefix)


def _linear_extension(basis, x, order):
    """[N, N', ...] of a 1-D basis at x, continued linearly beyond [0, 1]."""
    edge = np.clip(x, 0.0, 1.0)
    over = (x - edge)[:, None]
    values = [basis(edge, nu) for nu in range(max(order, 1) + 1)]
    outside = over != 0
    extended = [values[0] + over * values[1], values[1]]
    extended += [np.where(outside, 0.0, values[nu]) for nu in range(2, order + 1)]
    return extended[:order + 1]


def _gram(knots, degree, nu, nodes=8):
    """Exact Gram matrix of the nu-th basis derivatives over [0, 1] (Gauss-Legendre per span)."""
    breaks = np.unique(knots)
    x, w = np.polynomial.legendre.leggauss(nodes)
    half = (breaks[1:] - breaks[:-1])[:, None] / 2
    points = ((breaks[1:] + breaks[:-1])[:, None] / 2 + half * x).ravel()
    weights = (half * w).ravel()
    basis = BSpline(knots, np.eye(len(knots) - degree - 1), degree)(points, nu)
    return (basis * weights[:, None]).T @ basis


def bending_matrix(knots_s, knots_t, degree, size=(1.0, 1.0)):
    """Thin-plate energy int |S_ss|^2 + 2 |S_st|^2 + |S_tt|^2 over the chart, as c' P c.

    With `size`, the chart is a box of that size in plane units (x, y), and the energy is
    taken in those units and multiplied by the box area, which puts it in squared units of S,
    like the data term, whatever the scale. The unit box gives the plain chart energy.
    """
    gs = [_gram(knots_s, degree, nu) for nu in range(3)]
    gt = [_gram(knots_t, degree, nu) for nu in range(3)]
    sx, sy = size
    # dx dy = sx sy ds dt, and one more factor sx sy for the area.
    return (sx * sy) ** 2 * (np.kron(gs[2], gt[0]) / sx ** 4 + 2 * np.kron(gs[1], gt[1]) / (sx * sy) ** 2
                             + np.kron(gs[0], gt[2]) / sy ** 4)


def _penalised_solve(design, weights, values, penalty, bending):
    """Weighted penalised least squares: mean squared error + bending * c' P c.

    With weights summing to one and the chart the unit square, both terms are in squared
    units of `values`, so `bending` is unitless.
    """
    weighted = design.T * weights
    return np.linalg.solve(weighted @ design + bending * penalty, weighted @ values)


def _tangent_solve(shell, st, weights, points, tangents, penalty, bending, tangent_weight):
    """Control points minimising the residuals' normal part plus `tangent_weight` of their
    tangential part at the current closest points; `tangents` are orthonormal, (m, D, 2).

    Each row touches only the (degree + 1)^2 basis functions of its knot cell, so the
    tangential correction to the normal equations is accumulated cell by cell.
    """
    m, D, k = len(points), points.shape[1], shell.degree + 1
    design = shell.design(st)
    weighted = design.T * weights
    system = np.kron(weighted @ design + bending * penalty, np.eye(D))
    first = [np.clip(np.searchsorted(knots, st[:, axis], side='right') - 1, shell.degree, n - 1) - shell.degree
             for axis, (knots, n) in enumerate(((shell.knots_s, shell.ns), (shell.knots_t, shell.nt)))]
    columns = ((first[0][:, None] + np.arange(k))[:, :, None] * shell.nt
               + (first[1][:, None] + np.arange(k))[:, None, :]).reshape(m, k * k)
    local = np.take_along_axis(design, columns, axis=1)
    target = points.copy()
    for j in range(tangents.shape[2]):
        target -= (1 - tangent_weight) * (points * tangents[:, :, j]).sum(1)[:, None] * tangents[:, :, j]
    cells = first[0] * shell.nt + first[1]
    for cell in np.unique(cells):
        rows = np.flatnonzero(cells == cell)
        index = (columns[rows[0]][:, None] * D + np.arange(D)).ravel()
        block = np.zeros((len(index), len(index)))
        for j in range(tangents.shape[2]):
            paired = (local[rows][:, :, None] * tangents[rows, None, :, j]).reshape(len(rows), -1)
            block += (paired.T * weights[rows]) @ paired
        system[np.ix_(index, index)] -= (1 - tangent_weight) * block
    return np.linalg.solve(system, (weighted @ target).ravel()).reshape(-1, D)


def orthonormal_tangents(surface, st):
    jet = surface.jet(st, order=1)
    return np.linalg.qr(np.stack([jet[(1, 0)], jet[(0, 1)]], axis=-1))[0]


def closest_points(surface, points, start=None, config=None, search=True, bounds=(0.0, 1.0)):
    """Each point's closest point on the chart [0, 1]^2 of `surface` (on [bounds]^2 if given).

    Starts from the better of a coarse grid search over [0, 1]^2 and `start` (from `start`
    alone when `search` is False), then takes damped Newton steps on the squared distance,
    clamped to the bounds. Returns (st, squared distance).
    """
    c = config or SurfaceConfig()
    points = np.asarray(points, float)
    if start is None and not search:
        raise ValueError('closest_points needs a start or a grid search.')
    if search:
        axis = np.linspace(0.0, 1.0, c.search_grid)
        grid = np.stack(np.meshgrid(axis, axis, indexing='ij'), axis=-1).reshape(-1, 2)
        grid_values = surface(grid)
        grid_norms = (grid_values ** 2).sum(1)
        st = np.empty((len(points), 2))
        for low in range(0, len(points), c.chunk_rows):
            chunk = points[low:low + c.chunk_rows]
            distance = grid_norms[None, :] - 2 * chunk @ grid_values.T
            st[low:low + len(chunk)] = grid[np.argmin(distance, axis=1)]
    if start is not None:
        start = np.clip(np.asarray(start, float), *bounds)
        if search:
            better = (((points - surface(start)) ** 2).sum(1) < ((points - surface(st)) ** 2).sum(1))
            st[better] = start[better]
        else:
            st = start.copy()

    value = ((points - surface(st)) ** 2).sum(1)
    damping = np.full(len(points), 1e-3)
    active = np.ones(len(points), bool)
    for _ in range(c.newton_steps):
        if not active.any():
            break
        index = np.flatnonzero(active)
        converged = _newton_step(surface, points, st, value, damping, index, bounds, c.newton_tolerance)
        active[index[converged]] = False
    return st, value


def global_closest_points(surface, points, config=None, candidates=4):
    """`closest_points` from several starts, keeping each point's closest result.

    The starts are the point's `candidates` best local minima of the squared distance over the
    coarse search grid (fewer where the grid has fewer), so a point lying about equally close
    to two parts of the surface settles on the closer one instead of on whichever basin the
    single best grid node belongs to. Returns (st, squared distance).
    """
    c = config or SurfaceConfig()
    points = np.asarray(points, float)
    axis = np.linspace(0.0, 1.0, c.search_grid)
    n = len(axis)
    grid = np.stack(np.meshgrid(axis, axis, indexing='ij'), axis=-1).reshape(-1, 2)
    grid_values = surface(grid)
    grid_norms = (grid_values ** 2).sum(1)
    starts = np.empty((len(points), candidates, 2))
    for low in range(0, len(points), c.chunk_rows):
        chunk = points[low:low + c.chunk_rows]
        distance = (grid_norms[None, :] - 2 * chunk @ grid_values.T).reshape(-1, n, n)
        padded = np.pad(distance, ((0, 0), (1, 1), (1, 1)), constant_values=np.inf)
        local = np.ones(distance.shape, bool)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di or dj:
                    local &= distance <= padded[:, 1 + di:1 + di + n, 1 + dj:1 + dj + n]
        score = np.where(local, distance, np.inf).reshape(len(chunk), -1)
        best = np.argsort(score, axis=1)[:, :candidates]
        best = np.where(np.isfinite(np.take_along_axis(score, best, 1)), best, best[:, :1])
        starts[low:low + len(chunk)] = grid[best]
    st, value = closest_points(surface, points, start=starts[:, 0], config=c, search=False)
    for k in range(1, candidates):
        other = np.any(starts[:, k] != starts[:, 0], axis=1)
        if not other.any():
            continue
        trial_st, trial_value = closest_points(surface, points[other], start=starts[other, k], config=c,
                                               search=False)
        better = trial_value < value[other]
        index = np.flatnonzero(other)[better]
        st[index], value[index] = trial_st[better], trial_value[better]
    return st, value


def _newton_step(surface, points, st, value, damping, index, bounds=(0.0, 1.0), step_tolerance=1e-6):
    """One damped Newton step on the rows in `index`, updating `st`, `value` and `damping`
    in place. Returns, per row of `index`, whether it has converged."""
    points, current = points[index], st[index]
    jet = surface.jet(current, order=2)
    residual = points - jet[(0, 0)]
    tangents = np.stack([jet[(1, 0)], jet[(0, 1)]], axis=-1)                # (m, D, 2)
    gradient = -np.einsum('md,mdk->mk', residual, tangents)
    curvature = np.einsum('md,mdk->mk', residual,
                          np.stack([jet[(2, 0)], jet[(1, 1)], jet[(0, 2)]], axis=-1))
    hessian = np.einsum('mdk,mdl->mkl', tangents, tangents)
    hessian[:, 0, 0] -= curvature[:, 0]
    hessian[:, [0, 1], [1, 0]] -= curvature[:, 1:2]
    hessian[:, 1, 1] -= curvature[:, 2]
    size = np.einsum('mdk,mdk->m', tangents, tangents) / 2 + 1e-300
    system = hessian + (damping[index] * size)[:, None, None] * np.eye(2)
    trial = np.clip(current - np.linalg.solve(system, gradient[:, :, None])[:, :, 0], *bounds)
    trial_value = ((points - surface(trial)) ** 2).sum(1)
    accept = np.isfinite(trial_value) & (trial_value <= value[index])
    moved = np.abs(trial - current).max(1)
    st[index[accept]], value[index[accept]] = trial[accept], trial_value[accept]
    damping[index] = np.where(accept, np.maximum(damping[index] / 3, 1e-12),
                              np.minimum(damping[index] * 4, 1e12))
    return (accept & (moved < step_tolerance)) | (damping[index] >= 1e12)


def initial_chart(points, guide, weights, interior_knots=4):
    """A starting chart for the shape fit: s from `guide`, t from the main residual direction.

    s is `guide` rescaled to [0, 1]. t is the leading weighted principal direction of what a
    cubic spline of the points on `guide` leaves, also rescaled to [0, 1]. This only picks
    the starting point of the geometric fit; the fitted shape minimises closest-point
    distance whatever the start.
    """
    guide = np.asarray(guide, float)
    order = np.argsort(guide, kind='stable')
    ordered = guide[order]
    knots = np.r_[[ordered[0]] * 4,
                  np.quantile(ordered, np.arange(1, interior_knots + 1) / (interior_knots + 1)),
                  [ordered[-1]] * 4]
    curve = make_lsq_spline(ordered, points[order], knots, k=3, w=np.sqrt(weights[order]))
    residual = points - curve(guide)
    covariance = (residual * weights[:, None]).T @ residual / weights.sum()
    direction = np.linalg.eigh(covariance)[1][:, -1]
    across = residual @ direction

    def unit(values):
        return (values - values.min()) / (values.max() - values.min())
    return np.column_stack([unit(guide), unit(across)])


def fit_shape(points, weights, start=None, config=None, progress=None):
    """Orthogonal-distance least-squares surface through weighted points (`config.shape_model`).

    `start` is the starting chart of a 'bspline' shape (`initial_chart`); a 'convex' shape
    starts from the rows' weighted principal plane and ignores it. Returns (surface, st,
    squared distances, diagnostics). `weights` are normalised to sum to one, so the data
    term is a weighted mean squared distance.

    With `config.outer_weight` = a > 0 the fit uses `outer_weights`: each weight times
    1 + a r / r_rms, r being the row's distance from the weighted centroid. The objective is
    in those weights; the distance and variance diagnostics stay in the given ones.
    """
    c = config or SurfaceConfig()
    points = np.asarray(points, float)
    weights = np.asarray(weights, float) / np.sum(weights)
    if c.shape_model == 'bspline' and start is None:
        raise ValueError("A 'bspline' shape needs a starting chart.")
    fitting = outer_weights(points, weights, c.outer_weight)
    fit = _fit_convex_shape if c.shape_model == 'convex' else _fit_bspline_shape
    surface, st, penalty, history, plain_steps = fit(points, fitting, start, c, progress)
    st, distance = closest_points(surface, points, start=st, config=c)
    coefficients = surface.coefficients.reshape(len(penalty), -1)
    bending = float(np.einsum('pd,pq,qd->', coefficients, penalty, coefficients))
    history[-1] = float(fitting @ distance + c.shape_bending * bending)
    centred = points - weights @ points
    extra = {}
    if isinstance(surface, ConvexSurface):
        surface.hull = st[ConvexHull(st).vertices]
        extra = dict(hull_chart_share=float(ConvexHull(surface.hull).volume), **height_convexity(surface))
    diagnostics = dict(
        shape_model=c.shape_model, iterations=len(history), converged=len(history) < c.fit_iterations,
        plain_steps=plain_steps, objective=history[-1], objective_start=history[0],
        bending_energy=bending, outer_weight=c.outer_weight,
        # Mean square distance from the centroid under the fitting weights, relative to the given ones.
        fit_weight_radius_ratio=float(fitting @ (centred ** 2).sum(1) / (weights @ (centred ** 2).sum(1))),
        weighted_rms_distance=float(np.sqrt(weights @ distance)),
        variance_share_on_surface=float(1 - weights @ distance / (weights @ (centred ** 2).sum(1))),
        rows_on_patch_boundary=int(on_boundary(st).sum()),
        **residual_orthogonality(surface, points, st),
        **chart_regularity(surface),
        **convexity(surface), **extra)
    return surface, st, distance, diagnostics


def outer_weights(points, weights, strength):
    """`weights` times 1 + strength * r / r_rms, renormalised to sum to one.

    r is each point's distance from the weighted centroid and r_rms its weighted RMS, so a
    point at the typical distance gains `strength` relative to one at the centroid, and a
    point twice as far gains twice that. Zero strength returns `weights` unchanged.
    """
    if strength == 0:
        return weights
    distance = np.linalg.norm(points - weights @ points, axis=1)
    boosted = weights * (1 + strength * distance / np.sqrt(weights @ distance ** 2))
    return boosted / boosted.sum()


def _convexity_rows(shell, size, c):
    """Rows G of the linear constraints G c >= 0 that hold a height convex in plane units.

    Each row is d' Hess(f) d at one chart point for one direction d in the plane. The points
    are `c.convex_points` per knot span along each axis, spans' ends included; the directions
    are `c.convex_directions` evenly spread over a half turn. With 8 directions the smallest
    eigenvalue of the Hessian at a constraint point is at least -4% of the largest. Rows are
    scaled to unit length.
    """
    axes = []
    for knots in (shell.knots_s, shell.knots_t):
        breaks = np.unique(knots)
        axes.append(np.unique(np.concatenate([np.linspace(a, b, c.convex_points + 1)
                                              for a, b in zip(breaks[:-1], breaks[1:])])))
    st = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 2)
    N, M = shell.basis(st, order=2)
    sx, sy = size
    fxx, fxy, fyy = (shell._kron(N[2], M[0]) / sx ** 2, shell._kron(N[1], M[1]) / (sx * sy),
                     shell._kron(N[0], M[2]) / sy ** 2)
    angles = np.pi * np.arange(c.convex_directions) / c.convex_directions
    rows = np.concatenate([np.cos(a) ** 2 * fxx + 2 * np.cos(a) * np.sin(a) * fxy + np.sin(a) ** 2 * fyy
                           for a in angles])
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def _convex_solve(lower, right, constraints):
    """argmin 1/2 c' M c - right' c subject to constraints @ c >= 0, where M = lower lower'.

    Least distance programming through NNLS (Lawson and Hanson, ch. 23): with
    u = lower' (c - c0) and c0 = M^-1 right, the problem is min |u| subject to E u >= g,
    E = constraints lower'^-1 and g = -constraints c0.
    """
    free = cho_solve((lower, True), right)
    gap = -constraints @ free
    if np.all(gap <= 1e-12 * np.abs(constraints @ free).max(initial=1.0)):
        return free                                     # the unconstrained optimum is convex
    E = solve_triangular(lower, constraints.T, lower=True).T
    n = E.shape[1]
    y = nnls(np.vstack([E.T, gap]), np.r_[np.zeros(n), 1.0], maxiter=50 * len(gap))[0]
    residual = np.r_[E.T @ y, gap @ y - 1.0]
    return free + solve_triangular(lower.T, -residual[:n] / residual[n], lower=False)


def _fit_convex_shape(points, weights, start, c, progress):
    """Alternating fit of a ConvexSurface; `start` is not used.

    The base plane is the rows' weighted principal plane and the chart box their extent in it.
    The first unconstrained fit of the heights picks the convex direction n_1: the weighted
    mean of the heights' Laplacians at the rows, which is where the surface bends on average,
    with the sign that makes the height along it convex. Then each iteration fits every height
    at the current closest points (one Cholesky factor serves them all, and the height along
    n_1 is solved subject to convexity) and moves the closest points by warm-started Newton
    steps. Both steps lower the same objective.

    Returns (surface, st, penalty, objective history, plain steps taken (always 0)).
    """
    origin = weights @ points
    centred = points - origin
    frame = np.linalg.eigh((centred * weights[:, None]).T @ centred)[1][:, ::-1]
    xy = centred @ frame[:, :2]
    low, high = xy.min(0), xy.max(0)
    pad = c.box_padding * (high - low)
    box = np.array([low - pad, high - low + 2 * pad])
    st = (xy - box[0]) / box[1]
    knots_s, knots_t = (open_knots(k, c.degree) for k in c.shape_knots)
    shell = TensorSpline(knots_s, knots_t, c.degree, np.zeros((len(knots_s) - c.degree - 1,
                                                               len(knots_t) - c.degree - 1, 1)))
    penalty = bending_matrix(knots_s, knots_t, c.degree, box[1])
    constraints = _convexity_rows(shell, box[1], c)

    def fit_heights(st, frame, convex):
        design = shell.design(st)
        weighted = design.T * weights
        lower = np.linalg.cholesky(weighted @ design + c.shape_bending * penalty)
        right = weighted @ (centred @ frame[:, 2:])
        coefficients = cho_solve((lower, True), right)
        if convex:
            coefficients[:, 0] = _convex_solve(lower, right[:, 0], constraints)
        return ConvexSurface(origin, frame, box, TensorSpline(knots_s, knots_t, c.degree, coefficients))

    # The convex direction n_1, from the unconstrained heights over the principal plane: their
    # weighted mean Laplacian at the rows. Only the height along n_1 is held convex; the others
    # may bend either way, so the surface can be a saddle.
    free = fit_heights(st, frame, convex=False)
    jet = free.heights.jet(st, order=2)
    laplacian = weights @ (jet[(2, 0)] / box[1, 0] ** 2 + jet[(0, 2)] / box[1, 1] ** 2)
    rotation = np.linalg.qr(np.column_stack([laplacian, np.eye(len(laplacian))]))[0]
    rotation[:, 0] *= np.sign(rotation[:, 0] @ laplacian)
    frame = np.column_stack([frame[:, :2], frame[:, 2:] @ rotation])

    def objective(surface, distance):
        coefficients = surface.coefficients.reshape(len(penalty), -1)
        return float(weights @ distance + c.shape_bending * np.einsum('pd,pq,qd->', coefficients, penalty,
                                                                     coefficients))
    surface = fit_heights(st, frame, convex=True)
    distance = ((points - surface(st)) ** 2).sum(1)
    history = [objective(surface, distance)]
    for iteration in range(1, c.fit_iterations):
        st, distance = closest_points(surface, points, start=st, config=c, search=False)
        surface = fit_heights(st, frame, convex=True)
        distance = ((points - surface(st)) ** 2).sum(1)
        history.append(objective(surface, distance))
        if progress:
            progress(iteration, history[-1])
        if history[-2] - history[-1] <= c.fit_tolerance * history[-2]:
            break
    return surface, st, penalty, history, 0


def _fit_bspline_shape(points, weights, start, c, progress):
    """Alternating fit of a TensorSpline on a fixed chart.

    Returns (surface, st, penalty, objective history, plain steps taken).
    """
    knots_s, knots_t = (open_knots(k, c.degree) for k in c.shape_knots)
    penalty = bending_matrix(knots_s, knots_t, c.degree)
    empty = TensorSpline(knots_s, knots_t, c.degree, np.zeros((len(knots_s) - c.degree - 1,
                                                               len(knots_t) - c.degree - 1, 1)))
    st = np.clip(np.asarray(start, float), 0.0, 1.0)
    design = empty.design(st)
    coefficients = _penalised_solve(design, weights, points, penalty, c.shape_bending)
    bending_lambda = c.shape_bending

    def objective(coefficients, distance):
        return float(weights @ distance + bending_lambda * np.einsum('pd,pq,qd->', coefficients, penalty,
                                                                     coefficients))
    surface = TensorSpline(knots_s, knots_t, c.degree, coefficients)
    st, distance = closest_points(surface, points, start=st, config=c)
    history, plain_steps = [objective(coefficients, distance)], 0
    for iteration in range(1, c.fit_iterations):
        design = empty.design(st)
        candidates = []
        if c.tangent_weight < 1:
            # A row whose closest point is on the patch edge keeps its full residual: its
            # tangential part is real distance, which only growing the patch can remove.
            tangents = orthonormal_tangents(surface, st) * ~on_boundary(st)[:, None, None]
            candidates.append(_tangent_solve(empty, st, weights, points, tangents,
                                             penalty, bending_lambda, c.tangent_weight))
        candidates.append(None)  # the plain step, taken if the faster one does not lower the objective
        for candidate in candidates:
            if candidate is None:
                candidate = _penalised_solve(design, weights, points, penalty, c.shape_bending)
                plain_steps += 1
            trial = TensorSpline(knots_s, knots_t, c.degree, candidate)
            trial_st, trial_distance = closest_points(trial, points, start=st, config=c,
                                                      search=iteration % c.search_every == 0)
            value = objective(candidate, trial_distance)
            if value <= history[-1]:
                break
        if value > history[-1]:
            break                      # neither step improves: converged to working precision
        surface, st, distance, coefficients = trial, trial_st, trial_distance, candidate
        history.append(value)
        if progress:
            progress(iteration, value)
        if history[-2] - value <= c.fit_tolerance * history[-2]:
            break
    return surface, st, penalty, history, plain_steps


def on_boundary(st, tolerance=1e-9, bounds=(0.0, 1.0)):
    return np.any((st <= bounds[0] + tolerance) | (st >= bounds[1] - tolerance), axis=1)


def residual_orthogonality(surface, points, st):
    """How orthogonal the interior rows' residuals are to the surface: |cos| to the tangent plane."""
    inside = ~on_boundary(st)
    jet = surface.jet(st[inside], order=1)
    residual = points[inside] - jet[(0, 0)]
    basis = np.linalg.qr(np.stack([jet[(1, 0)], jet[(0, 1)]], axis=-1))[0]    # (m, D, 2)
    along = np.linalg.norm(np.einsum('md,mdk->mk', residual, basis), axis=1)
    cosine = along / np.maximum(np.linalg.norm(residual, axis=1), 1e-300)
    return dict(residual_tangent_cosine_median=float(np.median(cosine)),
                residual_tangent_cosine_max=float(cosine.max()))


def metric(jet):
    """First fundamental form (E, F, G) from a jet holding the first derivatives."""
    Ss, St = jet[(1, 0)], jet[(0, 1)]
    return (Ss * Ss).sum(1), (Ss * St).sum(1), (St * St).sum(1)


def chart_regularity(surface, grid=65):
    """Area element sqrt(EG - F^2) over the chart: its smallest value relative to its median."""
    axis = np.linspace(0.0, 1.0, grid)
    st = np.stack(np.meshgrid(axis, axis, indexing='ij'), axis=-1).reshape(-1, 2)
    E, F, G = metric(surface.jet(st, order=1))
    area = np.sqrt(np.maximum(E * G - F ** 2, 0.0))
    return dict(area_element_min_over_median=float(area.min() / np.median(area)))


def convexity(surface, grid=33):
    """Whether the surface bends one way, like a cap, rather than as a saddle.

    At each chart grid point the second fundamental form is taken along the unit mean-curvature
    normal; the surface is locally convex there when that form is definite. Returns the share
    of the chart where it is, and the median ratio of its two principal curvatures (1 = bends
    equally both ways, 0 = a cylinder, negative = a saddle).
    """
    axis = np.linspace(0.0, 1.0, grid)
    jet = surface.jet(np.stack(np.meshgrid(axis, axis, indexing='ij'), axis=-1).reshape(-1, 2), order=2)
    frame = np.linalg.qr(np.stack([jet[(1, 0)], jet[(0, 1)]], axis=-1))[0]
    ss, st, tt = (jet[key] - np.einsum('mdk,mek,me->md', frame, frame, jet[key])
                  for key in ((2, 0), (1, 1), (0, 2)))
    E, F, G = metric(jet)
    det = E * G - F ** 2
    mean = (G[:, None] * ss - 2 * F[:, None] * st + E[:, None] * tt) / det[:, None]
    normal = mean / np.maximum(np.linalg.norm(mean, axis=1), 1e-300)[:, None]
    L, M, N = ((part * normal).sum(1) for part in (ss, st, tt))
    # Principal curvatures along the normal: eigenvalues of g^-1 II.
    half_trace = (G * L - 2 * F * M + E * N) / (2 * det)
    product = (L * N - M ** 2) / det
    root = np.sqrt(np.maximum(half_trace ** 2 - product, 0.0))
    small, large = half_trace - root, half_trace + root
    ratio = np.where(np.abs(large) > 1e-300, small / np.where(large == 0, 1, large), 0.0)
    return dict(convex_chart_share=float((product > 0).mean()),
                principal_curvature_ratio_median=float(np.median(ratio)))


def height_convexity(surface, grid=65):
    """How convex a ConvexSurface's height f is over the chart, from its Hessian's eigenvalues.

    Returns the smallest eigenvalue relative to the median largest one (>= 0 when f is convex
    everywhere; the constraints allow about -0.04 between their points), and the share of the
    chart where the Hessian is positive semidefinite to that tolerance.
    """
    axis = np.linspace(0.0, 1.0, grid)
    eigenvalues = np.linalg.eigvalsh(surface.convex_hessian(
        np.stack(np.meshgrid(axis, axis, indexing='ij'), axis=-1).reshape(-1, 2)))
    scale = max(float(np.median(eigenvalues[:, 1])), 1e-300)
    return dict(convex_height_min_eigen_ratio=float(eigenvalues[:, 0].min() / scale),
                convex_height_psd_share=float((eigenvalues[:, 0] >= -0.05 * np.maximum(eigenvalues[:, 1], 0)).mean()))


def fit_scalar(st, values, weights, config=None):
    """u = h(s, t): penalised least-squares tensor spline of `values` on chart positions."""
    c = config or SurfaceConfig()
    weights = np.asarray(weights, float) / np.sum(weights)
    knots_s, knots_t = (open_knots(k, c.degree) for k in c.u_knots)
    shell = TensorSpline(knots_s, knots_t, c.degree,
                         np.zeros((len(knots_s) - c.degree - 1, len(knots_t) - c.degree - 1, 1)))
    coefficients = _penalised_solve(shell.design(st), weights, np.asarray(values, float)[:, None],
                                    bending_matrix(knots_s, knots_t, c.degree), c.u_bending)
    return TensorSpline(knots_s, knots_t, c.degree, coefficients)


def weighted_median(values, weights):
    order = np.argsort(values, kind='stable')
    cumulative = np.cumsum(np.asarray(weights, float)[order])
    return float(np.asarray(values, float)[order][np.searchsorted(cumulative, cumulative[-1] / 2)])


TABLE_KEYS = ('grid_axis', 'grid_surface_v', 'grid_parallel_ratio', 'grid_path_left_chart',
              'grid_path_u_gradient', 'grid_valid',
              'net_u', 'net_v', 'net_st', 'net_arc_length_parallel')


class OrthogonalCoordinates:
    """(u, v) on a fitted shape: u = h(s, t), v along the orthogonal trajectories of u.

    Building the coordinates caches two tables:

    - the net, (u, v) -> (s, t): each line of constant v starts on the level curve
      u = u_zero at its v and is integrated along u's surface gradient through evenly spaced
      u levels, accumulating its surface length (`arc_length_parallel`) as it goes;
    - the chart table, (s, t) -> (v, arc_length_parallel): every node of a regular chart grid
      is integrated along its line of constant v back to u = u_zero, and bicubic splines
      interpolate between the nodes.

    Points are mapped through the chart table, with u read from h exactly. The two tables are
    built independently, so their round trip (`table_diagnostics`) checks both.
    """

    def __init__(self, shape, u, u_zero, zero_st, zero_v, config, gradient_floor=0.0, tables=None):
        self.shape, self.u, self.u_zero = shape, u, float(u_zero)
        self.zero_st, self.zero_v = np.asarray(zero_st, float), np.asarray(zero_v, float)
        self.config = config
        self.gradient_floor = float(gradient_floor)
        self._tree = cKDTree(self.zero_st)
        self.tables = None
        if tables is not None:
            self._set_tables(tables)

    # ---- construction

    @classmethod
    def build(cls, shape, u, u_zero, reference, footpoints, weights, config=None):
        """Trace the level curve u = u_zero, set v = 0 where it passes closest to `reference`,
        and cache the net and the chart table.

        `footpoints` are the fitting rows' closest points (chart positions) and `weights` their
        weights; they set the gradient floor, `min_gradient_share` of the weighted median |grad u|.
        """
        c = config or SurfaceConfig()
        seed = cls._zero_seed(shape, u, u_zero, c)
        self = cls(shape, u, u_zero, seed[None], [0.0], c)
        halves = [self._trace_level(seed, sign) for sign in (-1.0, 1.0)]
        zero_st = np.vstack([halves[0][::-1], seed[None], halves[1]])
        steps = np.diff(zero_st, axis=0)
        middle = (zero_st[1:] + zero_st[:-1]) / 2
        E, F, G = metric(shape.jet(middle, order=1))
        lengths = np.sqrt(E * steps[:, 0] ** 2 + 2 * F * steps[:, 0] * steps[:, 1] + G * steps[:, 1] ** 2)
        zero_v = np.r_[0.0, np.cumsum(lengths)]
        origin = int(np.argmin(((shape(zero_st) - np.asarray(reference, float)) ** 2).sum(1)))
        zero_v -= zero_v[origin]
        # v increases with t where the zero curve passes the reference.
        neighbour = min(origin + 1, len(zero_st) - 1)
        if zero_st[neighbour, 1] < zero_st[max(origin - 1, 0), 1]:
            zero_v = -zero_v
        self = cls(shape, u, u_zero, zero_st, zero_v, c)
        self.gradient_floor = c.min_gradient_share * weighted_median(self.gradient_norm(footpoints), weights)
        tables = self._chart_table()
        tables.update(self._net())
        self._set_tables(tables)
        return self

    @staticmethod
    def _zero_seed(shape, u, u_zero, c, grid=129):
        """A point with u = u_zero inside the chart, near the chart's middle.

        It is searched for along chart lines through the middle first, along the chart axis
        in which u changes most (the chart's axes need not follow u), then along lines
        further out.
        """
        axis = np.linspace(0.0, 1.0, grid)
        grid_u = u(np.stack(np.meshgrid(axis, axis, indexing='ij'), axis=-1).reshape(-1, 2))[:, 0]
        grid_u = grid_u.reshape(grid, grid)
        along = int(np.abs(np.diff(grid_u, axis=1)).mean() > np.abs(np.diff(grid_u, axis=0)).mean())

        def point(position, fixed):
            return np.array([fixed, position] if along else [position, fixed])

        for fixed in np.r_[0.5, axis[np.argsort(np.abs(axis - 0.5))]]:
            values = u(np.array([point(x, fixed) for x in axis]))[:, 0] - u_zero
            crossing = np.flatnonzero(np.sign(values[:-1]) != np.sign(values[1:]))
            if len(crossing):
                i = crossing[np.argmin(np.abs(axis[crossing] - 0.5))]
                low, high = axis[i], axis[i + 1]
                for _ in range(60):
                    middle = (low + high) / 2
                    if np.sign(u(point(middle, fixed)[None])[0, 0] - u_zero) == np.sign(values[i]):
                        low = middle
                    else:
                        high = middle
                return point((low + high) / 2, fixed)
        raise ValueError(f'u never reaches {u_zero} inside the chart.')

    def _trace_level(self, seed, sign):
        """Chart points along the level curve u = u_zero from `seed` in one direction."""
        c, lo, hi = self.config, -self.config.chart_margin, 1 + self.config.chart_margin
        points, current = [], seed.copy()
        for _ in range(int(4 * (hi - lo) / c.zero_curve_step)):
            jet = self.u.jet(current[None], order=1)
            direction = sign * np.array([-jet[(0, 1)][0, 0], jet[(1, 0)][0, 0]])
            current = current + c.zero_curve_step * direction / np.linalg.norm(direction)
            current = self._to_level(current[None], self.u_zero)[0]
            if np.any(current < lo) or np.any(current > hi):
                break
            points.append(current.copy())
        return np.array(points).reshape(-1, 2)

    def _chart_table(self):
        """Integrate every node of a regular chart grid back to u = u_zero.

        `arc_length_parallel` is stored as its ratio to u - u_zero (the mean of 1 / |grad u|
        along the path), which is smooth and positive, so the mapped length takes its sign and
        its zero exactly from u.

        A node is invalid where its path breaks down (u has a critical point on the way) or
        lands off the traced level curve u = u_zero (which is then split, so it is not the
        curve that sets v). Invalid nodes take their nearest valid node's values, keeping the
        splines finite, and every point beside one is marked `invalid_u` when mapped.
        """
        axis = np.linspace(0.0, 1.0, self.config.grid_points)
        st = np.stack(np.meshgrid(axis, axis, indexing='ij'), axis=-1).reshape(-1, 2)
        values = self.integrate_coordinates(st)
        du = values['surface_u'] - self.u_zero
        with np.errstate(all='ignore'):
            ratio = np.where(np.abs(du) > 1e-12, values['arc_length_parallel'] / du, 1 / self.gradient_norm(st))
        valid = (np.isfinite(values['surface_v']) & np.isfinite(ratio)
                 & (values['zero_curve_miss'] <= self.config.zero_curve_tolerance))
        if not valid.any():
            raise ValueError('No chart-table node reaches the level curve u = u_zero.')
        nearest = cKDTree(st[valid]).query(st)[1]

        def filled(column):
            return np.where(valid, column, column[valid][nearest]).reshape(len(axis), len(axis))
        return dict(grid_axis=axis, grid_surface_v=filled(values['surface_v']),
                    grid_parallel_ratio=filled(ratio),
                    grid_path_left_chart=values['path_left_chart'].reshape(len(axis), len(axis)),
                    grid_path_u_gradient=np.nan_to_num(values['path_u_gradient'], nan=0.0).reshape(
                        len(axis), len(axis)),
                    grid_valid=valid.reshape(len(axis), len(axis)))

    def _net(self):
        """Lines of constant v integrated out from u = u_zero through evenly spaced u levels.

        v lines are evenly spaced over the part of the zero curve inside the chart; u levels,
        spaced evenly over the range of u on the chart, include u_zero. A line stops (NaN) where
        it leaves the chart margin.
        """
        c = self.config
        n_u, n_v = c.net_points
        axis = np.linspace(0.0, 1.0, c.grid_points)
        chart_u = self.u(np.stack(np.meshgrid(axis, axis, indexing='ij'), axis=-1).reshape(-1, 2))[:, 0]
        du = (chart_u.max() - chart_u.min()) / (n_u - 1)
        u_levels = self.u_zero + du * np.arange(np.ceil((chart_u.min() - self.u_zero) / du),
                                                np.floor((chart_u.max() - self.u_zero) / du) + 1)
        zero = int(np.argmin(np.abs(u_levels - self.u_zero)))
        u_levels[zero] = self.u_zero

        inside = np.all((self.zero_st >= 0) & (self.zero_st <= 1), axis=1)
        v_levels = np.linspace(self.zero_v[inside].min(), self.zero_v[inside].max(), n_v)
        order = np.argsort(self.zero_v)
        seeds = np.column_stack([np.interp(v_levels, self.zero_v[order], self.zero_st[order, k])
                                 for k in range(2)])
        seeds = self._to_level(seeds, self.u_zero)

        st = np.full((len(u_levels), n_v, 2), np.nan)
        length = np.full((len(u_levels), n_v), np.nan)
        st[zero], length[zero] = seeds, 0.0
        lo, hi = -c.chart_margin, 1 + c.chart_margin
        for sign, levels in ((1.0, range(zero + 1, len(u_levels))), (-1.0, range(zero - 1, -1, -1))):
            current, travelled, alive, previous = seeds.copy(), np.zeros(n_v), np.ones(n_v, bool), self.u_zero
            for i in levels:
                with np.errstate(all='ignore'):
                    end, step, _, _ = self._integrate(current, np.full(n_v, u_levels[i] - previous), c.net_steps)
                alive &= np.all(np.isfinite(end) & (end >= lo) & (end <= hi), axis=1)
                if not alive.any():
                    break
                current = np.where(alive[:, None], end, current)
                travelled = travelled + np.where(alive, step, 0.0)
                st[i, alive], length[i, alive] = end[alive], sign * travelled[alive]
                previous = u_levels[i]
        return dict(net_u=u_levels, net_v=v_levels, net_st=st, net_arc_length_parallel=length)

    def _set_tables(self, tables):
        missing = [key for key in TABLE_KEYS if key not in tables]
        if missing:
            raise ValueError(f'Coordinate tables lack {missing}; rebuild the coordinates.')
        self.tables = {key: np.asarray(tables[key]) for key in TABLE_KEYS}
        axis = self.tables['grid_axis']
        self._splines = {key: RectBivariateSpline(axis, axis, self.tables[f'grid_{key}'], kx=3, ky=3, s=0)
                         for key in ('surface_v', 'parallel_ratio')}

    # ---- geometry

    def _gradient(self, st):
        """u's surface gradient in chart components, g^-1 grad h, and its squared length."""
        shape_jet, u_jet = self.shape.jet(st, order=1), self.u.jet(st, order=1)
        E, F, G = metric(shape_jet)
        hs, ht = u_jet[(1, 0)][:, 0], u_jet[(0, 1)][:, 0]
        det = E * G - F ** 2
        up = np.column_stack([(G * hs - F * ht) / det, (E * ht - F * hs) / det])
        return up, hs * up[:, 0] + ht * up[:, 1]

    def gradient_norm(self, st):
        """|grad u| on the surface: target units per unit of surface length."""
        return np.sqrt(np.maximum(self._gradient(np.atleast_2d(np.asarray(st, float)))[1], 0.0))

    def _flow(self, st):
        """d(s, t)/du along u's surface gradient, and d(length)/du = 1 / |grad u|."""
        up, squared = self._gradient(st)
        return up / squared[:, None], 1 / np.sqrt(squared)

    def _to_level(self, st, level, steps=4):
        """Newton steps along u's surface gradient onto u = level."""
        for _ in range(steps):
            velocity, _ = self._flow(st)
            st = st + (level - self.u(st)[:, 0])[:, None] * velocity
        return st

    def zero_curve_points(self, v):
        """Chart points on the level curve u = u_zero at the given v (NaN beyond the traced curve)."""
        v = np.asarray(v, float)
        order = np.argsort(self.zero_v)
        st = np.column_stack([np.interp(v, self.zero_v[order], self.zero_st[order, k]) for k in range(2)])
        st = self._to_level(st, self.u_zero)
        st[(v < self.zero_v.min()) | (v > self.zero_v.max())] = np.nan
        return st

    def length_lines(self, seeds, lengths, substeps=4):
        """Chart points at signed surface lengths along the lines of constant v through `seeds`.

        `seeds` lie on u = u_zero and `lengths` is increasing and holds 0 (the seeds); positive
        lengths go towards larger u. Each line is integrated by RK4 in surface length,
        d(s, t)/dl = g^-1 grad h / |grad h|, with `substeps` steps between consecutive lengths.
        A line stops (NaN from there on) where it leaves the chart margin or where |grad u|
        falls below the gradient floor. Returns (len(lengths), len(seeds), 2).
        """
        lengths, seeds = np.asarray(lengths, float), np.asarray(seeds, float)
        zero = np.flatnonzero(lengths == 0)
        if len(zero) != 1 or np.any(np.diff(lengths) <= 0):
            raise ValueError('lengths must be increasing and hold 0 once.')
        zero = int(zero[0])
        lo, hi = -self.config.chart_margin, 1 + self.config.chart_margin

        def velocity(st):
            up, squared = self._gradient(st)
            return up / np.sqrt(squared)[:, None]

        out = np.full((len(lengths), len(seeds), 2), np.nan)
        out[zero] = seeds
        for indices in (range(zero + 1, len(lengths)), range(zero - 1, -1, -1)):
            current, previous = seeds.copy(), 0.0
            alive = np.all(np.isfinite(seeds), axis=1)
            for i in indices:
                h = (lengths[i] - previous) / substeps
                with np.errstate(all='ignore'):
                    for _ in range(substeps):
                        k1 = velocity(current)
                        k2 = velocity(current + h / 2 * k1)
                        k3 = velocity(current + h / 2 * k2)
                        k4 = velocity(current + h * k3)
                        current = current + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                    alive &= (np.all(np.isfinite(current) & (current >= lo) & (current <= hi), axis=1)
                              & (self.gradient_norm(np.nan_to_num(current)) >= self.gradient_floor))
                if not alive.any():
                    break
                out[i, alive] = current[alive]
                previous = lengths[i]
        return out

    def _v_on_zero_curve(self, st):
        """v of chart points lying on the level curve u = u_zero (segment projection)."""
        _, nearest = self._tree.query(st)
        best_v, best_d = np.empty(len(st)), np.full(len(st), np.inf)
        for offset in (-1, 0):
            a = np.clip(nearest + offset, 0, len(self.zero_st) - 2)
            p, q = self.zero_st[a], self.zero_st[a + 1]
            along = np.clip(((st - p) * (q - p)).sum(1) / np.maximum(((q - p) ** 2).sum(1), 1e-300), 0, 1)
            distance = ((p + along[:, None] * (q - p) - st) ** 2).sum(1)
            closer = distance < best_d
            best_d[closer] = distance[closer]
            best_v[closer] = (self.zero_v[a] + along * (self.zero_v[a + 1] - self.zero_v[a]))[closer]
        return best_v, np.sqrt(best_d)

    def _integrate(self, st, du_total, steps):
        """RK4 along u's surface gradient from `st` through u changes of `du_total`.

        Returns the end points, the surface length travelled, whether the path left the chart
        [0, 1]^2 (beyond it the surface is its tangent extension), and the smallest |grad u|
        met on the way.
        """
        st = np.array(st, float)
        du = np.asarray(du_total, float) / steps
        length = np.zeros(len(st))
        left = np.zeros(len(st), bool)
        slowest = self._flow(st)[1]
        for _ in range(steps):
            k1, l1 = self._flow(st)
            k2, l2 = self._flow(st + du[:, None] / 2 * k1)
            k3, l3 = self._flow(st + du[:, None] / 2 * k2)
            k4, l4 = self._flow(st + du[:, None] * k3)
            st = st + du[:, None] / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            length += np.abs(du) / 6 * (l1 + 2 * l2 + 2 * l3 + l4)
            left |= np.any((st < 0) | (st > 1), axis=1)
            slowest = np.fmax.reduce([slowest, l1, l2, l3, l4])
        return st, length, left, 1 / slowest

    def integrate_coordinates(self, st, steps=None):
        """u, v and the signed arc_length_parallel by integrating each point's line of constant v
        back to u = u_zero. It builds the chart table; `coordinates` reads the table instead."""
        st = np.asarray(st, float)
        u = self.u(st)[:, 0]
        with np.errstate(all='ignore'):
            end, length, left, gradient = self._integrate(st, self.u_zero - u,
                                                          steps or self.config.trajectory_steps)
            end = self._to_level(end, self.u_zero)
        v, miss = np.full(len(st), np.nan), np.full(len(st), np.inf)
        finite = np.all(np.isfinite(end), axis=1)
        v[finite], miss[finite] = self._v_on_zero_curve(end[finite])
        return dict(surface_u=u, surface_v=v, arc_length_parallel=np.sign(u - self.u_zero) * length,
                    path_left_chart=left, path_u_gradient=gradient, zero_curve_miss=miss)

    def _corners(self, key, st):
        """The table `key` at the four chart-grid nodes around each point, (4, m)."""
        axis, table = self.tables['grid_axis'], self.tables[key]
        cell = [np.clip(np.searchsorted(axis, st[:, k], side='right') - 1, 0, len(axis) - 2) for k in range(2)]
        return np.stack([table[cell[0] + a, cell[1] + b] for a in (0, 1) for b in (0, 1)])

    def coordinates(self, st):
        """u, v and the signed arc_length_parallel of chart points, through the chart table.

        `path_left_chart` holds if the path from any of the four surrounding nodes left the
        chart, and `valid` if all four nodes are valid; `path_u_gradient` is the smallest
        |grad u| on those nodes' paths and at the point.
        """
        st = np.clip(np.atleast_2d(np.asarray(st, float)), 0.0, 1.0)
        u = self.u(st)[:, 0]
        gradient = self.gradient_norm(st)
        return dict(
            surface_u=u, surface_v=self._splines['surface_v'].ev(st[:, 0], st[:, 1]),
            arc_length_parallel=(u - self.u_zero) * self._splines['parallel_ratio'].ev(st[:, 0], st[:, 1]),
            path_left_chart=self._corners('grid_path_left_chart', st).any(0),
            valid=self._corners('grid_valid', st).all(0), u_gradient=gradient,
            path_u_gradient=np.minimum(self._corners('grid_path_u_gradient', st).min(0), gradient))

    def map_points(self, points, start=None):
        """Every column of MAP_COLUMNS for points in the shape's ambient space."""
        points = np.asarray(points, float)
        st, distance = closest_points(self.shape, points, start=start, config=self.config)
        coords = self.coordinates(st)
        invalid = ~coords['valid'] | (coords['path_u_gradient'] < self.gradient_floor)
        status = np.select([invalid, on_boundary(st),
                            coords['path_left_chart']], list(STATUS[1:]), default=STATUS[0])
        return dict(surface_s=st[:, 0], surface_t=st[:, 1], surface_u=coords['surface_u'],
                    surface_v=coords['surface_v'], arc_length_parallel=coords['arc_length_parallel'],
                    arc_length_orthogonal=coords['surface_v'], surface_distance=np.sqrt(distance),
                    u_gradient=coords['u_gradient'], path_u_gradient=coords['path_u_gradient'],
                    coordinate_status=status)

    def orthogonality_check(self, samples=2000, seed=0):
        """|cos| between the u and v directions at random chart points (0 = orthogonal)."""
        st = np.random.default_rng(seed).uniform(0.05, 0.95, (samples, 2))
        velocity, _ = self._flow(st)
        u_jet, shape_jet = self.u.jet(st, order=1), self.shape.jet(st, order=1)
        along_u = velocity[:, :1] * shape_jet[(1, 0)] + velocity[:, 1:] * shape_jet[(0, 1)]
        level = np.column_stack([-u_jet[(0, 1)][:, 0], u_jet[(1, 0)][:, 0]])  # tangent of u = const
        along_v = level[:, :1] * shape_jet[(1, 0)] + level[:, 1:] * shape_jet[(0, 1)]
        cosine = np.abs((along_u * along_v).sum(1)) / (np.linalg.norm(along_u, axis=1)
                                                        * np.linalg.norm(along_v, axis=1))
        return float(cosine.max())

    def table_diagnostics(self):
        """The gradient check over the chart, and how well the two cached tables agree.

        The net's nodes inside the chart are mapped through the chart table and should return
        the net's own u, v and arc_length_parallel.
        """
        axis = self.tables['grid_axis']
        gradient = self.gradient_norm(np.stack(np.meshgrid(axis, axis, indexing='ij'), axis=-1).reshape(-1, 2))
        net = self.tables['net_st'].reshape(-1, 2)
        inside = np.flatnonzero(np.all(np.isfinite(net) & (net >= 0) & (net <= 1), axis=1))
        mapped = self.coordinates(net[inside])
        inside, mapped = inside[mapped['valid']], {k: v[mapped['valid']] for k, v in mapped.items()}
        u_net, v_net = np.meshgrid(self.tables['net_u'], self.tables['net_v'], indexing='ij')
        length = self.tables['net_arc_length_parallel'].ravel()[inside]
        valid = self.tables['grid_valid']
        return dict(
            u_gradient_floor=self.gradient_floor,
            chart_min_u_gradient=float(gradient.min()),
            chart_share_below_floor=float((gradient < self.gradient_floor).mean()),
            chart_min_path_u_gradient=float(self.tables['grid_path_u_gradient'][valid].min()),
            chart_table_invalid_share=float(1 - valid.mean()),
            net_nodes_checked=int(len(inside)),
            round_trip_max_abs_u=float(np.abs(mapped['surface_u'] - u_net.ravel()[inside]).max()),
            round_trip_max_abs_v=float(np.abs(mapped['surface_v'] - v_net.ravel()[inside]).max()),
            round_trip_max_abs_parallel=float(np.abs(mapped['arc_length_parallel'] - length).max()))

    # ---- persistence

    def arrays(self):
        return dict(**self.shape.arrays('shape'), **self.u.arrays('u_map'),
                    u_zero=np.asarray(self.u_zero), zero_curve_st=self.zero_st, zero_curve_v=self.zero_v,
                    u_gradient_floor=np.asarray(self.gradient_floor), **self.tables)

    @classmethod
    def from_arrays(cls, arrays, config):
        config = config if isinstance(config, SurfaceConfig) else SurfaceConfig(
            **{k: tuple(v) if isinstance(v, list) else v for k, v in config.items()})
        return cls(shape_from_arrays(arrays, 'shape'), TensorSpline.from_arrays(arrays, 'u_map'),
                   float(arrays['u_zero']), arrays['zero_curve_st'], arrays['zero_curve_v'], config,
                   float(arrays['u_gradient_floor']), {key: arrays[key] for key in TABLE_KEYS})

    def config_dict(self):
        return asdict(self.config)
