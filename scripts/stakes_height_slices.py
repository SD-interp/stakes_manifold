"""PLS2=f(PLS1, PLS3), nearest-height slices and signed longitudinal length.

These are slice coordinates, not orthogonal geodesic coordinates. Polynomial
coefficients are fitted only once; coverage heights do not participate in fitting.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import os
import time

import numpy as np
from numpy.polynomial import polynomial as poly
from scipy.integrate import cumulative_trapezoid


MAPPING_COLUMNS = (
    'surface_u', 'surface_v', 'arc_length_parallel', 'arc_length_orthogonal',
    'surface_projection_residual', 'slice_height_error', 'slice_index',
    'coordinate_status', 'surface_extended', 'height_extrapolated',
)
LEGACY_COLUMNS = ('reference_parameter', 'coordinate_mapping_residual', 'coordinate_candidates')


def centroid_plane_rotation(samples, ridge_relative=1e-6):
    """Rotate matched centroid trajectories into a shared plane (equal files).

    The normal maximizes between-file separation relative to within-curve
    variation. This is a best-fit orientation, not a flattening of the curves.
    Returned rows are orthonormal axes; apply as scores @ rotation.T.
    """
    from scipy.linalg import eigh

    samples = np.asarray(samples, dtype=float)
    if (samples.ndim != 3 or samples.shape[2] != 3 or samples.shape[1] < 2
            or samples.shape[0] < 1 or not np.isfinite(samples).all()
            or not np.isfinite(ridge_relative) or ridge_relative <= 0):
        raise ValueError('Expected finite file-by-progress-by-3 centroid trajectories.')
    within = samples - samples.mean(axis=1, keepdims=True)
    between = samples - samples.mean(axis=0, keepdims=True)
    within = within.reshape(-1, 3)
    between = between.reshape(-1, 3)
    sw = within.T @ within / len(within)
    sb = between.T @ between / len(between)
    mass = float(np.trace(sw))
    if mass <= 0:
        raise ValueError('Centroid trajectories have no longitudinal variation.')
    ridge = ridge_relative * mass / 3
    if np.trace(sb) <= np.finfo(float).eps * mass:
        normal = np.linalg.eigh(sw)[1][:, 0]
    else:
        normal = eigh(sb, sw + ridge*np.eye(3))[1][:, -1]
    normal /= np.linalg.norm(normal)
    sign_axis = 2 if abs(normal[2]) > 1e-12 else int(np.argmax(np.abs(normal)))
    normal *= 1 if normal[sign_axis] >= 0 else -1
    plane = np.linalg.svd(normal[None, :], full_matrices=True)[2][1:]
    first = np.linalg.eigh(plane @ sw @ plane.T)[1][:, -1] @ plane
    sign_axis = 0 if abs(first[0]) > 1e-12 else int(np.argmax(np.abs(first)))
    first *= 1 if first[sign_axis] >= 0 else -1
    rotation = np.stack([first, np.cross(normal, first), normal])
    return rotation, dict(
        within_curve_variance_in_plane=float(1 - normal @ sw @ normal / mass),
        separation_variance_along_normal=(float(normal @ sb @ normal / np.trace(sb))
                                          if np.trace(sb) > 0 else 0.0),
        ridge_relative=float(ridge_relative),
    )


@dataclass(frozen=True)
class SliceConfig:
    n_slices: int = 5000
    arc_relative_tolerance: float = 1e-4
    max_arc_nodes: int = 262145

    def __post_init__(self):
        if type(self.n_slices) is not int or self.n_slices < 2:
            raise ValueError('n_slices must be an integer >= 2.')
        if not 0 < self.arc_relative_tolerance < 1:
            raise ValueError('Invalid arc tolerance.')
        if type(self.max_arc_nodes) is not int or self.max_arc_nodes < 3:
            raise ValueError('Invalid arc node limit.')


class SliceSurface:
    """PLS2 = f(PLS1, PLS3) cut into fixed-PLS3 slices.

    `origin` places each slice's zero of `arc_length_parallel`: the PLS1 of that zero as
    polynomial coefficients in slice height, lowest degree first. A scalar gives every slice
    the same zero PLS1.
    """

    def __init__(self, center, scale, coefficients, x_bounds, z_bounds, origin, config=None):
        self.center = np.asarray(center, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.x_bounds = np.asarray(x_bounds, dtype=float)
        self.z_bounds = np.asarray(z_bounds, dtype=float)
        self.origin = np.atleast_1d(np.asarray(origin, dtype=float))
        self.config = config or SliceConfig()
        if (self.center.shape != (2,) or self.scale.shape != (2,)
                or self.coefficients.shape != (4, 4) or self.x_bounds.shape != (2,)
                or self.z_bounds.shape != (2,) or np.any(self.scale <= 0)
                or self.origin.ndim != 1 or not len(self.origin)
                or self.x_bounds[0] >= self.x_bounds[1] or self.z_bounds[0] > self.z_bounds[1]
                or not all(np.isfinite(a).all() for a in
                           (self.center, self.scale, self.coefficients, self.x_bounds,
                            self.z_bounds, self.origin))):
            raise ValueError('Invalid slice surface.')
        self.t_bounds = (self.x_bounds - self.center[0]) / self.scale[0]

    @classmethod
    def fit(cls, points, origin, config=None):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 10 or not np.isfinite(points).all():
            raise ValueError('Fit requires at least ten finite PLS1-3 rows.')
        predictors = points[:, [0, 2]]
        center = predictors.mean(axis=0)
        scale = predictors.std(axis=0)
        scale[scale == 0] = 1.0
        scaled = (predictors - center) / scale
        powers = [(i, d-i) for d in range(4) for i in range(d+1)]
        design = np.column_stack([scaled[:, 0]**i * scaled[:, 1]**j for i, j in powers])
        values, _, rank, singular = np.linalg.lstsq(design, points[:, 1], rcond=None)
        coefficients = np.zeros((4, 4))
        for value, (i, j) in zip(values, powers):
            coefficients[i, j] = value
        surface = cls(center, scale, coefficients, [points[:, 0].min(), points[:, 0].max()],
                      [points[:, 2].min(), points[:, 2].max()], origin, config)
        surface.fit_rank = int(rank)
        surface.fit_condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float('inf')
        return surface

    def origin_u(self, heights):
        """PLS1 of the zero of `arc_length_parallel` on the slices at `heights`."""
        return poly.polyval(np.asarray(heights, dtype=float), self.origin)

    def curve_coefficients(self, heights):
        z = (np.asarray(heights) - self.center[1]) / self.scale[1]
        return np.stack([poly.polyval(z, row) for row in self.coefficients], axis=-1)

    def evaluate(self, uv):
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        t = (uv[:, 0] - self.center[0]) / self.scale[0]
        clipped = np.clip(t, *self.t_bounds)
        c = self.curve_coefficients(uv[:, 1])
        y = sum(c[:, k] * clipped**k for k in range(4))
        slope = sum(k*c[:, k] * clipped**(k-1) for k in range(1, 4))
        y += (t-clipped)*slope
        return np.column_stack([uv[:, 0], y, uv[:, 1]])

    def build_cache(self, heights):
        heights = np.asarray(heights, dtype=float)
        if not heights.size or not np.isfinite(heights).all():
            raise ValueError('Coverage requires finite heights.')
        self.heights = np.linspace(heights.min(), heights.max(), self.config.n_slices)
        self.curves = self.curve_coefficients(self.heights)
        lo, hi = self.t_bounds
        width, sx = hi-lo, self.scale[0]
        tables, offsets = [], [0]
        for c in self.curves:
            # Rigorous trapezoidal integration + linear interpolation error bound:
            # speed=sqrt(sx**2+y'**2), |speed'|<=|y''|,
            # |speed''|<=y''**2/sx+|y'''|. Target <= tolerance*PLS1 width,
            # hence also <= tolerance*actual finite-domain slice length.
            second = max(abs(2*c[2]+6*c[3]*lo), abs(2*c[2]+6*c[3]*hi))
            curvature_bound = second**2/sx + abs(6*c[3])
            bound = width*curvature_bound/12 + second/8
            # Reserve half the tolerance for subtracting the origin lookup.
            tolerance = self.config.arc_relative_tolerance * width*sx / 2
            intervals = max(16, int(np.ceil(width*np.sqrt(bound/tolerance))))
            if intervals+1 > self.config.max_arc_nodes:
                raise ValueError('Arc table exceeds node limit; increase max_arc_nodes.')
            t = np.linspace(lo, hi, intervals+1)
            speed = np.hypot(sx, poly.polyval(t, poly.polyder(c)))
            table = cumulative_trapezoid(speed, t, initial=0)
            tables.append(table)
            offsets.append(offsets[-1]+len(table))
        self.arc_offsets = np.asarray(offsets, dtype=np.int64)
        self.arc_values = np.concatenate(tables)
        if not np.isfinite(self.arc_values).all():
            raise ValueError('Nonfinite slice cache.')
        return self

    def nearest_slices(self, z):
        if self.heights[-1] == self.heights[0]:
            return np.zeros(np.shape(z), dtype=np.int64)
        z = np.asarray(z)
        right = np.clip(np.searchsorted(self.heights, z, side='left'), 1, len(self.heights)-1)
        left = right-1
        return np.where(np.abs(z-self.heights[left]) <= np.abs(z-self.heights[right]), left, right)

    def _arc(self, indices, t):
        lo, hi = self.t_bounds
        start, stop = self.arc_offsets[indices], self.arc_offsets[indices+1]
        q = (np.clip(t, lo, hi)-lo)/(hi-lo)*(stop-start-1)
        k = np.minimum(np.floor(q).astype(int), stop-start-2)
        a = self.arc_values[start+k]
        value = a+(q-k)*(self.arc_values[start+k+1]-a)
        clipped = np.clip(t, lo, hi)
        c = self.curves[indices]
        slope = c[:, 1]+2*c[:, 2]*clipped+3*c[:, 3]*clipped**2
        return value + (t-clipped)*np.hypot(self.scale[0], slope)

    def _project(self, points, indices):
        c = self.curves[indices]
        sx, cx = self.scale[0], self.center[0]
        lo, hi = self.t_bounds
        # Stationary squared-distance polynomial, degree at most five.
        shifted = c.copy()
        shifted[:, 0] -= points[:, 1]
        derivative = c[:, 1:]*np.arange(1, 4)
        polynomial = np.zeros((len(points), 6))
        for i in range(4):
            for j in range(3):
                polynomial[:, i+j] += shifted[:, i]*derivative[:, j]
        polynomial[:, 0] += sx*(cx-points[:, 0])
        polynomial[:, 1] += sx*sx
        candidates = np.full((len(points), 9), lo)
        candidates[:, 1] = hi
        for column, end, side in [(2, lo, -1), (3, hi, 1)]:
            y = sum(c[:, k]*end**k for k in range(4))
            dy = c[:, 1]+2*c[:, 2]*end+3*c[:, 3]*end**2
            delta = (sx*(points[:, 0]-(cx+sx*end))+dy*(points[:, 1]-y))/(sx*sx+dy*dy)
            candidates[:, column] = end+side*np.maximum(side*delta, 0)
        fallback = np.zeros(len(points), dtype=bool)
        # Batched companion eigenvalues avoid one Python optimizer per row.
        degree = np.max(np.where(polynomial != 0, np.arange(6), 0), axis=1)
        for d in range(1, 6):
            rows = np.flatnonzero(degree == d)
            if not len(rows):
                continue
            matrix = np.zeros((len(rows), d, d))
            if d > 1:
                matrix[:, np.arange(1, d), np.arange(d-1)] = 1
            matrix[:, :, -1] = -polynomial[rows, :d]/polynomial[rows, d, None]
            try:
                roots = np.linalg.eigvals(matrix)
            except np.linalg.LinAlgError:
                # Isolate failing rows rather than discard the whole batch.
                roots = np.full((len(rows), d), np.nan+0j)
                for j, mat in enumerate(matrix):
                    try:
                        roots[j] = np.linalg.eigvals(mat)
                    except np.linalg.LinAlgError:
                        fallback[rows[j]] = True
            real = roots.real
            valid = (np.abs(roots.imag) <= 1e-8*(1+np.abs(real))) & (real >= lo) & (real <= hi)
            candidates[rows, 4:4+d] = np.where(valid, real, lo)
        for row in np.flatnonzero(fallback):
            grid = np.linspace(lo, hi, 1025)
            distance = (cx+sx*grid-points[row, 0])**2+(poly.polyval(grid, c[row])-points[row, 1])**2
            candidates[row, 4] = grid[np.argmin(distance)]
        candidates.sort(axis=1)  # Exact distance ties select the smaller PLS1.
        clipped = np.clip(candidates, lo, hi)
        y = sum(c[:, k, None]*clipped**k for k in range(4))
        dy = sum(k*c[:, k, None]*clipped**(k-1) for k in range(1, 4))
        y += (candidates-clipped)*dy
        distances = (cx+sx*candidates-points[:, 0, None])**2+(y-points[:, 1, None])**2
        selected = np.argmin(distances, axis=1)
        return candidates[np.arange(len(points)), selected], fallback

    def map_points(self, points, batch_rows=4096, checkpoint_dir=None, resume=True, progress_seconds=5):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('Mapping requires finite PLS1-3 inputs; export aborted.')
        if type(batch_rows) is not int or batch_rows < 1:
            raise ValueError('batch_rows must be positive.')
        if not hasattr(self, 'heights'):
            raise ValueError('Build the slice cache before mapping.')
        folder = None
        if checkpoint_dir is not None:
            digest = hashlib.sha256(Path(__file__).read_bytes())
            digest.update(json.dumps(asdict(self.config), sort_keys=True).encode())
            for key, value in sorted(self.arrays().items()):
                digest.update(key.encode())
                digest.update(np.ascontiguousarray(value).tobytes())
            digest.update(str((points.shape, batch_rows)).encode())
            digest.update(np.ascontiguousarray(points).tobytes())
            folder = Path(checkpoint_dir)/digest.hexdigest()
            folder.mkdir(parents=True, exist_ok=True)
        output = {k: [] for k in MAPPING_COLUMNS}
        last = time.monotonic()
        for start in range(0, len(points), batch_rows):
            batch = points[start:start+batch_rows]
            path = folder/f'{start:09d}.npz' if folder else None
            if resume and path is not None and path.exists():
                with np.load(path, allow_pickle=False) as archive:
                    result = {key: archive[key] for key in MAPPING_COLUMNS}
                if any(v.shape != (len(batch),) for v in result.values()):
                    raise ValueError('Invalid slice checkpoint shape.')
            else:
                indices = self.nearest_slices(batch[:, 2])
                t, fallback = self._project(batch, indices)
                x = self.center[0]+self.scale[0]*t
                z = self.heights[indices]
                projected = self.evaluate(np.column_stack([x, z]))
                origin = (self.origin_u(z)-self.center[0])/self.scale[0]
                result = dict(surface_u=x, surface_v=z,
                    arc_length_parallel=self._arc(indices, t)-self._arc(indices, origin),
                    arc_length_orthogonal=z.copy(),
                    surface_projection_residual=np.linalg.norm(projected-batch, axis=1),
                    slice_height_error=z-batch[:, 2], slice_index=indices,
                    coordinate_status=np.where(fallback, 'projection_fallback', 'ok'),
                    surface_extended=(t < self.t_bounds[0]) | (t > self.t_bounds[1]),
                    height_extrapolated=(z < self.z_bounds[0]) | (z > self.z_bounds[1]))
                self._validate_result(result)
                if path is not None:
                    temporary = path.with_suffix('.tmp')
                    with temporary.open('wb') as stream:
                        np.savez_compressed(stream, **result)
                    os.replace(temporary, path)
            self._validate_result(result)
            for key in output:
                output[key].append(result[key])
            if progress_seconds is not None and time.monotonic()-last >= progress_seconds:
                print(f'Slice coordinates: {start+len(batch):,}/{len(points):,}', flush=True)
                last = time.monotonic()
        return {k: np.concatenate(v) if v else np.array([], dtype=str if k == 'coordinate_status' else float)
                for k, v in output.items()}

    @staticmethod
    def _validate_result(result):
        if not all(np.isfinite(v).all() for k, v in result.items() if k != 'coordinate_status'):
            raise ValueError('Nonfinite mapping result; export aborted.')

    def map_point(self, point):
        return {key: values[0] for key, values in self.map_points([point], progress_seconds=None).items()}

    def arrays(self):
        return {'slice_'+key: np.asarray(getattr(self, key)) for key in (
            'center', 'scale', 'coefficients', 'x_bounds', 'z_bounds', 'origin',
            'heights', 'curves', 'arc_offsets', 'arc_values')}

    @classmethod
    def from_arrays(cls, arrays, config):
        model = cls(*(arrays['slice_'+key] for key in
                      ('center', 'scale', 'coefficients', 'x_bounds', 'z_bounds', 'origin')),
                    config=SliceConfig(**config))
        for key in ('heights', 'curves', 'arc_offsets', 'arc_values'):
            setattr(model, key, arrays['slice_'+key])
        if (model.heights.shape != (model.config.n_slices,)
                or model.curves.shape != (model.config.n_slices, 4)
                or model.arc_offsets.shape != (model.config.n_slices+1,)
                or model.arc_offsets[0] != 0 or model.arc_offsets[-1] != len(model.arc_values)
                or np.any(np.diff(model.arc_offsets) < 2)
                or np.any(np.diff(model.heights) < 0)
                or not all(np.isfinite(a).all() for a in model.arrays().values())):
            raise ValueError('Invalid saved slice cache.')
        return model


def validate_slice_mapping(model, points):
    """Independent dense-bracket minimization and quadrature on a small sample.

    This is a numerical validation, not a global mathematical certificate.
    Raises on failed accuracy checks so a notebook full run cannot pass the gate.
    """
    from scipy.integrate import quad
    from scipy.optimize import minimize_scalar

    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        raise ValueError('Validation requires at least one point.')
    started = time.perf_counter()
    result = model.map_points(points, progress_seconds=None)
    elapsed = time.perf_counter()-started
    lo, hi = model.t_bounds
    sx, cx = model.scale[0], model.center[0]
    arc_errors, projection_errors = [], []
    for i, point in enumerate(points):
        c = model.curves[result['slice_index'][i]]
        origin = (model.origin_u(model.heights[result['slice_index'][i]])-cx)/sx
        derivative = poly.polyder(c)

        def y(t):
            clipped = np.clip(t, lo, hi)
            return poly.polyval(clipped, c)+(t-clipped)*poly.polyval(clipped, derivative)

        def objective(t):
            return (cx+sx*t-point[0])**2+(y(t)-point[1])**2

        grid = np.linspace(lo, hi, 2049)
        distance = objective(grid)
        candidates = [lo, hi]
        minima = np.flatnonzero((distance[1:-1] <= distance[:-2]) &
                               (distance[1:-1] <= distance[2:]))+1
        for j in minima:
            solution = minimize_scalar(objective, bounds=(grid[j-1], grid[j+1]),
                                       method='bounded', options={'xatol': 1e-12})
            candidates.append(solution.x)
        # Independent bounded minimization of each ray, using an interval that
        # contains its nearest point by Cauchy-Schwarz and unit-speed geometry.
        for end, side in ((lo, -1), (hi, 1)):
            radius = np.sqrt(objective(end))/sx+1
            bounds = sorted((end, end+side*radius))
            solution = minimize_scalar(objective, bounds=bounds, method='bounded',
                                       options={'xatol': 1e-12})
            candidates.append(solution.x)
        reference_distance = np.sqrt(min(objective(t) for t in candidates))
        projected_t = (result['surface_u'][i]-cx)/sx
        actual_distance = np.sqrt(objective(projected_t))
        projection_error = abs(actual_distance-reference_distance)
        projection_errors.append(projection_error)
        if projection_error > 1e-4*max(sx*(hi-lo), reference_distance):
            raise AssertionError(f'Slice projection validation failed at sample {i}: {projection_error}')

        def integral_to(t):
            clipped = float(np.clip(t, lo, hi))
            inside = quad(lambda q: np.hypot(sx, poly.polyval(q, derivative)),
                          lo, clipped, epsabs=1e-10, epsrel=1e-11)[0]
            return inside+(t-clipped)*np.hypot(sx, poly.polyval(clipped, derivative))

        reference_arc = integral_to(projected_t)-integral_to(origin)
        total_length = integral_to(hi)
        relative_error = abs(result['arc_length_parallel'][i]-reference_arc)/total_length
        arc_errors.append(relative_error)
        if relative_error > model.config.arc_relative_tolerance+1e-10:
            raise AssertionError(f'Slice arc validation failed at sample {i}: {relative_error}')
    in_grid = (points[:, 2] >= model.heights[0]) & (points[:, 2] <= model.heights[-1])
    spacing = (model.heights[-1]-model.heights[0])/(len(model.heights)-1)
    height_error = np.abs(result['slice_height_error'])
    if np.any(height_error[in_grid] > spacing/2+1e-12*max(1, np.max(np.abs(model.heights)))):
        raise AssertionError('Nearest-height quantization bound failed.')
    fallback_count = int(np.count_nonzero(result['coordinate_status'] != 'ok'))
    if fallback_count:
        raise AssertionError(f'Validation sample has {fallback_count} projection fallbacks.')
    return dict(rows=len(points), mapping_seconds=elapsed,
                rows_per_second=len(points)/max(elapsed, 1e-12),
                max_projection_distance_error=float(max(projection_errors)),
                max_arc_error_relative_to_slice_length=float(max(arc_errors)),
                max_height_error=float(height_error.max()),
                projection_fallback_rows=fallback_count,
                extended_rows=int(result['surface_extended'].sum()),
                height_extrapolated_rows=int(result['height_extrapolated'].sum()),
                cache_bytes=int(sum(a.nbytes for a in model.arrays().values())))
