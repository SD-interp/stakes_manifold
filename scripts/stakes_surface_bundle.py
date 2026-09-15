from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import BSpline, make_lsq_spline

def fit_centroid_spline(points, interior_knots=1, degree=3):
    """The PLS notebook's chord-parameterized, cubic least-squares spline."""
    points = np.asarray(points, dtype=np.float64)
    if (degree != 3 or points.ndim != 2 or len(points) < 5
            or not np.isfinite(points).all()):
        raise ValueError("A cubic centroid spline needs at least five finite points.")
    if (isinstance(interior_knots, bool) or not isinstance(interior_knots, int)
            or interior_knots < 0):
        raise ValueError("interior_knots must be a nonnegative integer.")
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if (steps <= 0).any():
        raise ValueError("Consecutive centroid positions coincide.")
    chord = np.r_[0.0, np.cumsum(steps)]
    parameters = chord / chord[-1]
    count = min(interior_knots, len(points) - degree - 2)
    interior = np.quantile(parameters, np.arange(1, count + 1) / (count + 1))
    knots = np.r_[np.zeros(degree + 1), interior, np.ones(degree + 1)]
    spline = make_lsq_spline(parameters, points, knots, k=degree)
    return spline, parameters


def spline_arrays(prefix, spline):
    return {f"{prefix}_knots": spline.t, f"{prefix}_coefficients": spline.c,
            f"{prefix}_degree": np.asarray(spline.k)}


def spline_from_arrays(arrays, prefix):
    return BSpline(arrays[f"{prefix}_knots"], arrays[f"{prefix}_coefficients"],
                   int(arrays[f"{prefix}_degree"]), extrapolate=False)


def project_saved_bcpc(activations, arrays, metadata):
    """Reproduce BCPC scores and the original closest-spline target, without refitting.

    Pass batches from the saved model/layer/position. The root search, quadrature,
    and exact-distance tie orientation match the source BCPC notebook.
    """
    from numpy.polynomial import polynomial as poly

    activations = np.asarray(activations, dtype=np.float64)
    if (activations.ndim != 2 or activations.shape[1] != len(arrays["bcpc_mean"])
            or not np.isfinite(activations).all()):
        raise ValueError("Expected a finite activation batch with the saved feature width.")
    scores = (activations - arrays["bcpc_mean"]) @ arrays["bcpc_components"]
    curve = spline_from_arrays(arrays, "bcpc")
    if curve.k != 3:
        raise ValueError("The saved BCPC projection requires a cubic spline.")
    reverse = metadata["bcpc_settings"]["reverse_arc_length"]
    edges = np.unique(curve.t[curve.k:-curve.k])

    def speed(t):
        return float(np.linalg.norm(curve(t, 1)))

    lengths = np.array([quad(speed, a, b, epsabs=1e-8, epsrel=1e-8)[0]
                        for a, b in zip(edges[:-1], edges[1:])])
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    spans = []
    for left, right in zip(edges[:-1], edges[1:]):
        width = right - left
        coefficients = np.stack([curve(left), curve(left, 1) * width,
                                 curve(left, 2) * width ** 2 / 2,
                                 curve(left, 3) * width ** 3 / 6])
        spans.append((left, width, coefficients))
    parameters, arc_lengths = [], []
    for point in scores:
        candidates = list(edges)
        for left, width, coefficients in spans:
            stationary = np.zeros(6)
            for axis in range(scores.shape[1]):
                residual = coefficients[:, axis].copy()
                residual[0] -= point[axis]
                term = poly.polymul(residual, poly.polyder(coefficients[:, axis]))
                stationary[:len(term)] += term
            for root in poly.polyroots(stationary):
                if abs(root.imag) <= 1e-8 and -1e-10 <= root.real <= 1 + 1e-10:
                    candidates.append(left + width * np.clip(root.real, 0, 1))
        candidates = np.unique(candidates)
        if reverse:
            candidates = candidates[::-1]
        t = float(candidates[np.argmin(np.sum((curve(candidates) - point) ** 2, axis=1))])
        span = min(np.searchsorted(edges, t, side="right") - 1, len(lengths) - 1)
        length = cumulative[span] + quad(speed, edges[span], t, epsabs=1e-8, epsrel=1e-8)[0]
        arc_lengths.append(float(np.clip(cumulative[-1] - length if reverse else length,
                                         0, cumulative[-1])))
        parameters.append(t)
    closest = curve(np.asarray(parameters))
    return dict(bcpc_scores=scores, bcpc_arc_length=np.asarray(arc_lengths),
                spline_parameter=np.asarray(parameters), closest_bcpc_scores=closest,
                distance_to_spline=np.linalg.norm(scores - closest, axis=1))


def load_surface_bundle(directory):
    """Load saved transforms and geometry without fitting or pickle.

    Returns (arrays, metadata, coordinates). Raw activation transforms are
    (X - arrays['bcpc_mean']) @ arrays['bcpc_components'] and
    (X - arrays['pls_mean']) @ arrays['pls_rotations']. Feed each PLS row to
    coordinates.map_point or map_points. Only the version-2 height-slice
    bundle produced by this pipeline is supported.
    """
    directory = Path(directory)
    metadata = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 2:
        raise ValueError("Unsupported stakes surface model version.")
    if hashlib.sha256((directory / "model.npz").read_bytes()).hexdigest() != metadata.get("model_sha256"):
        raise ValueError("Model archive does not match its metadata checksum.")
    with np.load(directory / "model.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if metadata['schema_version'] == 2:
        from .stakes_height_slices import SliceSurface
        if metadata.get('geometry_type') != 'pls2_cubic_height_slices':
            raise ValueError('Unsupported version-2 surface geometry.')
        return arrays, metadata, SliceSurface.from_arrays(arrays, metadata['slice_config'])
