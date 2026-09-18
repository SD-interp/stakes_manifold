"""PLS1-3 height-slice surface on the full-fit BCPC arc length.

This replaces `stakes_surface_pipeline.py`, which fitted its own BCPC. Here the target and
the stakes classes come from the saved BCPC bundle in `artifacts/<model>/bcpc/`
(`bcpc_bundle.py`). The surface geometry and its weighting are unchanged:

1. every horizon-free row outside `cv.EXCLUDED_TEMPLATES` is joined to `bcpc/rows.parquet`
   on (source_file, source_row), whose saved arc lengths are taken as given. With
   `SurfaceSettings.include_excluded_templates`, the excluded templates' rows are added,
   joined to `bcpc/excluded_rows.parquet`: the BCPC never saw them, but the surface fits them
2. 16 centred, unscaled PLS components against `bcpc_arc_length`
3. per-template curves: each template's PLS1-3 as a cubic spline of `bcpc_arc_length`,
   sampled over the arc-length range every template covers, and PLS1-3 rotated into the
   curves' best-fit shared plane
4. the cubic graph PLS2 = f(PLS1, PLS3), and a zero line: for each PLS3 height, the PLS1 at
   which rows reach `refined_stakes` = 1 (`bcpc_arc_length` = the bundle's `arc_zero`)
5. 5,000 fixed-PLS3 slices, validated on a sample, then every row mapped onto them

`arc_length_parallel` is the signed length along the row's slice from where the slice meets
the zero line, so each slice has its own zero.
`arc_length_orthogonal` is the snapped PLS3 height (a legacy name; it is not a geodesic
length). The weighting is the old pipeline's: PLS gives every template file equal total
weight, each template's curve counts its rows equally, the plane rotation and the zero
line count every template equally, and the surface fit counts every row equally. These are in-sample descriptive fits, not held-out
evaluations.

Output, in `artifacts/<model>/pls/` beside the BCPC bundle:

- `pls/`            model.npz, model.json, rows.parquet (with each row's `prompt` text) and plots/
- `pls/inference/`  one CSV per cached inference corpus, with the surface coordinates and
                    the bundle's `bcpc_arc_length` and `refined_stakes`
"""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline, make_lsq_spline

from . import bcpc_bundle
from . import bcpc_cross_validation as cv
from . import cache_activations, inference_datasets, inference_projection
from .pipeline_config import ROOT
from .pls_fit import WeightedPls
from .stakes_height_slices import (
    MAPPING_COLUMNS, SliceConfig, SliceSurface, centroid_plane_rotation, validate_slice_mapping,
)
from .stakes_surface_bundle import spline_arrays

SCHEMA_VERSION = 2  # 2: template curves on bcpc_arc_length; a zero per slice (slice_origin)
GEOMETRY_TYPE = 'bcpc_pls2_cubic_height_slices'
TARGET = 'bcpc_arc_length'
STAKES = 'stakes'
BCPC_COLUMNS = ['bcpc_arc_length', 'refined_stakes']
SURFACE_COLUMNS = ['PLS1', 'PLS2', 'PLS3']
# PLS1-3 columns hold rotated scores; the plots name the rotated axes explicitly.
ROTATED_AXIS_TITLES = ('rotated PLS1 (along curves)', 'rotated PLS2 (in plane)',
                       'rotated PLS3 (plane normal)')
COORDINATE_COLUMNS = ['surface_u', 'surface_v', 'arc_length_parallel',
                      'arc_length_orthogonal', 'surface_projection_residual']
CODE_FILES = ('scripts/bcpc_surface.py', 'scripts/stakes_height_slices.py', 'scripts/pls_fit.py',
              'scripts/inference_projection.py')
WEIGHTING = dict(  # the old surface pipeline's
    pls='equal total weight per template file, split equally among its rows',
    template_curves='equal weight per row within its template',
    rotation='equal weight per template',
    surface='equal weight per row',
    zero_line='equal total weight per template file, split equally among its rows')

# Plot palette on a light chart surface. Stakes classes are ordered, so they take
# the blue ordinal ramp (lightest step still >= 2:1 on the surface), never cycled hues.
STAKES_RAMP = ('#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6',
               '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b')
SEQUENTIAL_SCALE = [[0.0, '#cde2fb'], [0.5, '#2a78d6'], [1.0, '#0d366b']]
# The slices' zero line is red, so it stands apart from the blue rows and the grey/black lines.
ZERO_LINE_COLOR = '#d62728'
SURFACE_COLOR = '#fcfcfb'
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
GRID = '#e1e0d9'
BASELINE = '#c3c2b7'


@dataclass(frozen=True)
class SurfaceSettings:
    pls_components: int = 16
    spline_degree: int = 3
    spline_interior_knots: int = 1
    arc_samples: int = 401
    # The rotation samples every template's curve between the largest of the templates' low
    # arc-length quantiles and the smallest of their high ones.
    curve_range_quantiles: tuple = (0.01, 0.99)
    plane_rotation_ridge: float = 1e-6
    slices: SliceConfig = field(default_factory=lambda: SliceConfig(
        n_slices=5000, arc_relative_tolerance=1e-4, max_arc_nodes=262145))
    chunk_rows: int = 1024
    mapping_batch_rows: int = 4096
    validation_rows: int = 32
    reload_check_rows: int = 256
    seed: int = 42
    # Also fit the rows of cv.EXCLUDED_TEMPLATES, with the arc lengths the BCPC bundle saved
    # for them in excluded_rows.parquet (scored by the full fit, which never saw them).
    include_excluded_templates: bool = False

    def __post_init__(self):
        if self.pls_components < 3:
            raise ValueError('The slice surface needs at least PLS1-3.')
        low, high = self.curve_range_quantiles
        if not 0 <= low < high <= 1:
            raise ValueError('curve_range_quantiles must satisfy 0 <= low < high <= 1.')


@dataclass
class SurfaceFit:
    config: object
    settings: SurfaceSettings
    rows: pd.DataFrame          # every horizon-free row: bundle columns, PLS1-16, coordinates
    pls: WeightedPls
    slice_rotation: np.ndarray  # rows are the rotated PLS1-3 axes in unrotated PLS1-3
    pls_rotations: np.ndarray   # raw activations -> rotated PLS scores, (d, components)
    centroids: pd.DataFrame     # per-template class means in rotated PLS1-3 (plots only)
    file_splines: dict          # {source_file: BSpline of bcpc_arc_length} in rotated PLS1-3
    surface: SliceSurface       # fitted graph with its slice cache
    diagnostics: dict
    bcpc_metadata: dict
    check_activations: np.ndarray  # a few raw rows kept for the reload check in `save`
    check_positions: np.ndarray


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, writer):
    """Write through a temporary file so a reader never sees a partial file."""
    temporary = path.with_name(path.name + '.tmp')
    writer(temporary)
    os.replace(temporary, path)


def surface_dir(config):
    return config.run_dir / 'pls'


def inference_dir(config):
    return surface_dir(config) / 'inference'


def has_bcpc_bundle(run_dir):
    return (Path(run_dir) / 'bcpc' / 'rows.parquet').is_file()


def _bundle_link(config):
    """Checksums tying the surface to the exact BCPC bundle it was fitted on."""
    return {name: _sha256(config.bcpc_dir / name) for name in ('model.json', 'rows.parquet')}


def _check_bundle_link(config, metadata):
    if _bundle_link(config) != metadata['bcpc_bundle']:
        raise ValueError(f'{config.bcpc_dir} changed after the surface in {surface_dir(config)} '
                         'was fitted; refit the surface.')


def _join_bundle_rows(data, path):
    """`data`'s rows joined to the bundle rows saved at `path`, which must cover them exactly.

    The prompt text comes from the caches, so a bundle saved before it carried `prompt`
    still yields it; a bundle that does carry it must agree.
    """
    saved = pd.read_parquet(path)
    keys = ['source_file', 'source_row']
    rows = data.rows[keys].merge(saved, on=keys, how='left', validate='one_to_one', indicator=True)
    if len(saved) != len(rows) or not rows.pop('_merge').eq('both').all():
        raise ValueError(f'{path} does not cover exactly the cached rows; '
                         'rerun notebooks/2_bcpc_out_of_fold.ipynb.')
    if 'prompt' not in rows:
        rows.insert(rows.columns.get_loc('task') + 1, 'prompt', data.rows['prompt'].to_numpy())
    elif not rows['prompt'].eq(data.rows['prompt']).all():
        raise ValueError(f'Prompt text in {path} differs from the caches.')
    if not rows[STAKES].eq(data.rows[STAKES]).all():
        raise ValueError(f'Stakes classes in {path} differ from the cache labels.')
    return rows


def _load_rows(run_dir, include_excluded=False):
    """Activations of every horizon-free row, joined to the bundle's rows in cache order.

    The bundle's saved `bcpc_arc_length` is taken as given; the BCPC is not rebuilt or rerun.
    With `include_excluded`, the rows of `cv.EXCLUDED_TEMPLATES` follow the fitted ones,
    joined to `excluded_rows.parquet`; `in_bcpc_fit` marks which rows the BCPC was fitted on.
    """
    data = cv.load_model(run_dir)
    config = data.config
    metadata = json.loads((config.bcpc_dir / 'model.json').read_text(encoding='utf-8'))
    for key in ('model_name', 'layer_component', 'position'):
        if metadata[key] != getattr(config, key):
            raise ValueError(f'BCPC bundle/config {key} mismatch in {config.bcpc_dir}.')
    rows = _join_bundle_rows(data, config.bcpc_dir / 'rows.parquet').assign(in_bcpc_fit=True)
    X = data.X
    if include_excluded:
        path = config.bcpc_dir / 'excluded_rows.parquet'
        if not path.is_file():
            raise ValueError(f'{config.bcpc_dir} has no excluded_rows.parquet; '
                             'rerun notebooks/2_bcpc_out_of_fold.ipynb.')
        excluded = cv.load_model(run_dir, excluded=True)
        rows = pd.concat([rows, _join_bundle_rows(excluded, path).assign(in_bcpc_fit=False)],
                         ignore_index=True)
        del data
        X = np.concatenate([X, excluded.X])
        del excluded
    if not rows['stakes_rank'].eq(rows[STAKES].map({c: i + 1 for i, c in enumerate(cv.LEVELS)})).all():
        raise ValueError('Bundle stakes_rank does not follow the ladder order.')
    rows['horizon_type'] = 'horizon_free'
    rows['pls_fit_weight'] = 1.0 / (rows['source_file'].nunique()
                                    * rows.groupby('source_file')['source_row'].transform('size'))
    return config, metadata, rows, X


def fit_arc_length_curve(arc, points, interior_knots=1, degree=3):
    """Least-squares spline of one template's PLS1-3 row scores on their `bcpc_arc_length`.

    The boundary knots sit at the template's smallest and largest arc length and the interior
    knots at its arc-length quantiles, so the curve is the template's smoothed mean position
    as a function of arc length. Tied arc lengths (rows at either end of the BCPC spline) are
    allowed.
    """
    if (isinstance(interior_knots, bool) or not isinstance(interior_knots, int)
            or interior_knots < 0):
        raise ValueError('interior_knots must be a nonnegative integer.')
    arc, points = np.asarray(arc, dtype=np.float64), np.asarray(points, dtype=np.float64)
    if (arc.ndim != 1 or points.shape != (len(arc), 3)
            or not (np.isfinite(arc).all() and np.isfinite(points).all())):
        raise ValueError('Expected finite arc lengths and matching PLS1-3 rows.')
    order = np.argsort(arc, kind='stable')
    arc, points = arc[order], points[order]
    if len(np.unique(arc)) < degree + interior_knots + 2:
        raise ValueError('Too few distinct arc lengths for a template curve.')
    interior = np.quantile(arc, np.arange(1, interior_knots + 1) / (interior_knots + 1))
    knots = np.r_[np.full(degree + 1, arc[0]), interior, np.full(degree + 1, arc[-1])]
    return make_lsq_spline(arc, points, knots, k=degree)


def fit_surface(run_dir, settings=None):
    """Fit PLS, the template-curve plane rotation and the slice surface; map every row."""
    s = settings or SurfaceSettings()
    config, bcpc_metadata, rows, X = _load_rows(run_dir, s.include_excluded_templates)
    n_rows, feature_count = X.shape
    if min(feature_count, n_rows - 1) < s.pls_components:
        raise ValueError(f'Insufficient rows or features for {s.pls_components} PLS components.')
    pls_columns = [f'PLS{i}' for i in range(1, s.pls_components + 1)]

    # PLS: `cv.load_model` keeps each cache's rows contiguous and in cache order.
    positions = rows.groupby('source_file', sort=False).indices
    for rel, index in positions.items():
        if not np.array_equal(index, np.arange(index[0], index[0] + len(index))):
            raise RuntimeError(f'Rows of {rel} are not contiguous.')

    def batches(rel):
        index = positions[rel]
        for start in range(0, len(index), s.chunk_rows):
            stop = min(start + s.chunk_rows, len(index))
            yield start, stop, X[index[start]:index[start] + (stop - start)]

    frames = {rel: rows.iloc[index].reset_index(drop=True) for rel, index in positions.items()}
    pls = WeightedPls(s.pls_components, TARGET).fit(batches, frames, feature_count)
    if pls.rows != n_rows or not np.isclose(pls.mass, 1.0):
        raise RuntimeError('PLS row count or total fitting mass mismatch.')
    unrotated = np.empty((n_rows, s.pls_components))
    for start in range(0, n_rows, s.chunk_rows):
        stop = min(start + s.chunk_rows, n_rows)
        unrotated[start:stop] = (X[start:stop] - pls.x_mean) @ pls.base_rotations
    if not np.isfinite(unrotated).all():
        raise RuntimeError('Nonfinite PLS scores.')

    # Per-template curves of PLS1-3 on bcpc_arc_length, sampled at matched arc lengths over the
    # range every template covers; the samples set the shared-plane rotation.
    arc = rows[TARGET].to_numpy()
    files = sorted(positions)  # one template order throughout
    splines = {rel: fit_arc_length_curve(arc[positions[rel]], unrotated[positions[rel], :3],
                                         s.spline_interior_knots, s.spline_degree) for rel in files}
    q_low, q_high = s.curve_range_quantiles
    arc_low = max(float(np.quantile(arc[positions[rel]], q_low)) for rel in files)
    arc_high = min(float(np.quantile(arc[positions[rel]], q_high)) for rel in files)
    if not arc_low < arc_high:
        raise ValueError(f'The templates share no {TARGET} range between quantiles {q_low} and {q_high}.')
    arc_grid = np.linspace(arc_low, arc_high, s.arc_samples)
    # As in the old pipeline, every template counts equally in the rotation.
    rotation, rotation_diagnostics = centroid_plane_rotation(
        [splines[rel](arc_grid) for rel in files], ridge_relative=s.plane_rotation_ridge)
    file_splines = {rel: BSpline(spline.t, spline.c @ rotation.T, spline.k, extrapolate=spline.extrapolate)
                    for rel, spline in splines.items()}
    pls_rotations = pls.base_rotations.copy()
    pls_rotations[:, :3] = pls.base_rotations[:, :3] @ rotation.T
    rows[pls_columns] = unrotated
    rows[SURFACE_COLUMNS] = unrotated[:, :3] @ rotation.T
    rotated = rows[SURFACE_COLUMNS].to_numpy()
    curve_residual = np.concatenate([
        np.linalg.norm(file_splines[rel](arc[positions[rel]]) - rotated[positions[rel]], axis=1)
        for rel in files])
    curves = pd.Series(dict(arc_length_low=arc_low, arc_length_high=arc_high,
                            range_quantile_low=q_low, range_quantile_high=q_high,
                            row_residual_rms=float(np.sqrt(np.mean(curve_residual ** 2)))),
                       name='template curves')

    # Per-template class means in rotated PLS1-3, used only for the plots.
    centroids = (rows.groupby(['source_file', 'template', STAKES, 'stakes_rank'], sort=False)
                 [SURFACE_COLUMNS].mean().reset_index().sort_values(['source_file', 'stakes_rank'],
                                                                   kind='stable'))

    # Zero line: each slice's zero of arc_length_parallel is the PLS1 at which rows at the slice's
    # height reach refined_stakes = 1 (bcpc_arc_length = arc_zero), from an equal-template fit of
    # PLS1 on arc length, PLS3 and their product. It is linear in slice height.
    arc_zero = float(bcpc_metadata['refined_stakes']['arc_zero'])
    height = rotated[:, 2]
    design = np.column_stack([np.ones(n_rows), arc, height, arc * height])
    weight, pls1 = rows['pls_fit_weight'].to_numpy(), rotated[:, 0]
    beta = np.linalg.lstsq(design * np.sqrt(weight)[:, None], pls1 * np.sqrt(weight), rcond=None)[0]
    zero_line_r2 = 1 - (np.sum(weight * (pls1 - design @ beta) ** 2)
                        / np.sum(weight * (pls1 - np.average(pls1, weights=weight)) ** 2))
    origin = np.array([beta[0] + beta[1] * arc_zero, beta[2] + beta[3] * arc_zero])

    # Surface: equal-row cubic graph, cut into slices that each start at the zero line.
    surface_points = rows[SURFACE_COLUMNS].to_numpy()
    surface = SliceSurface.fit(surface_points, origin, s.slices)
    residual = surface.evaluate(surface_points[:, [0, 2]])[:, 1] - surface_points[:, 1]
    surface.build_cache(surface_points[:, 2])
    rng = np.random.default_rng(s.seed)
    validation = validate_slice_mapping(
        surface, surface_points[rng.choice(n_rows, min(n_rows, s.validation_rows), replace=False)])
    mapped = surface.map_points(surface_points, batch_rows=s.mapping_batch_rows)
    rows = pd.concat([rows, pd.DataFrame(mapped, index=rows.index)], axis=1)
    if not np.isfinite(rows[COORDINATE_COLUMNS].to_numpy()).all():
        raise RuntimeError('Nonfinite surface coordinates.')

    check_positions = np.sort(rng.choice(n_rows, min(n_rows, s.reload_check_rows), replace=False))
    diagnostics = dict(
        pls=pd.DataFrame({'component': pls_columns,
                          'score_variance': pls.component_variances,
                          'cumulative_x_variance_share': np.cumsum(pls.x_variance_share),
                          'cumulative_weighted_training_r2': pls.cumulative_r2}),
        weights=rows.groupby('register').agg(templates=('template', 'nunique'), rows=('source_row', 'size'),
                                             total_pls_weight=('pls_fit_weight', 'sum')),
        template_curves=curves,
        rotation=pd.Series(rotation_diagnostics, name='template-curve plane rotation'),
        surface=pd.Series(dict(rows=n_rows, design_rank=surface.fit_rank, condition=surface.fit_condition,
                               pls2_residual_rms=float(np.sqrt(np.mean(residual ** 2))),
                               zero_line_arc_length=arc_zero, zero_line_pls1_at_height_0=origin[0],
                               zero_line_pls1_per_height=origin[1], zero_line_fit_r2=float(zero_line_r2)),
                           name='surface fit'),
        validation=pd.Series(validation, name='slice validation'),
        mapping=rows.groupby('coordinate_status').size().rename('rows').to_frame(),
        flags=rows[['surface_extended', 'height_extrapolated']].sum().rename('rows').to_frame(),
    )
    return SurfaceFit(config, s, rows, pls, rotation, pls_rotations, centroids, file_splines,
                      surface, diagnostics, bcpc_metadata,
                      X[check_positions].copy(), check_positions)


def save(result, notebook=None):
    """Write model.npz, model.json and rows.parquet to pls/; return the folder.

    The saved model is reloaded and must reproduce a sample of rows' coordinates from their
    raw activations before this returns.
    """
    config, s, pls, rows = result.config, result.settings, result.pls, result.rows
    directory = surface_dir(config)
    directory.mkdir(parents=True, exist_ok=True)
    export_rotation = np.eye(s.pls_components)
    export_rotation[:3, :3] = result.slice_rotation
    arrays = dict(
        pls_mean=pls.x_mean, pls_target_mean=np.asarray(pls.y_mean),
        pls_rotations=result.pls_rotations, pls_base_rotations=pls.base_rotations,
        slice_rotation=result.slice_rotation, pls_export_rotation=export_rotation,
        pls_weights=pls.weights @ export_rotation.T, pls_loadings=pls.loadings @ export_rotation.T,
        pls_target_loadings=export_rotation @ pls.target_loadings,
        pls_component_variances=np.diag(export_rotation @ pls.score_covariance @ export_rotation.T),
        stakes_order=np.asarray(cv.LEVELS, dtype=str),
        pls_component_names=np.asarray([f'PLS{i}' for i in range(1, s.pls_components + 1)], dtype=str),
        surface_component_names=np.asarray(SURFACE_COLUMNS, dtype=str),
        **result.surface.arrays())
    file_spline_keys = {}
    for i, (rel, spline) in enumerate(result.file_splines.items()):
        file_spline_keys[rel] = f'file_spline_{i}'
        arrays.update(spline_arrays(f'file_spline_{i}', spline))

    def write_npz(path):
        with path.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
    _write(directory / 'model.npz', write_npz)
    _write(directory / 'rows.parquet', lambda path: rows.to_parquet(path, engine='pyarrow', index=False))

    def plain(table):
        return json.loads(table.to_json())
    metadata = dict(
        schema_version=SCHEMA_VERSION, geometry_type=GEOMETRY_TYPE,
        created_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        notebook=notebook, module='scripts/bcpc_surface.py',
        model_name=config.model_name, layer_component=config.layer_component, position=config.position,
        feature_count=int(len(pls.x_mean)), rows=int(len(rows)), fitting_rows='horizon_free',
        excluded_templates=[] if s.include_excluded_templates else list(cv.EXCLUDED_TEMPLATES),
        bcpc_excluded_templates=list(cv.EXCLUDED_TEMPLATES),
        target=TARGET, stakes_order=list(cv.LEVELS), stakes_merges=cv.STAKES_MERGES,
        bcpc_bundle=_bundle_link(config), weighting=WEIGHTING,
        settings=asdict(s),
        pls_explained=dict(frame='unrotated PLS components (the PLS1-3 rotation preserves subspace totals)',
                           total_feature_variance=pls.total_x_variance,
                           cumulative_x_variance_share=[float(v) for v in np.cumsum(pls.x_variance_share)],
                           cumulative_weighted_training_r2=[float(v) for v in pls.cumulative_r2]),
        template_curves=dict(parameter=TARGET, **plain(result.diagnostics['template_curves'])),
        rotation=plain(result.diagnostics['rotation']),
        surface_fit=plain(result.diagnostics['surface']),
        validation=plain(result.diagnostics['validation']),
        slice_config=asdict(s.slices), file_spline_keys=file_spline_keys,
        score_frame='template-curve plane rotation of PLS1-3; PLS4-16 unrotated; no rescaling',
        coordinate_units='unscaled 3D PLS scores',
        coordinate_meaning=dict(surface_u='projected PLS1', surface_v='snapped PLS3',
                                arc_length_parallel=('signed slice length from where the slice meets the '
                                                     'zero line (refined_stakes = 1); slice_origin holds '
                                                     'its PLS1 as coefficients in slice height'),
                                arc_length_orthogonal='PLS3 height relative to zero; not a geodesic length'),
        transform="(X - pls_mean) @ pls_rotations, then map PLS1-3 onto the slices",
        files={name: _sha256(directory / name) for name in ('model.npz', 'rows.parquet')},
        code={name: _sha256(ROOT / name) for name in CODE_FILES},
        runtime_versions=bcpc_bundle._runtime_versions())
    _write(directory / 'model.json',
           lambda path: path.write_text(json.dumps(metadata, indent=2), encoding='utf-8'))

    arrays, metadata, coordinates = load_surface(config)
    scores = (result.check_activations - arrays['pls_mean']) @ arrays['pls_rotations']
    mapped = coordinates.map_points(scores[:, :3], progress_seconds=None)
    expected = rows.iloc[result.check_positions]
    if not (np.allclose(scores, expected[list(arrays['pls_component_names'])].to_numpy(), rtol=0, atol=1e-8)
            and all(np.allclose(mapped[c], expected[c], rtol=0, atol=1e-8) for c in COORDINATE_COLUMNS)):
        raise RuntimeError(f'The reloaded surface does not reproduce the saved coordinates in {directory}.')
    return directory


def load_surface(config):
    """(arrays, metadata, coordinates) of the saved surface, the bundle that
    `inference_projection.project_caches` takes."""
    directory = surface_dir(config)
    metadata = json.loads((directory / 'model.json').read_text(encoding='utf-8'))
    if metadata.get('schema_version') != SCHEMA_VERSION or metadata.get('geometry_type') != GEOMETRY_TYPE:
        raise ValueError(f'Unsupported surface model in {directory}.')
    if _sha256(directory / 'model.npz') != metadata['files']['model.npz']:
        raise ValueError(f'{directory / "model.npz"} does not match its metadata checksum.')
    for key in ('model_name', 'layer_component', 'position'):
        if metadata[key] != getattr(config, key):
            raise ValueError(f'Surface/config {key} mismatch in {directory}.')
    _check_bundle_link(config, metadata)
    with np.load(directory / 'model.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return arrays, metadata, SliceSurface.from_arrays(arrays, metadata['slice_config'])


def project_inference(config, datasets=None):
    """Map every cached inference corpus through the saved surface and BCPC bundle.

    Returns {dataset name: (csv path, csv rows, diagnostics)}; each CSV is written to
    pls/inference/<dataset>.csv. The original inference/<dataset>/ files are not touched.
    """
    bundle = load_surface(config)
    fit, _ = bcpc_bundle.load_bcpc_bundle(config.bcpc_dir)

    def scores(X):
        frame = bcpc_bundle.score_activations(fit, X)
        return {column: frame[column].to_numpy() for column in BCPC_COLUMNS}

    directory = inference_dir(config)
    directory.mkdir(parents=True, exist_ok=True)
    results = {}
    for dataset in inference_datasets.cached_datasets(config, datasets or inference_datasets.DATASETS):
        records = dataset.build_records()
        paths = cache_activations.cached_paths(
            config, records, dataset.directory(config) / 'activations', dataset.cache_namespace)
        result, diagnostics = inference_projection.project_caches(
            config, records, paths, row_fields=dataset.row_fields,
            csv_columns=[*dataset.csv_columns, *BCPC_COLUMNS], bundle=bundle, scores=scores)
        path = directory / f'{dataset.name}.csv'
        _write(path, lambda temporary: result.to_csv(temporary, index=False, encoding='utf-8'))
        results[dataset.name] = (path, result, diagnostics)
    return results


# ---------------------------------------------------------------- plots


def stakes_colors(order):
    """Ordered classes spread evenly over the ordinal ramp, lightest for the lowest stakes."""
    steps = np.linspace(0, len(STAKES_RAMP) - 1, len(order)).round().astype(int)
    return {name: STAKES_RAMP[step] for name, step in zip(order, steps)}


def style_figure(fig, title, axis_names):
    axis = dict(backgroundcolor=SURFACE_COLOR, gridcolor=GRID, zerolinecolor=BASELINE,
                color=INK_SECONDARY, showbackground=True)
    fig.update_layout(
        template='plotly_white', height=800, paper_bgcolor=SURFACE_COLOR,
        title=dict(text=title, font=dict(color=INK, size=16)),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=INK_SECONDARY),
        # A horizontal legend below the scene never collides with a colorbar.
        legend=dict(itemsizing='constant', bgcolor=SURFACE_COLOR, orientation='h',
                    yanchor='top', y=0, x=0),
        margin=dict(l=16, r=16, t=64, b=16),
        scene=dict(aspectmode='data', camera=dict(eye=dict(x=1.7, y=1.7, z=1.0)),
                   **{f'{key}axis': dict(title=name, **axis) for key, name in zip('xyz', axis_names)}),
    )
    return fig


def _plot_rows(rows, max_rows, seed):
    return rows if max_rows is None or len(rows) <= max_rows else rows.sample(max_rows, random_state=seed)


def _spline_lines(result):
    """Every template's curve over its own arc-length range, as one trace with gaps, so it
    takes one legend entry."""
    lines, labels = [], []
    for rel, spline in result.file_splines.items():
        parameters = np.linspace(spline.t[spline.k], spline.t[-spline.k - 1], result.settings.arc_samples)
        lines.extend([spline(parameters), np.full((1, 3), np.nan)])
        labels.extend([Path(rel).stem] * (len(parameters) + 1))
    return np.vstack(lines), labels


def plot_centroid_splines(result, max_rows=8_000, seed=42):
    """Rotated PLS1-3 rows by stakes class, per-template class means, and the template curves."""
    import plotly.graph_objects as go

    rows = _plot_rows(result.rows, max_rows, seed)
    colors = stakes_colors(cv.LEVELS)
    fig = go.Figure()
    for name in cv.LEVELS:
        group = rows.loc[rows[STAKES].eq(name)]
        # Legend-only trace: the translucent scatter would render a washed-out swatch.
        fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', name=name,
                                   legendgroup=name, marker=dict(size=8, color=colors[name])))
        fig.add_trace(go.Scatter3d(
            x=group['PLS1'], y=group['PLS2'], z=group['PLS3'], mode='markers', name=name,
            legendgroup=name, showlegend=False, opacity=0.35, marker=dict(size=2, color=colors[name]),
            customdata=group[['template', 'task', TARGET, 'arc_length_parallel']].to_numpy(),
            hovertemplate=(f'<b>{name}</b><br>%{{customdata[0]}}<br>%{{customdata[1]}}'
                           '<br>bcpc_arc_length %{customdata[2]:.3g}'
                           '<br>arc_length_parallel %{customdata[3]:.3g}<extra></extra>')))
    centroids = result.centroids
    fig.add_trace(go.Scatter3d(
        x=centroids['PLS1'], y=centroids['PLS2'], z=centroids['PLS3'], mode='markers',
        name='Template class means', marker=dict(size=6, color=centroids[STAKES].map(colors).tolist(),
                                                 line=dict(color=INK, width=1)),
        customdata=centroids[['template', STAKES]].to_numpy(),
        hovertemplate='<b>%{customdata[1]}</b> mean<br>%{customdata[0]}<extra></extra>'))
    line, labels = _spline_lines(result)
    fig.add_trace(go.Scatter3d(
        x=line[:, 0], y=line[:, 1], z=line[:, 2], mode='lines', name=f'Template curves on {TARGET}',
        line=dict(color=INK_SECONDARY, width=3), text=labels,
        hovertemplate='%{text}<extra>curve</extra>', connectgaps=False))
    return style_figure(fig, f'{result.config.model_slug}: rotated PLS1-3, template class means '
                             'and arc-length curves', ROTATED_AXIS_TITLES)


def plot_surface(result, max_rows=8_000, seed=42, grid_points=129, margin=0.10):
    """The PLS2 = f(PLS1, PLS3) graph, representative height slices, the slices' zero line,
    and template curves."""
    import plotly.graph_objects as go

    rows, surface = result.rows, result.surface
    plot_rows = _plot_rows(rows, max_rows, seed)
    fig = go.Figure(go.Scatter3d(
        x=plot_rows['PLS1'], y=plot_rows['PLS2'], z=plot_rows['PLS3'], mode='markers', name='Rows',
        opacity=0.35, marker=dict(size=2, color=plot_rows[TARGET], colorscale=SEQUENTIAL_SCALE,
                                  colorbar=dict(title=TARGET, thickness=12)),
        customdata=plot_rows[['template', STAKES, TARGET, 'arc_length_parallel',
                              'arc_length_orthogonal']].to_numpy(),
        hovertemplate=('<b>%{customdata[1]}</b><br>%{customdata[0]}<br>bcpc_arc_length %{customdata[2]:.3g}'
                       '<br>parallel %{customdata[3]:.3g}, orthogonal %{customdata[4]:.3g}<extra></extra>')))
    # Draw the surface, slices and zero line only inside the rows' PLS1-3 box, padded on each side
    # by `margin` of the rows' range on that axis; the cubic graph runs far outside it.
    points = rows[SURFACE_COLUMNS].to_numpy()
    pad = (points.max(axis=0) - points.min(axis=0)) * margin
    low, high = points.min(axis=0) - pad, points.max(axis=0) + pad

    def in_box(values):
        values = values.copy()
        values[np.any((values < low) | (values > high), axis=-1)] = np.nan
        return values

    xgrid = np.linspace(low[0], high[0], grid_points)
    zgrid = np.linspace(max(low[2], surface.heights[0]), min(high[2], surface.heights[-1]), grid_points)
    u, v = np.meshgrid(xgrid, zgrid)
    mesh = in_box(surface.evaluate(np.column_stack([u.ravel(), v.ravel()]))).reshape(grid_points, grid_points, 3)
    fig.add_trace(go.Surface(x=mesh[:, :, 0], y=mesh[:, :, 1], z=mesh[:, :, 2], opacity=0.35,
                             showscale=False, colorscale=[[0, BASELINE], [1, BASELINE]],
                             name='Graph with tangent extensions', showlegend=True, hoverinfo='skip'))
    for n, i in enumerate(np.unique(np.linspace(0, len(surface.heights) - 1, 9).astype(int))):
        line = in_box(surface.evaluate(np.column_stack([xgrid, np.full(len(xgrid), surface.heights[i])])))
        fig.add_trace(go.Scatter3d(x=line[:, 0], y=line[:, 1], z=line[:, 2], mode='lines',
                                   name='Height slices', legendgroup='slices', showlegend=n == 0,
                                   connectgaps=False,
                                   line=dict(color=INK_SECONDARY, width=3),
                                   hovertemplate=f'Slice {i}<br>PLS3 {surface.heights[i]:.3g}<extra></extra>'))
    # Every slice's zero of arc_length_parallel, which lies on the surface.
    zero_heights = np.linspace(surface.heights[0], surface.heights[-1], grid_points)
    zero = in_box(surface.evaluate(np.column_stack([surface.origin_u(zero_heights), zero_heights])))
    fig.add_trace(go.Scatter3d(x=zero[:, 0], y=zero[:, 1], z=zero[:, 2], mode='lines',
                               name='Slice zeros (refined_stakes = 1)', connectgaps=False,
                               line=dict(color=ZERO_LINE_COLOR, width=6),
                               hovertemplate=('arc_length_parallel = 0<br>PLS1 %{x:.3g}, PLS3 %{z:.3g}'
                                              '<extra></extra>')))
    line, labels = _spline_lines(result)
    fig.add_trace(go.Scatter3d(x=line[:, 0], y=line[:, 1], z=line[:, 2], mode='lines',
                               name='Template curves', line=dict(color=INK, width=4), text=labels,
                               hovertemplate='%{text}<extra>curve</extra>', connectgaps=False))
    return style_figure(fig, f'{result.config.model_slug}: PLS2 = f(PLS1, PLS3), height slices and slice zeros',
                        ROTATED_AXIS_TITLES)


def save_plots(result, max_rows=8_000, seed=42):
    """Write every figure as self-contained HTML to pls/plots/; return {name: figure}."""
    plots_dir = surface_dir(result.config) / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        'pls_centroid_splines': plot_centroid_splines(result, max_rows, seed),
        'pls_surface': plot_surface(result, max_rows, seed),
    }
    for name, fig in figures.items():
        # Embed plotly.js so the files open offline.
        _write(plots_dir / f'{name}.html',
               lambda path, fig=fig: fig.write_html(path, include_plotlyjs=True, full_html=True))
    return figures
