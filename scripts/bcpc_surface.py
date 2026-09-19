"""Geometric surface in 16-D PLS space, with (u, v) coordinates tied to the BCPC arc length.

This replaces the PLS1-3 height-slice surface, which rotated PLS1-3 into the template
curves' shared plane and fitted the graph PLS2 = f(PLS1, PLS3) as a fast approximation.
The target comes from the saved BCPC bundle in `artifacts/<model>/bcpc/` (`bcpc_bundle.py`).
Downstream of that bundle nothing uses the hand-assigned stakes classes, so this module
never reads them, and the surface outputs do not carry them.

1. every horizon-free row outside `cv.EXCLUDED_TEMPLATES` is joined to `bcpc/rows.parquet`
   on (source_file, source_row), whose saved arc lengths are taken as given. With
   `SurfaceSettings.include_excluded_templates`, the excluded templates' rows are added,
   joined to `bcpc/excluded_rows.parquet`: the BCPC never saw them, but the surface fits them
2. 16 centred, unscaled PLS components against `bcpc_arc_length`, unrotated
3. shape: a two-parameter surface in all 16 PLS scores, fitted by orthogonal-distance
   (closest-point) least squares (`geometric_surface.fit_shape`): by default a B-spline
   surface held convex along one normal direction. The arc length is not used; for the
   older general 'bspline' shape it only picks the starting chart
4. u: a smooth function on the surface, fitted separately to the rows' `bcpc_arc_length` at
   their closest points
5. v: the lines of constant v are the orthogonal trajectories of the lines of constant u,
   and v is surface length along the level curve u = `arc_zero` (refined_stakes = 1),
   measured from where that curve passes closest to the rows' weighted mean

Each row is mapped by its closest point on the surface. `surface_u` is u there,
`arc_length_parallel` is the signed surface length along the row's v line from u = `arc_zero`,
and `arc_length_orthogonal` = `surface_v` is the surface length along the zero curve.
The weighting is equal total weight per template file, split equally among its rows,
in PLS, the shape and the u map. These are in-sample descriptive fits, not held-out
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

from . import bcpc_bundle
from . import bcpc_cross_validation as cv
from . import cache_activations, inference_datasets, inference_projection
from . import geometric_surface as gs
from .pipeline_config import ROOT
from .pls_fit import WeightedPls

SCHEMA_VERSION = 3  # 3: geometric surface in all PLS components, orthogonal (u, v)
GEOMETRY_TYPE = 'bcpc_pls_geometric_surface'
TARGET = 'bcpc_arc_length'
BCPC_COLUMNS = ['bcpc_arc_length', 'refined_stakes']
# Bundle columns that come from the hand-assigned classes; the surface stage drops them.
CLASS_COLUMNS = ['stakes', 'stakes_original', 'stakes_rank', 'refined_residual']
COORDINATE_COLUMNS = ['surface_s', 'surface_t', 'surface_u', 'surface_v', 'arc_length_parallel',
                      'arc_length_orthogonal', 'surface_distance']
CODE_FILES = ('scripts/bcpc_surface.py', 'scripts/geometric_surface.py', 'scripts/pls_fit.py',
              'scripts/inference_projection.py')
WEIGHTING = dict(
    pls='equal total weight per template file, split equally among its rows',
    shape='equal total weight per template file, split equally among its rows',
    u_map='equal total weight per template file, split equally among its rows')

# Plot palette on a light chart surface. Continuous values take the blue sequential ramp.
SEQUENTIAL_SCALE = [[0.0, '#cde2fb'], [0.5, '#2a78d6'], [1.0, '#0d366b']]
ZERO_LINE_COLOR = '#d62728'  # the level curve u = arc_zero, apart from the blue rows and grey net
SURFACE_COLOR = '#fcfcfb'
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
GRID = '#e1e0d9'
BASELINE = '#c3c2b7'


@dataclass(frozen=True)
class SurfaceSettings:
    pls_components: int = 16
    surface: gs.SurfaceConfig = field(default_factory=gs.SurfaceConfig)
    chunk_rows: int = 1024
    reload_check_rows: int = 256
    seed: int = 42
    # Also fit the rows of cv.EXCLUDED_TEMPLATES, with the arc lengths the BCPC bundle saved
    # for them in excluded_rows.parquet (scored by the full fit, which never saw them).
    include_excluded_templates: bool = False

    def __post_init__(self):
        if self.pls_components < 3:
            raise ValueError('A two-parameter surface needs at least three PLS components.')


@dataclass
class SurfaceFit:
    config: object
    settings: SurfaceSettings
    rows: pd.DataFrame          # every horizon-free row: bundle columns, PLS1-16, coordinates
    pls: WeightedPls
    coordinates: gs.OrthogonalCoordinates
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
    still yields it; a bundle that does carry it must agree. The class columns are dropped.
    """
    saved = pd.read_parquet(path).drop(columns=CLASS_COLUMNS, errors='ignore')
    keys = ['source_file', 'source_row']
    rows = data.rows[keys].merge(saved, on=keys, how='left', validate='one_to_one', indicator=True)
    if len(saved) != len(rows) or not rows.pop('_merge').eq('both').all():
        raise ValueError(f'{path} does not cover exactly the cached rows; '
                         'rerun notebooks/2_bcpc_out_of_fold.ipynb.')
    if 'prompt' not in rows:
        rows.insert(rows.columns.get_loc('task') + 1, 'prompt', data.rows['prompt'].to_numpy())
    elif not rows['prompt'].eq(data.rows['prompt']).all():
        raise ValueError(f'Prompt text in {path} differs from the caches.')
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
    rows['horizon_type'] = 'horizon_free'
    rows['pls_fit_weight'] = 1.0 / (rows['source_file'].nunique()
                                    * rows.groupby('source_file')['source_row'].transform('size'))
    return config, metadata, rows, X


def fit_pls(run_dir, settings=None):
    """Load every horizon-free row and fit PLS against `bcpc_arc_length`.

    Returns (config, bundle metadata, rows with PLS1-n columns, pls, activations).
    """
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
    scores = np.empty((n_rows, s.pls_components))
    for start in range(0, n_rows, s.chunk_rows):
        stop = min(start + s.chunk_rows, n_rows)
        scores[start:stop] = (X[start:stop] - pls.x_mean) @ pls.base_rotations
    if not np.isfinite(scores).all():
        raise RuntimeError('Nonfinite PLS scores.')
    rows[pls_columns] = scores
    return config, bcpc_metadata, rows, pls, X


@dataclass
class ShapeFit:
    config: object
    settings: SurfaceSettings
    rows: pd.DataFrame          # every horizon-free row: bundle columns, PLS scores, closest points
    pls: WeightedPls
    shape: object               # gs.ConvexSurface or gs.TensorSpline, in all PLS components
    diagnostics: pd.Series


def _start(scores, arc, weight, settings):
    """The starting chart of a 'bspline' shape; a 'convex' shape starts from its principal plane."""
    return gs.initial_chart(scores, arc, weight) if settings.surface.shape_model == 'bspline' else None


def fit_shape(run_dir, settings=None, progress=None):
    """Fit PLS and the geometric surface only; nothing is saved.

    `progress`, if given, is called as progress(iteration, objective) during the shape fit.
    """
    s = settings or SurfaceSettings()
    config, _, rows, pls, X = fit_pls(run_dir, s)
    del X
    scores = rows[[f'PLS{i}' for i in range(1, s.pls_components + 1)]].to_numpy()
    arc, weight = rows[TARGET].to_numpy(), rows['pls_fit_weight'].to_numpy()
    shape, st, distance, diagnostics = gs.fit_shape(scores, weight, _start(scores, arc, weight, s),
                                                    s.surface, progress)
    rows['surface_s'], rows['surface_t'] = st[:, 0], st[:, 1]
    rows['surface_distance'] = np.sqrt(distance)
    return ShapeFit(config, s, rows, pls, shape, pd.Series(diagnostics, name='geometric surface fit'))


def fit_surface(run_dir, settings=None, progress=None):
    """Fit PLS, the geometric surface, the u map and the orthogonal coordinates; map every row.

    `progress`, if given, is called as progress(iteration, objective) during the shape fit.
    """
    s = settings or SurfaceSettings()
    config, bcpc_metadata, rows, pls, X = fit_pls(run_dir, s)
    n_rows = len(rows)
    pls_columns = [f'PLS{i}' for i in range(1, s.pls_components + 1)]
    scores = rows[pls_columns].to_numpy()

    # 1. shape, by closest-point least squares in all PLS components
    arc, weight = rows[TARGET].to_numpy(), rows['pls_fit_weight'].to_numpy()
    shape, st, _, shape_diagnostics = gs.fit_shape(scores, weight, _start(scores, arc, weight, s),
                                                   s.surface, progress)
    # 2. u, learnt separately at the rows' closest points
    u_map = gs.fit_scalar(st, arc, weight, s.surface)
    # 3. v, orthogonal to u; v = 0 where the zero curve passes closest to the weighted mean
    arc_zero = float(bcpc_metadata['refined_stakes']['arc_zero'])
    coordinates = gs.OrthogonalCoordinates.build(shape, u_map, arc_zero, weight @ scores / weight.sum(),
                                                 s.surface)

    # Every row mapped from scratch, exactly as a new point would be.
    mapped = coordinates.map_points(scores)
    rows = pd.concat([rows, pd.DataFrame(mapped, index=rows.index)], axis=1)
    if not np.isfinite(rows[COORDINATE_COLUMNS].to_numpy()).all():
        raise RuntimeError('Nonfinite surface coordinates.')
    rows['arc_length_error'] = rows[TARGET] - rows['surface_u']
    footpoint_shift = np.abs(rows[['surface_s', 'surface_t']].to_numpy() - st).max(1)

    w = weight / weight.sum()
    u_error = rows['arc_length_error'].to_numpy()
    rng = np.random.default_rng(s.seed)
    check_positions = np.sort(rng.choice(n_rows, min(n_rows, s.reload_check_rows), replace=False))
    diagnostics = dict(
        pls=pd.DataFrame({'component': pls_columns,
                          'score_variance': pls.component_variances,
                          'cumulative_x_variance_share': np.cumsum(pls.x_variance_share),
                          'cumulative_weighted_training_r2': pls.cumulative_r2}),
        weights=rows.groupby('register').agg(templates=('template', 'nunique'), rows=('source_row', 'size'),
                                             total_pls_weight=('pls_fit_weight', 'sum')),
        shape=pd.Series(dict(**shape_diagnostics,
                             rows_with_other_footpoint_when_remapped=int((footpoint_shift > 1e-6).sum())),
                        name='geometric surface fit'),
        u_map=pd.Series(dict(
            target=TARGET,
            weighted_r2=float(1 - w @ u_error ** 2 / (w @ (arc - w @ arc) ** 2)),
            weighted_rms_error=float(np.sqrt(w @ u_error ** 2)),
            max_abs_error=float(np.abs(u_error).max()),
            target_range=float(arc.max() - arc.min())), name='u map'),
        coordinates=pd.Series(dict(
            arc_zero=arc_zero,
            zero_curve_points=len(coordinates.zero_st),
            zero_curve_v_low=float(coordinates.zero_v.min()),
            zero_curve_v_high=float(coordinates.zero_v.max()),
            max_abs_cos_u_v=coordinates.orthogonality_check(),
            max_zero_curve_miss=float(rows['zero_curve_miss'].max())), name='orthogonal coordinates'),
        mapping=rows.groupby('coordinate_status').size().rename('rows').to_frame(),
    )
    return SurfaceFit(config, s, rows, pls, coordinates, diagnostics, bcpc_metadata,
                      X[check_positions].copy(), check_positions)


def save(result, notebook=None):
    """Write model.npz, model.json and rows.parquet to pls/; return the folder.

    The saved model is reloaded and must reproduce a sample of rows' coordinates from their
    raw activations before this returns.
    """
    config, s, pls, rows = result.config, result.settings, result.pls, result.rows
    directory = surface_dir(config)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        pls_mean=pls.x_mean, pls_target_mean=np.asarray(pls.y_mean),
        pls_rotations=pls.base_rotations, pls_weights=pls.weights, pls_loadings=pls.loadings,
        pls_target_loadings=pls.target_loadings, pls_component_variances=pls.component_variances,
        pls_component_names=np.asarray([f'PLS{i}' for i in range(1, s.pls_components + 1)], dtype=str),
        **result.coordinates.arrays())

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
        target=TARGET, bcpc_bundle=_bundle_link(config), weighting=WEIGHTING,
        settings=asdict(s), surface_config=result.coordinates.config_dict(),
        pls_explained=dict(total_feature_variance=pls.total_x_variance,
                           cumulative_x_variance_share=[float(v) for v in np.cumsum(pls.x_variance_share)],
                           cumulative_weighted_training_r2=[float(v) for v in pls.cumulative_r2]),
        shape_fit=plain(result.diagnostics['shape']),
        u_map=plain(result.diagnostics['u_map']),
        coordinates=plain(result.diagnostics['coordinates']),
        mapping=plain(result.diagnostics['mapping']['rows']),
        score_frame=f'unrotated PLS1-{s.pls_components}; no rescaling',
        coordinate_units='unscaled PLS scores',
        coordinate_meaning=dict(
            surface_s='chart parameter of the closest point (working parameterisation)',
            surface_t='chart parameter of the closest point (working parameterisation)',
            surface_u=f'u at the closest point: a smooth function on the surface fitted to {TARGET}',
            surface_v='surface length along the level curve u = arc_zero, from where it passes '
                      'closest to the rows\' weighted mean; lines of constant v are orthogonal to '
                      'lines of constant u',
            arc_length_parallel='signed surface length along the row\'s line of constant v, from '
                                'u = arc_zero (refined_stakes = 1); positive where u > arc_zero',
            arc_length_orthogonal='surface_v, under its legacy name',
            surface_distance='distance from the row to its closest point, in PLS score units',
            coordinate_status=('interior; patch_boundary (the closest point is on the chart edge, so '
                               'the row lies beyond the fitted patch); path_extrapolated (its line of '
                               'constant v leaves the chart before reaching u = arc_zero)')),
        transform=f'(X - pls_mean) @ pls_rotations, then map PLS1-{s.pls_components} onto the surface',
        files={name: _sha256(directory / name) for name in ('model.npz', 'rows.parquet')},
        code={name: _sha256(ROOT / name) for name in CODE_FILES},
        runtime_versions=bcpc_bundle._runtime_versions())
    _write(directory / 'model.json',
           lambda path: path.write_text(json.dumps(metadata, indent=2), encoding='utf-8'))

    arrays, metadata, coordinates = load_surface(config)
    scores = (result.check_activations - arrays['pls_mean']) @ arrays['pls_rotations']
    mapped = coordinates.map_points(scores)
    expected = rows.iloc[result.check_positions]
    if not (np.allclose(scores, expected[list(arrays['pls_component_names'])].to_numpy(), rtol=0, atol=1e-8)
            and all(np.allclose(mapped[c], expected[c], rtol=0, atol=1e-8) for c in COORDINATE_COLUMNS)):
        raise RuntimeError(f'The reloaded surface does not reproduce the saved coordinates in {directory}.')
    return directory


def load_surface(config):
    """(arrays, metadata, coordinates) of the saved surface, the bundle that
    `inference_projection.project_caches` takes. `coordinates.map_points` takes all PLS scores."""
    directory = surface_dir(config)
    metadata = json.loads((directory / 'model.json').read_text(encoding='utf-8'))
    if metadata.get('schema_version') != SCHEMA_VERSION or metadata.get('geometry_type') != GEOMETRY_TYPE:
        raise ValueError(f'Unsupported surface model in {directory}; refit it with '
                         'notebooks/3_pls_arc_length_surface.ipynb.')
    if _sha256(directory / 'model.npz') != metadata['files']['model.npz']:
        raise ValueError(f'{directory / "model.npz"} does not match its metadata checksum.')
    for key in ('model_name', 'layer_component', 'position'):
        if metadata[key] != getattr(config, key):
            raise ValueError(f'Surface/config {key} mismatch in {directory}.')
    _check_bundle_link(config, metadata)
    with np.load(directory / 'model.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return arrays, metadata, gs.OrthogonalCoordinates.from_arrays(arrays, metadata['surface_config'])


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


def _row_hover(rows):
    customdata = rows[['template', 'task', TARGET, 'surface_u', 'surface_v',
                       'arc_length_parallel', 'surface_distance']].to_numpy()
    template = ('%{customdata[0]}<br>%{customdata[1]}<br>bcpc_arc_length %{customdata[2]:.3g}'
                '<br>u %{customdata[3]:.3g}, v %{customdata[4]:.3g}'
                '<br>arc_length_parallel %{customdata[5]:.3g}'
                '<br>distance to surface %{customdata[6]:.3g}<extra></extra>')
    return customdata, template


def _net_lines(coordinates, rows, count=9, samples=161):
    """Chart polylines of a few lines of constant u and constant v over the rows' range."""
    shape, u_map = coordinates.shape, coordinates.u
    axis = np.linspace(0.0, 1.0, samples)
    grid = np.stack(np.meshgrid(axis, axis, indexing='ij'), axis=-1).reshape(-1, 2)
    u_grid = u_map(grid)[:, 0].reshape(samples, samples)
    u_lines = []
    for level in np.quantile(rows['surface_u'], np.linspace(0.05, 0.95, count)):
        # One crossing per chart column t: the level curve as s(t) wherever it is monotone in s.
        points = []
        for j in range(samples):
            column = u_grid[:, j] - level
            crossing = np.flatnonzero(np.sign(column[:-1]) != np.sign(column[1:]))
            if len(crossing):
                i = crossing[0]
                share = column[i] / (column[i] - column[i + 1])
                points.append([axis[i] + share * (axis[i + 1] - axis[i]), axis[j]])
            else:
                points.append([np.nan, np.nan])
        u_lines.append((float(level), np.array(points)))
    v_lines = []
    for v in np.quantile(rows['surface_v'], np.linspace(0.05, 0.95, count)):
        seed = coordinates.zero_st[np.argmin(np.abs(coordinates.zero_v - v))]
        levels = np.linspace(rows['surface_u'].min(), rows['surface_u'].max(), 65)
        path = coordinates.trajectory(seed, levels)
        inside = np.all((path >= 0) & (path <= 1), axis=1)
        path[~inside] = np.nan
        v_lines.append((float(v), path))
    return u_lines, v_lines


def _surface_mesh(shape, grid_points):
    """PLS1-3 of the surface over its chart, (n, n, 3); a ConvexSurface is trimmed to its hull."""
    axis = np.linspace(0.0, 1.0, grid_points)
    st = np.stack(np.meshgrid(axis, axis, indexing='ij'), axis=-1).reshape(-1, 2)
    mesh = shape(st)[:, :3]
    if isinstance(shape, gs.ConvexSurface):
        mesh[~shape.inside_hull(st)] = np.nan
    return mesh.reshape(grid_points, grid_points, 3)


def plot_shape(result, max_rows=8_000, seed=42, grid_points=61):
    """A ShapeFit in PLS1-3: the rows coloured by bcpc_arc_length, and the fitted surface over
    its chart (a projection of the surface fitted in all PLS components)."""
    import plotly.graph_objects as go

    rows = _plot_rows(result.rows, max_rows, seed)
    customdata = rows[['template', 'task', TARGET, 'surface_distance']].to_numpy()
    fig = go.Figure(go.Scatter3d(
        x=rows['PLS1'], y=rows['PLS2'], z=rows['PLS3'], mode='markers', name='Rows',
        opacity=0.35, marker=dict(size=2, color=rows[TARGET], colorscale=SEQUENTIAL_SCALE,
                                  colorbar=dict(title=TARGET, thickness=12)),
        customdata=customdata,
        hovertemplate=('%{customdata[0]}<br>%{customdata[1]}<br>bcpc_arc_length %{customdata[2]:.3g}'
                       '<br>distance to surface %{customdata[3]:.3g}<extra></extra>')))
    mesh = _surface_mesh(result.shape, grid_points)
    fig.add_trace(go.Surface(x=mesh[:, :, 0], y=mesh[:, :, 1], z=mesh[:, :, 2], opacity=0.45,
                             showscale=False, colorscale=[[0, INK_SECONDARY], [1, INK_SECONDARY]],
                             name='Fitted surface', showlegend=True, hoverinfo='skip'))
    kind = 'convex B-spline' if isinstance(result.shape, gs.ConvexSurface) else 'B-spline'
    return style_figure(fig, f'{result.config.model_slug}: {kind} surface fitted in '
                             f'PLS1-{result.settings.pls_components}, drawn in PLS1-3',
                        ('PLS1', 'PLS2', 'PLS3'))


def plot_surface(result, max_rows=8_000, seed=42, grid_points=81):
    """The surface in PLS1-3 (a projection of the 16-D surface), its (u, v) net, the zero curve
    u = arc_zero, and the rows coloured by bcpc_arc_length."""
    import plotly.graph_objects as go

    rows, coordinates = result.rows, result.coordinates
    plot_rows = _plot_rows(rows, max_rows, seed)
    customdata, hover = _row_hover(plot_rows)
    fig = go.Figure(go.Scatter3d(
        x=plot_rows['PLS1'], y=plot_rows['PLS2'], z=plot_rows['PLS3'], mode='markers', name='Rows',
        opacity=0.35, marker=dict(size=2, color=plot_rows[TARGET], colorscale=SEQUENTIAL_SCALE,
                                  colorbar=dict(title=TARGET, thickness=12)),
        customdata=customdata, hovertemplate=hover))
    mesh = _surface_mesh(coordinates.shape, grid_points)
    fig.add_trace(go.Surface(x=mesh[:, :, 0], y=mesh[:, :, 1], z=mesh[:, :, 2], opacity=0.3,
                             showscale=False, colorscale=[[0, BASELINE], [1, BASELINE]],
                             name='Surface (PLS1-3 view)', showlegend=True, hoverinfo='skip'))
    u_lines, v_lines = _net_lines(coordinates, rows)
    for n, (level, chart) in enumerate(u_lines):
        line = coordinates.shape(np.nan_to_num(chart))[:, :3]
        line[np.isnan(chart).any(1)] = np.nan
        fig.add_trace(go.Scatter3d(x=line[:, 0], y=line[:, 1], z=line[:, 2], mode='lines',
                                   name='Lines of constant u', legendgroup='u', showlegend=n == 0,
                                   connectgaps=False, line=dict(color=INK_SECONDARY, width=3),
                                   hovertemplate=f'u = {level:.3g}<extra></extra>'))
    for n, (v, chart) in enumerate(v_lines):
        line = coordinates.shape(np.nan_to_num(chart))[:, :3]
        line[np.isnan(chart).any(1)] = np.nan
        fig.add_trace(go.Scatter3d(x=line[:, 0], y=line[:, 1], z=line[:, 2], mode='lines',
                                   name='Lines of constant v', legendgroup='v', showlegend=n == 0,
                                   connectgaps=False, line=dict(color=INK, width=3, dash='dash'),
                                   hovertemplate=f'v = {v:.3g}<extra></extra>'))
    zero = coordinates.zero_st.copy()
    zero[~np.all((zero >= 0) & (zero <= 1), axis=1)] = np.nan
    line = coordinates.shape(np.nan_to_num(zero))[:, :3]
    line[np.isnan(zero).any(1)] = np.nan
    fig.add_trace(go.Scatter3d(x=line[:, 0], y=line[:, 1], z=line[:, 2], mode='lines',
                               name='u = arc_zero (refined_stakes = 1)', connectgaps=False,
                               line=dict(color=ZERO_LINE_COLOR, width=6),
                               hovertemplate='arc_length_parallel = 0<extra></extra>'))
    return style_figure(fig, f'{result.config.model_slug}: geometric surface in PLS1-16, '
                             'drawn in PLS1-3', ('PLS1', 'PLS2', 'PLS3'))


def plot_coordinates(result, max_rows=8_000, seed=42):
    """Rows in the surface's own coordinates: v across, arc_length_parallel up."""
    import plotly.graph_objects as go

    rows = _plot_rows(result.rows, max_rows, seed)
    customdata, hover = _row_hover(rows)
    fig = go.Figure(go.Scattergl(
        x=rows['surface_v'], y=rows['arc_length_parallel'], mode='markers', name='Rows',
        marker=dict(size=4, opacity=0.5, color=rows[TARGET], colorscale=SEQUENTIAL_SCALE,
                    colorbar=dict(title=TARGET, thickness=12)),
        customdata=customdata, hovertemplate=hover))
    fig.add_hline(y=0, line=dict(color=ZERO_LINE_COLOR, width=2))
    fig.update_layout(
        template='plotly_white', height=640, paper_bgcolor=SURFACE_COLOR, plot_bgcolor=SURFACE_COLOR,
        title=dict(text=f'{result.config.model_slug}: rows in surface coordinates '
                        '(red: u = arc_zero)', font=dict(color=INK, size=16)),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=INK_SECONDARY),
        margin=dict(l=16, r=16, t=64, b=16),
        xaxis=dict(title='surface_v (length along u = arc_zero)', gridcolor=GRID, zeroline=False),
        yaxis=dict(title='arc_length_parallel (length along constant v)', gridcolor=GRID, zeroline=False))
    return fig


def plot_u_map(result, max_rows=8_000, seed=42):
    """The learnt u against the target it approximates."""
    import plotly.graph_objects as go

    rows = _plot_rows(result.rows, max_rows, seed)
    customdata, hover = _row_hover(rows)
    low, high = float(rows[TARGET].min()), float(rows[TARGET].max())
    fig = go.Figure([
        go.Scattergl(x=rows[TARGET], y=rows['surface_u'], mode='markers', name='Rows',
                     marker=dict(size=4, opacity=0.4, color='#2a78d6'),
                     customdata=customdata, hovertemplate=hover),
        go.Scatter(x=[low, high], y=[low, high], mode='lines', name='u = bcpc_arc_length',
                   line=dict(color=INK_SECONDARY, width=2, dash='dot'), hoverinfo='skip')])
    fig.update_layout(
        template='plotly_white', height=560, paper_bgcolor=SURFACE_COLOR, plot_bgcolor=SURFACE_COLOR,
        title=dict(text=f'{result.config.model_slug}: learnt u against {TARGET}', font=dict(color=INK, size=16)),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=INK_SECONDARY),
        margin=dict(l=16, r=16, t=64, b=16), legend=dict(orientation='h', yanchor='top', y=-0.12, x=0),
        xaxis=dict(title=TARGET, gridcolor=GRID, zeroline=False),
        yaxis=dict(title='surface_u', gridcolor=GRID, zeroline=False))
    return fig


def save_plots(result, max_rows=8_000, seed=42):
    """Write every figure as self-contained HTML to pls/plots/; return {name: figure}."""
    plots_dir = surface_dir(result.config) / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        'pls_surface': plot_surface(result, max_rows, seed),
        'surface_coordinates': plot_coordinates(result, max_rows, seed),
        'u_map': plot_u_map(result, max_rows, seed),
    }
    for name, fig in figures.items():
        # Embed plotly.js so the files open offline.
        _write(plots_dir / f'{name}.html',
               lambda path, fig=fig: fig.write_html(path, include_plotlyjs=True, full_html=True))
    return figures
