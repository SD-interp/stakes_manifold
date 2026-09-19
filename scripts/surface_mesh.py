"""(u, v) coordinates and an evenly spaced mesh on the saved PLS surface, following the
gradient of bcpc_arc_length, and the readout of new points in those coordinates.

Reads the surface that `notebooks/3_pls_arc_length_surface.ipynb` saved in
`artifacts/<model>/pls/shape/` (`bcpc_surface.save_shape`): the PLS transform, the convex
B-spline surface and every row's closest point on it. Nothing here reads the stakes classes
(the BCPC's class-stratified `cv_fold` is dropped too). `bcpc_arc_length` only sets the
direction of the coordinates and colours the mesh.

1. field: a smooth function a(s, t) on the surface, fitted by penalised least squares to the
   rows' `bcpc_arc_length` at their closest points (`geometric_surface.fit_scalar`). It only
   supplies the direction in which bcpc_arc_length increases on the surface.
2. coordinates: the start curve is the level curve of a through the rows' weighted median
   `bcpc_arc_length`. Gradient lines run along a's surface gradient, crossing the start curve.
   - u: signed surface length along a point's gradient line from the start curve, positive
     towards larger bcpc_arc_length;
   - v: surface length along the start curve to where the point's gradient line crosses it,
     from where the curve passes closest to the rows' weighted mean. v is constant along
     every gradient line.
   Both are read from `geometric_surface.OrthogonalCoordinates`' chart table (its
   `arc_length_parallel` and `surface_v`), which integrates the gradient lines once on a
   regular chart grid, so reading out a point needs no integration.
3. mesh: nodes at equal steps of u along gradient lines that are equally spaced in v, so the
   nodes are evenly spaced in surface length along both families of lines. The mesh's sides
   along the gradient lines follow the gradient exactly. Its sides of constant u are not
   level curves of a wherever a's gradient varies along a level curve, so there they are not
   orthogonal to the gradient; `cell_side_angles` measures by how much.
4. mapping: `bcpc_arc_length` at every mesh node, learnt from the rows for colouring. Each row
   is interpolated bilinearly in (u, v) from its cell's four nodes; node values minimise the
   weighted squared error plus a second-difference penalty along both mesh directions, whose
   weight is chosen by cross-validation over held-out tasks.
5. readout: `SurfaceReadout` maps activations or PLS scores to (u, v): PLS transform, closest
   point on the surface, chart-table lookup. `save_mesh` writes it to `pls/mesh/`.

Weights are the saved `pls_fit_weight`: equal total weight per template file, split equally
among its rows.
"""
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

from . import bcpc_bundle
from . import bcpc_surface as bs
from . import geometric_surface as gs
from .pipeline_config import ROOT, load_run_config
from .plot_style import MUTED

TARGET = bs.TARGET
# Columns of the saved shape rows this stage never carries: the class columns, if an older
# bundle still had them, and the BCPC's fold, which was dealt by stakes class.
DROPPED_COLUMNS = [*bs.CLASS_COLUMNS, 'cv_fold']
COLORSCALE = 'Viridis'           # colours of bcpc_arc_length in the mesh plot
MESH_SCHEMA_VERSION = 1
MESH_GEOMETRY_TYPE = 'bcpc_pls_surface_mesh'
READOUT_COLUMNS = ('u', 'v', 'surface_s', 'surface_t', 'surface_distance', 'readout_status')
READOUT_STATUS = ('interior', 'outside_data', 'patch_boundary', 'invalid_field')
CODE_FILES = ('scripts/surface_mesh.py', 'scripts/geometric_surface.py', 'scripts/bcpc_surface.py')


@dataclass(frozen=True)
class MeshSettings:
    levels: int = 41                    # nodes along each gradient line over the rows' u range
    lines: int = 51                     # gradient lines over the rows' v range
    substeps: int = 4                   # RK4 steps between neighbouring nodes of a gradient line
    smoothing: tuple = tuple(np.logspace(-4, 4, 17))  # candidate penalty weights of the mapping
    folds: int = 5
    seed: int = 42
    zero_curve_step: float = 1 / 256    # chart step when tracing the start curve

    def __post_init__(self):
        if self.levels < 3 or self.lines < 3:
            raise ValueError('The mesh needs at least three levels and three lines.')
        if not self.smoothing or min(self.smoothing) <= 0:
            raise ValueError('smoothing must hold positive penalty weights.')


@dataclass
class SurfaceData:
    config: object
    metadata: dict              # pls/shape/model.json
    arrays: dict                # pls/shape/model.npz: the PLS transform and the shape
    shape: object               # gs.ConvexSurface in the saved PLS components
    surface_config: gs.SurfaceConfig
    rows: pd.DataFrame          # saved rows: PLS scores and closest points, no classes
    arc_zero: float             # refined_stakes = 1 + 8 (bcpc_arc_length - arc_zero) / arc_scale
    arc_scale: float

    @property
    def pls_columns(self):
        return [f'PLS{i}' for i in range(1, self.metadata['settings']['pls_components'] + 1)]


class SurfaceReadout:
    """Activations or PLS scores -> (u, v) on the surface.

    A point is projected with the PLS transform, moved to its closest point on the surface
    (Newton steps from its few best grid local minima, `gs.global_closest_points`) and read from
    the coordinates' chart table. `readout_status` is `invalid_field` where the
    arc-length gradient is too flat for u to be defined, `patch_boundary` where the closest
    point is on the chart's edge (the point lies beyond the fitted patch), `outside_data`
    where it is outside the convex hull of the fitting rows' closest points, else `interior`.
    """

    def __init__(self, pls_mean, pls_rotations, coordinates):
        self.pls_mean, self.pls_rotations = np.asarray(pls_mean, float), np.asarray(pls_rotations, float)
        self.coordinates = coordinates

    @property
    def shape(self):
        return self.coordinates.shape

    def scores(self, X):
        return (np.asarray(X, float) - self.pls_mean) @ self.pls_rotations

    def at_chart(self, st):
        """u, v and status of chart points already on the surface."""
        st = np.atleast_2d(np.asarray(st, float))
        c = self.coordinates.coordinates(st)
        invalid = ~c['valid'] | (c['path_u_gradient'] < self.coordinates.gradient_floor)
        status = np.select([invalid, gs.on_boundary(st), ~self.shape.inside_hull(st)],
                           ['invalid_field', 'patch_boundary', 'outside_data'], default='interior')
        return c['arc_length_parallel'], c['surface_v'], status

    def from_scores(self, scores):
        st, distance = gs.global_closest_points(self.shape, np.asarray(scores, float), self.coordinates.config)
        u, v, status = self.at_chart(st)
        return pd.DataFrame(dict(u=u, v=v, surface_s=st[:, 0], surface_t=st[:, 1],
                                 surface_distance=np.sqrt(distance), readout_status=status))

    def __call__(self, X):
        """(u, v) of raw activations X, (n, features)."""
        return self.from_scores(self.scores(X))


@dataclass
class Mesh:
    readout: SurfaceReadout     # its coordinates' u is the field a
    u_step: float
    v_step: float
    u: np.ndarray               # (n_u,) u of each node row, multiples of u_step
    v: np.ndarray               # (n_v,) v of each gradient line, multiples of v_step
    st: np.ndarray              # (n_u, n_v, 2) chart position of every node, NaN where a line stops
    points: np.ndarray          # (n_u, n_v, D) PLS position of every node
    on_data: np.ndarray         # (n_u, n_v) node inside the convex hull of the rows' closest points
    start_level: float
    settings: MeshSettings
    diagnostics: pd.Series

    @property
    def coordinates(self):
        return self.readout.coordinates


@dataclass
class MeshMapping:
    values: np.ndarray          # (n_u, n_v) learnt bcpc_arc_length, NaN on nodes no row touches
    smoothing: float
    selection: pd.DataFrame     # cross-validated error per candidate penalty weight
    rows: pd.DataFrame          # the rows with u, v, their mesh cell and mapped arc length
    diagnostics: pd.Series


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_surface(run_dir):
    """The shape saved by notebook 3 for the model in `run_dir`, with its rows.

    `bs.load_shape` checks the model file's checksum and its link to the BCPC bundle; the rows
    file is checked here, and every row's saved closest point must reproduce its saved distance.
    """
    config = load_run_config(run_dir)
    arrays, metadata, shape = bs.load_shape(config)
    path = bs.shape_dir(config) / 'rows.parquet'
    if _sha256(path) != metadata['files']['rows.parquet']:
        raise ValueError(f'{path} does not match its metadata checksum.')
    rows = pd.read_parquet(path).drop(columns=DROPPED_COLUMNS, errors='ignore')
    surface_config = gs.SurfaceConfig(**{key: tuple(value) if isinstance(value, list) else value
                                         for key, value in metadata['settings']['surface'].items()})
    refined = json.loads((config.bcpc_dir / 'model.json').read_text(encoding='utf-8'))['refined_stakes']
    data = SurfaceData(config, metadata, arrays, shape, surface_config, rows,
                       float(refined['arc_zero']), float(refined['arc_scale']))
    st = rows[['surface_s', 'surface_t']].to_numpy()
    distance = np.linalg.norm(rows[data.pls_columns].to_numpy() - shape(st), axis=1)
    if not np.allclose(distance, rows['surface_distance'], rtol=0, atol=1e-6):
        raise ValueError(f'The saved closest points in {path} do not reproduce their distances.')
    return data


def refined_stakes(data, arc_length):
    return 1 + 8 * (np.asarray(arc_length) - data.arc_zero) / data.arc_scale


def _multiples(values, step):
    return step * np.arange(np.floor(values.min() / step), np.ceil(values.max() / step) + 1)


def build_mesh(data, settings=None):
    """Fit the arc-length field, build the (u, v) coordinates and the evenly spaced mesh."""
    m = settings or MeshSettings()
    rows = data.rows
    st = rows[['surface_s', 'surface_t']].to_numpy()
    arc, weight = rows[TARGET].to_numpy(), rows['pls_fit_weight'].to_numpy()
    weight = weight / weight.sum()
    config = replace(data.surface_config, zero_curve_step=m.zero_curve_step)

    field = gs.fit_scalar(st, arc, weight, config)
    start_level = gs.weighted_median(arc, weight)
    reference = weight @ rows[data.pls_columns].to_numpy()
    coordinates = gs.OrthogonalCoordinates.build(data.shape, field, start_level, reference, st, weight, config)
    readout = SurfaceReadout(data.arrays['pls_mean'], data.arrays['pls_rotations'], coordinates)

    # Node spacing from the rows' own (u, v) range; u = 0 on the start curve is a node row.
    u_rows, v_rows, status = readout.at_chart(st)
    usable = status != 'invalid_field'
    u_step = float(np.ptp(u_rows[usable]) / (m.levels - 1))
    v_step = float(np.ptp(v_rows[usable]) / (m.lines - 1))
    u_nodes, v_nodes = _multiples(u_rows[usable], u_step), _multiples(v_rows[usable], v_step)
    with np.errstate(all='ignore'):
        seeds = coordinates.zero_curve_points(v_nodes)
    node_st = coordinates.length_lines(seeds, u_nodes, m.substeps)

    finite = np.isfinite(node_st).all(-1)
    points = np.full((*finite.shape, data.shape.dimension), np.nan)
    points[finite] = data.shape(node_st[finite])
    on_data = np.zeros(finite.shape, bool)
    on_data[finite] = data.shape.inside_hull(node_st[finite])

    # The nodes read back through the chart table must return their own (u, v).
    u_back, v_back, _ = readout.at_chart(node_st[on_data])
    i, j = np.nonzero(on_data)
    residual = arc - field(st)[:, 0]
    diagnostics = pd.Series(dict(
        field_weighted_r2=float(1 - weight @ residual ** 2 / (weight @ (arc - weight @ arc) ** 2)),
        field_weighted_rms_error=float(np.sqrt(weight @ residual ** 2)),
        start_level=start_level, u_step=u_step, v_step=v_step,
        rows_u_low=float(u_rows[usable].min()), rows_u_high=float(u_rows[usable].max()),
        rows_v_low=float(v_rows[usable].min()), rows_v_high=float(v_rows[usable].max()),
        rows_invalid_field=int((~usable).sum()),
        nodes=int(finite.size), nodes_on_line=int(finite.sum()), nodes_on_data=int(on_data.sum()),
        node_readout_max_abs_u=float(np.abs(u_back - u_nodes[i]).max()),
        node_readout_max_abs_v=float(np.abs(v_back - v_nodes[j]).max()),
        max_abs_cos_gradient_level=coordinates.orthogonality_check(),
        **coordinates.table_diagnostics()), name='mesh')
    return Mesh(readout, u_step, v_step, u_nodes, v_nodes, node_st, points, on_data, start_level, m, diagnostics)


def locate_rows(mesh, rows):
    """Each row's (u, v) at the closest point in its `surface_s`, `surface_t` and its mesh cell.

    A row is `located` when its u is defined, its (u, v) lies inside the mesh's range and its
    cell's four nodes exist.
    """
    st = rows[['surface_s', 'surface_t']].to_numpy()
    u, v, status = mesh.readout.at_chart(st)
    n_u, n_v = len(mesh.u), len(mesh.v)
    u_index = (u - mesh.u[0]) / mesh.u_step
    v_index = (v - mesh.v[0]) / mesh.v_step
    in_range = (u_index >= 0) & (u_index <= n_u - 1) & (v_index >= 0) & (v_index <= n_v - 1)
    valid = status != 'invalid_field'
    i = np.clip(np.floor(np.nan_to_num(u_index)).astype(int), 0, n_u - 2)
    j = np.clip(np.floor(np.nan_to_num(v_index)).astype(int), 0, n_v - 2)
    finite = np.isfinite(mesh.st).all(-1)
    corners = finite[i, j] & finite[i + 1, j] & finite[i, j + 1] & finite[i + 1, j + 1]
    return pd.DataFrame(dict(
        u=u, v=v, readout_status=status, field_value=mesh.coordinates.u(st)[:, 0],
        mesh_u_index=u_index, mesh_v_index=v_index, mesh_cell_u=i, mesh_cell_v=j,
        located=in_range & valid & corners,
        location_status=np.select([~valid, ~in_range, ~corners], ['invalid_field', 'outside_mesh', 'no_cell'],
                                  default='located')), index=rows.index)


def _design(located, n_v, nodes, n_nodes):
    """Sparse bilinear interpolation matrix (rows, solved nodes) of located rows."""
    i, j = located['mesh_cell_u'].to_numpy(), located['mesh_cell_v'].to_numpy()
    a = located['mesh_u_index'].to_numpy() - i
    b = located['mesh_v_index'].to_numpy() - j
    corners = [(i, j, (1 - a) * (1 - b)), (i + 1, j, a * (1 - b)), (i, j + 1, (1 - a) * b), (i + 1, j + 1, a * b)]
    row = np.tile(np.arange(len(located)), 4)
    column = nodes[np.concatenate([ci * n_v + cj for ci, cj, _ in corners])]
    value = np.concatenate([w for *_, w in corners])
    return sparse.csr_matrix((value, (row, column)), shape=(len(located), n_nodes))


def _penalty(active):
    """Second differences of the node values along both mesh directions, over triples of
    neighbouring nodes that are all in `active`, as a sparse (differences, active nodes) matrix."""
    index = np.full(active.shape, -1)
    index[active] = np.arange(active.sum())
    triples = []
    for axis in (0, 1):
        a = np.moveaxis(index, axis, 0)
        first, middle, last = a[:-2], a[1:-1], a[2:]
        keep = (first >= 0) & (middle >= 0) & (last >= 0)
        triples.append(np.stack([first[keep], middle[keep], last[keep]], 1))
    triples = np.concatenate(triples)
    row = np.repeat(np.arange(len(triples)), 3)
    value = np.tile([1.0, -2.0, 1.0], len(triples))
    return sparse.csr_matrix((value, (row, triples.ravel())), shape=(len(triples), int(active.sum())))


def _solve(B, weight, y, gram_penalty, smoothing):
    """Node values minimising sum w (y - B z)^2 / sum w + smoothing * |D z|^2 / nodes, with a
    vanishing ridge towards the weighted mean so nodes no training row reaches stay defined."""
    w = weight / weight.sum()
    normal = (B.T.multiply(w)) @ B
    n = B.shape[1]
    ridge = 1e-9 * normal.diagonal().mean()
    system = normal + (smoothing / n) * gram_penalty + ridge * sparse.identity(n)
    return spsolve(system.tocsc(), B.T @ (w * y) + ridge * (w @ y))


def fit_mapping(data, mesh, settings=None):
    """Learn bcpc_arc_length on the mesh nodes, for colouring; the penalty weight is chosen by
    held-out tasks.

    The folds hold out whole tasks, dealt stratified by their mean `bcpc_arc_length`
    (`bcpc_surface.task_folds`). The chosen weight is the largest whose mean held-out error
    is within one standard error of the lowest (the one-standard-error rule).
    """
    m = settings or MeshSettings()
    # Every row is read out exactly as a new point would be, closest point found anew.
    read = mesh.readout.from_scores(data.rows[data.pls_columns].to_numpy()).set_index(data.rows.index)
    saved = data.rows[['surface_s', 'surface_t']].to_numpy()
    rows = data.rows.assign(surface_s=read['surface_s'], surface_t=read['surface_t'],
                            surface_distance=read['surface_distance'],
                            closest_point_moved=np.abs(read[['surface_s', 'surface_t']].to_numpy()
                                                       - saved).max(1) > 1e-6)
    where = locate_rows(mesh, rows)
    located = where[where['located']]
    n_u, n_v = len(mesh.u), len(mesh.v)

    # Node values are solved on every node the mesh has, so the smoothness penalty ties each
    # node to its neighbours even at the edge of the data; they are kept on the corners of
    # cells that hold a row (`active`).
    solved = np.isfinite(mesh.st).all(-1)
    active = np.zeros((n_u, n_v), bool)
    for di in (0, 1):
        for dj in (0, 1):
            active[located['mesh_cell_u'] + di, located['mesh_cell_v'] + dj] = True
    nodes = np.full(solved.size, -1)
    nodes[solved.ravel()] = np.arange(solved.sum())
    B = _design(located, n_v, nodes, int(solved.sum()))
    D = _penalty(solved)
    gram_penalty = (D.T @ D).tocsr()
    y = rows.loc[located.index, TARGET].to_numpy()
    weight = rows.loc[located.index, 'pls_fit_weight'].to_numpy()

    folds = bs.task_folds(rows.loc[located.index], m.folds, m.seed)
    records = []
    for smoothing in m.smoothing:
        for fold in range(m.folds):
            train, held = folds != fold, folds == fold
            z = _solve(B[train], weight[train], y[train], gram_penalty, smoothing)
            error = y[held] - B[held] @ z
            records.append(dict(smoothing=smoothing, fold=fold,
                                mse=float(weight[held] @ error ** 2 / weight[held].sum())))
    selection = pd.DataFrame(records).groupby('smoothing')['mse'].agg(cv_mse='mean', cv_mse_se='std')
    selection['cv_mse_se'] /= np.sqrt(m.folds)
    best = selection['cv_mse'].idxmin()
    threshold = selection.loc[best, 'cv_mse'] + selection.loc[best, 'cv_mse_se']
    chosen = float(selection.index[selection['cv_mse'] <= threshold].max())
    selection.attrs.update(best=float(best), one_se=chosen, at_grid_edge=chosen == max(m.smoothing))

    z = _solve(B, weight, y, gram_penalty, chosen)
    values = np.full((n_u, n_v), np.nan)
    values[solved] = z
    values[~active] = np.nan
    fitted = B @ z

    out = rows.join(where)
    out['mesh_arc_length'] = np.nan
    out.loc[located.index, 'mesh_arc_length'] = fitted
    out['mesh_refined_stakes'] = refined_stakes(data, out['mesh_arc_length'])

    w_all = rows['pls_fit_weight'].to_numpy()
    w = weight / weight.sum()
    residual = y - fitted
    variance = w @ (y - w @ y) ** 2
    # How much the learnt values vary across the gradient lines at equal u, against their
    # total variation over the active nodes. It would be 0 if every row of nodes at equal u
    # lay on a level curve of bcpc_arc_length.
    count = active.sum(1, keepdims=True)
    row_mean = np.where(active, values, 0.0).sum(1, keepdims=True) / np.maximum(count, 1)
    across = np.nansum((values - row_mean) ** 2)
    total = np.nansum((values - np.nanmean(values)) ** 2)
    diagnostics = pd.Series(dict(
        rows_located=int(len(located)), rows_not_located=int(len(rows) - len(located)),
        rows_closest_point_moved=int(rows['closest_point_moved'].sum()),
        located_weight_share=float(w_all[where['located'].to_numpy()].sum() / w_all.sum()),
        active_nodes=int(active.sum()),
        smoothing=chosen, smoothing_lowest_cv=float(best), smoothing_at_grid_edge=chosen == max(m.smoothing),
        weighted_r2=float(1 - w @ residual ** 2 / variance),
        weighted_rms_error=float(np.sqrt(w @ residual ** 2)),
        cv_rms_error=float(np.sqrt(selection.loc[chosen, 'cv_mse'])),
        cv_q2=float(1 - selection.loc[chosen, 'cv_mse'] / variance),
        variance_share_at_equal_u=float(across / total),
        **cell_side_angles(mesh, active)), name='mapping')
    return MeshMapping(values, chosen, selection, out, diagnostics)


def cell_side_angles(mesh, active):
    """|cos| between each active cell's two sides at its corner, in the full PLS space: the
    chord along the gradient line and the chord along the row of nodes at equal u."""
    p = mesh.points
    along = p[1:, :-1] - p[:-1, :-1]
    across = p[:-1, 1:] - p[:-1, :-1]
    cells = active[:-1, :-1] & active[1:, :-1] & active[:-1, 1:]
    with np.errstate(all='ignore'):
        cosine = np.abs((along * across).sum(-1)) / (np.linalg.norm(along, axis=-1) * np.linalg.norm(across, axis=-1))
    cosine = cosine[cells & np.isfinite(cosine)]
    return dict(cell_side_cos_median=float(np.median(cosine)), cell_side_cos_p95=float(np.quantile(cosine, 0.95)))


# ---------------------------------------------------------------- saving


def mesh_dir(config):
    return bs.surface_dir(config) / 'mesh'


def save_mesh(data, mesh, mapping, notebook=None, check_rows=256, seed=42):
    """Write the readout and the mesh to pls/mesh/: model.npz, model.json and rows.parquet.

    The saved readout is reloaded before this returns. It must reproduce the saved rows'
    (u, v) at their closest points, and a sample of rows read out again from their PLS scores
    must land on the same (u, v) to within 1e-4 of a mesh step.
    Returns the folder.
    """
    config = data.config
    directory = mesh_dir(config)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = dict(pls_mean=data.arrays['pls_mean'], pls_rotations=data.arrays['pls_rotations'],
                  pls_component_names=data.arrays['pls_component_names'],
                  **mesh.coordinates.arrays(),
                  mesh_u=mesh.u, mesh_v=mesh.v, mesh_u_step=np.asarray(mesh.u_step),
                  mesh_v_step=np.asarray(mesh.v_step), mesh_st=mesh.st, mesh_on_data=mesh.on_data,
                  mesh_arc_length=mapping.values)
    rows = mapping.rows

    def write_npz(path):
        with path.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
    bs._write(directory / 'model.npz', write_npz)
    bs._write(directory / 'rows.parquet', lambda path: rows.to_parquet(path, engine='pyarrow', index=False))
    shape_directory = bs.shape_dir(config)
    metadata = dict(
        schema_version=MESH_SCHEMA_VERSION, geometry_type=MESH_GEOMETRY_TYPE,
        created_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        notebook=notebook, module='scripts/surface_mesh.py',
        model_name=config.model_name, layer_component=config.layer_component, position=config.position,
        rows=int(len(rows)), pls_components=len(data.pls_columns),
        shape=dict(files={name: _sha256(shape_directory / name) for name in ('model.json', 'model.npz')}),
        settings=asdict(mesh.settings),
        surface_config=mesh.coordinates.config_dict(),
        start_level=mesh.start_level,
        readout=dict(
            u='signed surface length along the point\'s gradient line of bcpc_arc_length, from the '
              'start curve (the level curve at start_level); positive towards larger bcpc_arc_length',
            v='surface length along the start curve to where the point\'s gradient line crosses it, '
              'from where the curve passes closest to the rows\' weighted mean; constant along '
              'each gradient line',
            surface_distance='distance from the point to its closest point on the surface, in PLS units',
            readout_status='interior; outside_data (closest point outside the convex hull of the '
                           'fitting rows\' closest points); patch_boundary (closest point on the '
                           'chart edge); invalid_field (arc-length gradient too flat for u)',
            transform='(X - pls_mean) @ pls_rotations, closest point on the shape, chart-table lookup'),
        mesh=dict(u_step=mesh.u_step, v_step=mesh.v_step, nodes_u=len(mesh.u), nodes_v=len(mesh.v),
                  meaning='nodes at u = mesh_u[i] along the gradient line at v = mesh_v[j]; '
                          'mesh_arc_length is the learnt bcpc_arc_length there, for colouring'),
        mesh_diagnostics=json.loads(mesh.diagnostics.to_json()),
        mapping_diagnostics=json.loads(mapping.diagnostics.to_json()),
        files={name: _sha256(directory / name) for name in ('model.npz', 'rows.parquet')},
        code={name: _sha256(ROOT / name) for name in CODE_FILES},
        runtime_versions=bcpc_bundle._runtime_versions())
    bs._write(directory / 'model.json',
              lambda path: path.write_text(json.dumps(metadata, indent=2), encoding='utf-8'))

    readout, _, _ = load_mesh(config)
    st = rows[['surface_s', 'surface_t']].to_numpy()
    u, v, _ = readout.at_chart(st)
    if not (np.allclose(u, rows['u'], rtol=0, atol=1e-9, equal_nan=True)
            and np.allclose(v, rows['v'], rtol=0, atol=1e-9, equal_nan=True)):
        raise RuntimeError(f'The reloaded readout does not reproduce the saved rows in {directory}.')
    sample = np.sort(np.random.default_rng(seed).choice(len(rows), min(len(rows), check_rows), replace=False))
    fresh = readout.from_scores(rows[data.pls_columns].to_numpy()[sample])
    tolerance = 1e-4 * min(mesh.u_step, mesh.v_step)
    agree = ((np.abs(fresh['u'].to_numpy() - rows['u'].to_numpy()[sample]) <= tolerance)
             & (np.abs(fresh['v'].to_numpy() - rows['v'].to_numpy()[sample]) <= tolerance))
    if not agree.all():
        raise RuntimeError(f'{(~agree).sum()} of {len(agree)} sampled rows read out from their PLS scores '
                           f'miss their saved (u, v) in {directory}.')
    return directory


def load_mesh(config):
    """(readout, metadata, arrays) of the mesh saved by `save_mesh`.

    `readout(X)` takes raw activations, `readout.from_scores` PLS scores. The saved shape it was
    built on must be unchanged.
    """
    directory = mesh_dir(config)
    metadata = json.loads((directory / 'model.json').read_text(encoding='utf-8'))
    if (metadata.get('schema_version') != MESH_SCHEMA_VERSION
            or metadata.get('geometry_type') != MESH_GEOMETRY_TYPE):
        raise ValueError(f'Unsupported mesh model in {directory}; rebuild it with notebooks/4_surface_mesh.ipynb.')
    if _sha256(directory / 'model.npz') != metadata['files']['model.npz']:
        raise ValueError(f'{directory / "model.npz"} does not match its metadata checksum.')
    for key in ('model_name', 'layer_component', 'position'):
        if metadata[key] != getattr(config, key):
            raise ValueError(f'Mesh/config {key} mismatch in {directory}.')
    shape_directory = bs.shape_dir(config)
    for name, digest in metadata['shape']['files'].items():
        if _sha256(shape_directory / name) != digest:
            raise ValueError(f'{shape_directory / name} changed after the mesh in {directory} was built; '
                             'rebuild it with notebooks/4_surface_mesh.ipynb.')
    with np.load(directory / 'model.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    coordinates = gs.OrthogonalCoordinates.from_arrays(arrays, metadata['surface_config'])
    return SurfaceReadout(arrays['pls_mean'], arrays['pls_rotations'], coordinates), metadata, arrays


# ---------------------------------------------------------------- plots


def _polylines(points, keep, axis):
    """PLS1-3 polylines through kept nodes along `axis` (0: gradient lines, 1: rows of equal u),
    joined into one array with NaN breaks between lines and wherever a node is not kept."""
    p = np.where(keep[..., None], points[..., :3], np.nan)
    p = np.moveaxis(p, axis, 1)              # lines run along axis 1
    gap = np.full((p.shape[0], 1, 3), np.nan)
    return np.concatenate([p, gap], axis=1).reshape(-1, 3)


def plot_mesh(data, mesh, mapping, max_rows=0, seed=42):
    """The mesh in PLS1-3 (a projection of the mesh in all PLS components), nodes coloured by
    the learnt bcpc_arc_length. `max_rows` > 0 adds that many rows, hidden until toggled."""
    import plotly.graph_objects as go

    keep = mesh.on_data & np.isfinite(mapping.values)
    low, high = np.nanmin(mapping.values[keep]), np.nanmax(mapping.values[keep])
    fig = go.Figure()
    if max_rows:
        rows = bs._plot_rows(mapping.rows, max_rows, seed)
        fig.add_trace(go.Scatter3d(
            x=rows['PLS1'], y=rows['PLS2'], z=rows['PLS3'], mode='markers', name='Rows',
            visible='legendonly', opacity=0.25,
            marker=dict(size=2, color=rows[TARGET], colorscale=COLORSCALE, cmin=low, cmax=high),
            customdata=rows[['template', 'task', TARGET, 'u', 'v']].to_numpy(),
            hovertemplate='%{customdata[0]}<br>%{customdata[1]}<br>bcpc_arc_length %{customdata[2]:.3g}'
                          '<br>u %{customdata[3]:.3g}, v %{customdata[4]:.3g}<extra></extra>'))
    for axis, name, color in ((0, 'Gradient lines (constant v)', bs.INK_SECONDARY),
                              (1, 'Lines of constant u', MUTED)):
        line = _polylines(mesh.points, keep, axis)
        fig.add_trace(go.Scatter3d(x=line[:, 0], y=line[:, 1], z=line[:, 2], mode='lines', name=name,
                                   connectgaps=False, hoverinfo='skip',
                                   line=dict(color=color, width=3 if axis == 0 else 2)))
    i, j = np.nonzero(keep)
    nodes = mesh.points[i, j]
    customdata = np.column_stack([mesh.u[i], mesh.v[j], mapping.values[i, j],
                                  refined_stakes(data, mapping.values[i, j])])
    fig.add_trace(go.Scatter3d(
        x=nodes[:, 0], y=nodes[:, 1], z=nodes[:, 2], mode='markers', name='Mesh nodes',
        marker=dict(size=4, color=mapping.values[i, j], colorscale=COLORSCALE, cmin=low, cmax=high,
                    colorbar=dict(title=TARGET, thickness=12), line=dict(width=0)),
        customdata=customdata,
        hovertemplate=('u %{customdata[0]:.3g}, v %{customdata[1]:.3g}'
                       '<br>bcpc_arc_length (mesh) %{customdata[2]:.3g}'
                       '<br>refined_stakes (mesh) %{customdata[3]:.3g}<extra></extra>')))
    return bs.style_figure(fig, f'{data.config.model_slug}: (u, v) mesh on the surface fitted in '
                                f'PLS1-{len(data.pls_columns)}, drawn in PLS1-3',
                           ('PLS1', 'PLS2', 'PLS3'))
