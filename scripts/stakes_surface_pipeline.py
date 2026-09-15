"""Horizon-free stakes: BCPC target, 16 PLS scores, and the PLS1-3 height-slice surface.

Stages, in order (each method depends on the previous one):

1. `load_caches`         inventory the run's activation caches
2. `fit_bcpc`            seven-component BCPC and `bcpc_arc_length`, equal mass per file
3. `fit_pls`             16 centered, unscaled PLS components against `bcpc_arc_length`
4. `rotate_plane`        per-file centroid splines; rotate PLS1-3 into their best-fit plane
5. `fit_surface`         PLS2 = f(PLS1, PLS3), equal-row cubic graph on horizon-free rows
6. `project_stated_rows` frozen BCPC and PLS transforms applied to stated caches
7. `build_slice_cache`   5,000 fixed-PLS3 curves, validated on a sample
8. `map_coordinates`     `arc_length_parallel` and `arc_length_orthogonal` for every row
9. `export`              rows.parquet, model.npz, model.json under the run's surface folder

`arc_length_parallel` is signed slice length from the common origin;
`arc_length_orthogonal` is snapped PLS3 height (legacy name, not geodesic length).
Stated rows determine height-grid coverage only, never fitted coefficients. These are
descriptive in-sample fits, not held-out predictive evaluations.
"""
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
import gc
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from .bcpc_arc_length import BcpcArcLength, BcpcSettings
from .cache_inventory import CacheInventory
from .pipeline_config import ROOT
from .pls_fit import WeightedPls
from .stakes_height_slices import (
    LEGACY_COLUMNS, MAPPING_COLUMNS, SliceConfig, SliceSurface,
    centroid_plane_rotation, validate_slice_mapping,
)
from .stakes_surface_bundle import fit_centroid_spline, spline_arrays
from .time_horizon import log10_time_horizon_months

SURFACE_COMPONENTS = (1, 2, 3)
TARGET = 'bcpc_arc_length'
STAKES = 'stakes'
PLS_WEIGHTING = 'equal_source_file'
COORDINATE_COLUMNS = ['surface_u', 'surface_v', 'arc_length_parallel',
                      'arc_length_orthogonal', 'surface_projection_residual']

# Plot palette on a light chart surface. Stakes classes are ordered, so they take
# the blue ordinal ramp (lightest step still >= 2:1 on the surface), never cycled hues.
STAKES_RAMP = ('#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6',
               '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b')
SEQUENTIAL_SCALE = [[0.0, '#cde2fb'], [0.5, '#2a78d6'], [1.0, '#0d366b']]
SURFACE_COLOR = '#fcfcfb'
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
BASELINE = '#c3c2b7'


@dataclass(frozen=True)
class SurfaceSettings:
    bcpc: BcpcSettings = field(default_factory=BcpcSettings)
    pls_components: int = 16
    pls_spline_degree: int = 3
    pls_spline_interior_knots: int = 1
    arc_samples: int = 401
    plane_rotation_ridge: float = 1e-6
    slices: SliceConfig = field(default_factory=lambda: SliceConfig(
        n_slices=5000, arc_relative_tolerance=1e-4, max_arc_nodes=262145))
    read_batch_rows: int = 1024
    mapping_batch_rows: int = 4096
    mapping_progress_seconds: float = 5.0
    mapping_resume: bool = True
    validation_rows_per_type: int = 32
    validation_seed: int = 42


class StakesSurfacePipeline:
    def __init__(self, config, settings=None):
        self.config = config
        self.settings = settings or SurfaceSettings()
        self.output_dir = config.surface_dir
        self.pls_columns = [f'PLS{i}' for i in range(1, self.settings.pls_components + 1)]
        self.surface_columns = [f'PLS{i}' for i in SURFACE_COMPONENTS]
        if self.settings.pls_components < 3:
            raise ValueError('The slice surface requires exactly PLS1-3.')

    def load_caches(self):
        c = self.config
        self.inventory = CacheInventory(
            c.activations_dir, c.model_name, c.layer_component, c.position,
            c.stakes_merges, self.settings.read_batch_rows, key_root=c.run_dir)
        return self.inventory.summary()

    def fit_bcpc(self):
        inventory = self.inventory
        fit_files = inventory.fit_files
        rows = pd.concat([inventory.entries[rel]['frame'] for rel in fit_files], ignore_index=True)
        weights = np.concatenate([
            np.full(len(inventory.entries[rel]['frame']),
                    1.0 / (len(fit_files) * len(inventory.entries[rel]['frame'])))
            for rel in fit_files])
        X = np.empty((len(rows), inventory.feature_count), dtype=np.float64)
        offset = 0
        for rel in fit_files:
            for start, stop, block in inventory.batches(rel):
                X[offset + start:offset + stop] = block
            offset += len(inventory.entries[rel]['frame'])
        self.bcpc = BcpcArcLength(self.settings.bcpc)
        projected_rows = self.bcpc.fit(X, weights, rows, STAKES)
        del X
        gc.collect()
        projected_rows['log10_time_horizon_months'] = log10_time_horizon_months(projected_rows)
        self.projected_rows = projected_rows
        return projected_rows

    def fit_pls(self):
        """Fit PLS with equal total weight per file and score every horizon-free row."""
        inventory, n_components = self.inventory, self.settings.pls_components
        projected_rows = self.projected_rows
        n_rows = len(projected_rows)
        if min(inventory.feature_count, n_rows - 1) < n_components:
            raise ValueError(f'Insufficient rows or features for exactly {n_components} PLS components.')
        group_counts = projected_rows.groupby('pls_weight_group').size()
        n_groups = len(group_counts)
        projected_rows['pls_fit_weight'] = projected_rows['pls_weight_group'].map(
            (1.0 / (n_groups * group_counts)).to_dict()).to_numpy(dtype=float)
        aligned = {rel: frame.reset_index(drop=True)
                   for rel, frame in projected_rows.groupby('source_file', sort=False)}
        frames = {rel: aligned[rel] for rel in inventory.fit_files}
        self.pls = WeightedPls(n_components, TARGET).fit(inventory.batches, frames, inventory.feature_count)
        if self.pls.rows != n_rows or not np.isclose(self.pls.mass, 1.0):
            raise RuntimeError('PLS row count or total fitting mass mismatch.')
        self.weight_summary = projected_rows.groupby('pls_weight_group').agg(
            rows=('source_row', 'size'), total_weight=('pls_fit_weight', 'sum'))
        if not np.allclose(self.weight_summary['total_weight'], 1.0 / n_groups):
            raise RuntimeError('PLS groups do not have equal fitting mass.')
        self.group_counts = group_counts

        # Score every horizon-free row; no sampling enters the surface or file curves.
        score_frames = []
        for rel in inventory.fit_files:
            frame = aligned[rel].copy()
            values = np.empty((len(frame), n_components), dtype=np.float64)
            for start, stop, block in inventory.batches(rel):
                values[start:stop] = (block - self.pls.x_mean) @ self.pls.base_rotations
            if not np.isfinite(values).all():
                raise RuntimeError(f'Nonfinite PLS scores: {rel}')
            frame[self.pls_columns] = values
            score_frames.append(frame)
        fit_rows = pd.concat(score_frames, ignore_index=True)
        fit_rows['file'] = fit_rows['source_file'].map(lambda value: PurePosixPath(value).stem)
        if fit_rows[['source_file', 'source_row']].duplicated().any() or len(fit_rows) != n_rows:
            raise RuntimeError('Horizon-free score rows are not a one-to-one cache projection.')
        self.fit_rows = fit_rows
        self.fit_pls_unrotated = fit_rows[self.pls_columns].to_numpy(copy=True)
        self.pls_rotations = self.pls.base_rotations.copy()

        print(f'Fitted {n_components} PLS components on {self.pls.rows:,} horizon-free rows '
              f'from {len(inventory.fit_files)} files, against {TARGET}.')
        print(f'Target mean {self.pls.y_mean:.6g}, target sd {np.sqrt(self.pls.y_variance):.6g} '
              f'(arc-length units).')
        return pd.DataFrame({'component': self.pls_columns,
                             'score_variance': self.pls.component_variances,
                             'cumulative_x_variance_share': np.cumsum(self.pls.x_variance_share),
                             'cumulative_weighted_training_r2': self.pls.cumulative_r2})

    def rotate_plane(self):
        """Per-file stakes-centroid splines; rotate PLS1-3 into their best-fit shared plane.

        The normal maximizes between-file separation relative to within-curve variation;
        the first in-plane axis follows longitudinal variation. PLS4-16 are untouched.
        """
        s, surface_columns = self.settings, self.surface_columns
        fit_files = self.inventory.fit_files
        fit_rows = self.fit_rows
        # Reset to the fitted PLS frame so rerunning never rotates twice.
        fit_rows[self.pls_columns] = self.fit_pls_unrotated
        stakes_means = fit_rows.groupby(STAKES)[TARGET].mean().sort_values(kind='stable')
        self.stakes_order = stakes_means.index.tolist()
        stakes_rank = {name: i for i, name in enumerate(self.stakes_order)}
        file_splines, centroid_frames = {}, []
        for rel in fit_files:
            group = fit_rows.loc[fit_rows['source_file'].eq(rel)]
            frame = group.groupby(STAKES, sort=False)[surface_columns].mean().reset_index()
            frame['stakes_rank'] = frame[STAKES].map(stakes_rank)
            frame = frame.sort_values('stakes_rank').reset_index(drop=True)
            frame['source_file'] = rel
            frame['file'] = PurePosixPath(rel).stem
            spline, parameters = fit_centroid_spline(
                frame[surface_columns].to_numpy(), s.pls_spline_interior_knots, s.pls_spline_degree)
            frame['spline_parameter'] = parameters
            frame['spline_residual'] = np.linalg.norm(
                spline(parameters) - frame[surface_columns].to_numpy(), axis=1)
            file_splines[rel] = spline
            centroid_frames.append(frame)
        pls_centroids = pd.concat(centroid_frames, ignore_index=True)
        progress = np.linspace(0, len(self.stakes_order) - 1, s.arc_samples)
        trajectory_samples = []
        for rel, frame in zip(fit_files, centroid_frames):
            parameters = np.interp(progress, np.arange(len(self.stakes_order)), frame['spline_parameter'])
            trajectory_samples.append(file_splines[rel](parameters))
        self.slice_rotation, self.rotation_diagnostics = centroid_plane_rotation(
            trajectory_samples, ridge_relative=s.plane_rotation_ridge)
        self.pls_export_rotation = np.eye(s.pls_components)
        self.pls_export_rotation[:3, :3] = self.slice_rotation
        fit_rows[surface_columns] = self.fit_pls_unrotated[:, :3] @ self.slice_rotation.T
        pls_centroids[surface_columns] = pls_centroids[surface_columns].to_numpy() @ self.slice_rotation.T
        self.pls_centroids = pls_centroids
        self.file_splines = {rel: BSpline(spline.t, spline.c @ self.slice_rotation.T, spline.k,
                                          extrapolate=spline.extrapolate)
                             for rel, spline in file_splines.items()}
        # Fold the same rotation into raw-activation projection for stated/new rows.
        self.pls_rotations = self.pls.base_rotations.copy()
        self.pls_rotations[:, :3] = self.pls.base_rotations[:, :3] @ self.slice_rotation.T
        return (pd.Series(self.rotation_diagnostics, name='Centroid-plane rotation'),
                pls_centroids[['file', STAKES, 'spline_parameter', 'spline_residual']])

    def fit_surface(self):
        """Fit the cubic height graph; the origin is the equal-file first-class PLS1 centroid."""
        surface_points = self.fit_rows[self.surface_columns].to_numpy(dtype=np.float64)
        origin_centroids = self.pls_centroids.loc[
            self.pls_centroids[STAKES].eq(self.stakes_order[0]), 'PLS1']
        if len(origin_centroids) != len(self.inventory.fit_files):
            raise ValueError('Every fitting file must contribute the origin stakes class.')
        self.longitudinal_origin = float(origin_centroids.mean())
        self.surface = SliceSurface.fit(surface_points, self.longitudinal_origin, self.settings.slices)
        fit_residual = self.surface.evaluate(surface_points[:, [0, 2]])[:, 1] - surface_points[:, 1]
        return dict(rows=len(surface_points), design_rank=self.surface.fit_rank,
                    condition=self.surface.fit_condition,
                    pls2_residual_rms=float(np.sqrt(np.mean(fit_residual ** 2))),
                    longitudinal_origin_pls1=self.longitudinal_origin)

    def project_stated_rows(self):
        """Apply frozen BCPC and PLS transforms to stated caches (zero fitting weight)."""
        inventory, bcpc = self.inventory, self.bcpc
        all_score_frames = [self.fit_rows]
        for rel in inventory.stated_files:
            frame = inventory.entries[rel]['frame'].copy()
            bcpc_values = np.empty((len(frame), len(bcpc.score_columns)), dtype=np.float64)
            pls_values = np.empty((len(frame), self.settings.pls_components), dtype=np.float64)
            for start, stop, block in inventory.batches(rel):
                bcpc_values[start:stop] = bcpc.transform(block)
                pls_values[start:stop] = (block - self.pls.x_mean) @ self.pls_rotations
            if not np.isfinite(bcpc_values).all() or not np.isfinite(pls_values).all():
                raise RuntimeError(f'Nonfinite projected scores: {rel}')
            frame[bcpc.score_columns] = bcpc_values
            frame = bcpc.add_arc_length_columns(frame, bcpc_values)
            frame[self.pls_columns] = pls_values
            frame['log10_time_horizon_months'] = log10_time_horizon_months(frame)
            frame['pls_fit_weight'] = 0.0
            frame['file'] = PurePosixPath(rel).stem
            all_score_frames.append(frame)
        all_rows = pd.concat(all_score_frames, ignore_index=True)
        self._check_identity(all_rows, 'Projected rows do not match the source inventory.')
        if (set(map(tuple, all_rows[['source_file', 'source_row']].to_numpy()))
                != set(map(tuple, inventory.all_keys.to_numpy()))):
            raise RuntimeError('Projected source keys do not match the inventory.')
        self.all_rows = all_rows
        return all_rows.groupby('horizon_type').size().rename('rows').to_frame()

    def build_slice_cache(self):
        """Build height coverage from every row's PLS3 (no refit) and validate a sample."""
        s = self.settings
        started = time.perf_counter()
        self.coordinates = self.surface.build_cache(self.all_rows['PLS3'].to_numpy(dtype=np.float64))
        cache_bytes = sum(a.nbytes for a in self.coordinates.arrays().values())
        print(f'{len(self.coordinates.heights):,} slices; cache {cache_bytes / 2**20:.2f} MiB; '
              f'build {time.perf_counter() - started:.2f}s')
        validation_rows = pd.concat([
            group.sample(min(len(group), s.validation_rows_per_type), random_state=s.validation_seed)
            for _, group in self.all_rows.groupby('horizon_type', sort=False)])
        self.validation_report = validate_slice_mapping(
            self.coordinates, validation_rows[self.surface_columns].to_numpy())
        return pd.Series(self.validation_report, name='slice validation')

    def map_coordinates(self):
        """Nearest-height slice projection of every row, with resumable checkpoints."""
        s = self.settings
        mapped = self.coordinates.map_points(
            self.all_rows[self.surface_columns].to_numpy(dtype=np.float64),
            batch_rows=s.mapping_batch_rows,
            checkpoint_dir=self.output_dir / 'slice_mapping_checkpoints',
            resume=s.mapping_resume, progress_seconds=s.mapping_progress_seconds,
        )
        all_rows = pd.concat([
            self.all_rows.drop(columns=[*MAPPING_COLUMNS, *LEGACY_COLUMNS], errors='ignore'),
            pd.DataFrame(mapped, index=self.all_rows.index),
        ], axis=1)
        if not np.isfinite(all_rows[COORDINATE_COLUMNS].to_numpy()).all():
            raise RuntimeError('Finite-coordinate contract violated; export aborted.')
        self._check_identity(all_rows, 'Mapping changed row count or source identity.')
        self.all_rows = all_rows
        return dict(
            status=all_rows.groupby(['horizon_type', 'coordinate_status']).size().rename('rows').to_frame(),
            flags=all_rows.groupby('horizon_type')[['surface_extended', 'height_extrapolated']].sum(),
            residual=all_rows.groupby('horizon_type')['surface_projection_residual'].agg(['count', 'mean', 'max']),
        )

    def _check_identity(self, all_rows, message):
        if (len(all_rows) != len(self.inventory.all_keys)
                or all_rows[['source_file', 'source_row']].duplicated().any()):
            raise RuntimeError(message)

    def export(self, notebook=None):
        """Write rows.parquet, model.npz and model.json atomically; return the output folder."""
        all_rows, bcpc, pls, s, c = self.all_rows, self.bcpc, self.pls, self.settings, self.config
        if not np.isfinite(all_rows[COORDINATE_COLUMNS].to_numpy()).all():
            raise RuntimeError('Nonfinite coordinates; export aborted.')
        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        rotation = self.pls_export_rotation
        projection = bcpc.projection
        model_arrays = dict(
            bcpc_mean=projection['mean'], bcpc_components=projection['components'],
            bcpc_eigenvalues=projection['eigenvalues'],
            bcpc_classes=np.asarray(projection['classes'], dtype=str),
            bcpc_class_mass=projection['class_mass'], bcpc_class_means=projection['class_means'],
            bcpc_centroids=bcpc.centroids[bcpc.score_columns].to_numpy(),
            bcpc_centroid_classes=np.asarray(bcpc.centroids.index, dtype=str),
            bcpc_anchors=bcpc.anchors[bcpc.score_columns].to_numpy(),
            bcpc_anchor_names=np.asarray(bcpc.anchors.index, dtype=str),
            pls_mean=pls.x_mean, pls_target_mean=np.asarray(pls.y_mean), pls_rotations=self.pls_rotations,
            pls_base_rotations=pls.base_rotations, slice_rotation=self.slice_rotation,
            pls_export_rotation=rotation,
            pls_weights=pls.weights @ rotation.T, pls_loadings=pls.loadings @ rotation.T,
            pls_target_loadings=rotation @ pls.target_loadings,
            pls_component_variances=np.diag(rotation @ pls.score_covariance @ rotation.T),
            stakes_order=np.asarray(self.stakes_order, dtype=str),
            bcpc_component_names=np.asarray(bcpc.score_columns, dtype=str),
            pls_component_names=np.asarray(self.pls_columns, dtype=str),
            surface_component_names=np.asarray(self.surface_columns, dtype=str),
        )
        model_arrays.update(spline_arrays('bcpc', bcpc.curve))
        model_arrays.update(self.coordinates.arrays())
        for name, indices in bcpc.anchor_indices.items():
            model_arrays[f'bcpc_{name}_row_indices'] = indices
        file_spline_keys = {}
        for i, (rel, spline) in enumerate(self.file_splines.items()):
            prefix = f'file_spline_{i}'
            file_spline_keys[rel] = prefix
            model_arrays.update(spline_arrays(prefix, spline))
        b = s.bcpc
        model_metadata = dict(
            schema_version=2, notebook=notebook, pipeline='scripts/stakes_surface_pipeline.py',
            geometry_type='pls2_cubic_height_slices',
            model_name=c.model_name, layer_component=c.layer_component, position=c.position,
            feature_count=self.inventory.feature_count,
            fitting_horizon_type='horizon_free', target=TARGET,
            bcpc_weighting='equal_source_file', pls_weighting=PLS_WEIGHTING,
            pls_weight_group='source_file (one template cache per group)',
            pls_group_counts={str(key): int(value) for key, value in self.group_counts.items()},
            surface_weighting='equal_row', origin_weighting='equal_file_first_stakes_class_centroid',
            bcpc_settings=dict(components=b.n_components, centroid_method=b.centroid_method,
                anchor_weight=b.anchor_weight, internal_knots=b.internal_knots,
                end_curvature_penalty=b.end_curvature_penalty, degree=b.degree,
                zero_anchor=bcpc.zero_anchor, reverse_arc_length=bcpc.reverse_arc_length,
                total_arc_length=bcpc.total_arc_length),
            pls_settings=dict(components=s.pls_components, centered=True, standardized=False,
                spline_degree=s.pls_spline_degree, spline_interior_knots=s.pls_spline_interior_knots),
            pls_explained=dict(frame='unrotated PLS components (the PLS1-3 rotation preserves subspace totals)',
                total_feature_variance=self.pls.total_x_variance,
                x_variance_share=[float(v) for v in self.pls.x_variance_share],
                cumulative_x_variance_share=[float(v) for v in np.cumsum(self.pls.x_variance_share)],
                cumulative_weighted_training_r2=[float(v) for v in self.pls.cumulative_r2]),
            slice_config=asdict(s.slices), stakes_merges=c.stakes_merges,
            score_frame='centroid-plane rotation of PLS1-3; PLS4-16 unrotated; no rescaling',
            surface_components=list(SURFACE_COMPONENTS),
            rotation_settings=dict(method='between-file / within-curve generalized eigenvector',
                fitting_rows='horizon_free', weighting='equal_file_and_matched_stakes_progress',
                samples_per_file=s.arc_samples, **self.rotation_diagnostics),
            validation=self.validation_report,
            file_spline_keys=file_spline_keys,
            coordinate_units='unscaled 3D PLS scores',
            coordinate_meaning=dict(surface_u='projected PLS1', surface_v='snapped PLS3',
                arc_length_parallel='signed slice length from saved PLS1 origin',
                arc_length_orthogonal='PLS3 height relative to zero; not orthogonal geodesic length'),
            extensions='infinite endpoint tangent rays in PLS1; polynomial extrapolation in PLS3',
            numerical_search='all real stationary roots plus endpoints and tangent rays; flagged finite fallback',
            sources=[dict(source_file=rel, size=entry['signature'][0],
                          mtime_ns=entry['signature'][1], rows=len(entry['frame']),
                          horizon_type=entry['horizon_type'])
                     for rel, entry in self.inventory.entries.items()],
            geometry_source_sha256=hashlib.sha256(
                (ROOT / 'scripts/stakes_height_slices.py').read_bytes()).hexdigest(),
            runtime_versions=runtime_versions(),
        )
        self.inventory.check_unchanged()
        with (output_dir / 'model.npz.tmp').open('wb') as stream:
            np.savez_compressed(stream, **model_arrays)
        os.replace(output_dir / 'model.npz.tmp', output_dir / 'model.npz')
        model_metadata['model_sha256'] = hashlib.sha256((output_dir / 'model.npz').read_bytes()).hexdigest()
        (output_dir / 'model.json.tmp').write_text(json.dumps(model_metadata, indent=2), encoding='utf-8')
        os.replace(output_dir / 'model.json.tmp', output_dir / 'model.json')
        all_rows.to_parquet(output_dir / 'rows.parquet.tmp', engine='pyarrow', index=False)
        os.replace(output_dir / 'rows.parquet.tmp', output_dir / 'rows.parquet')
        print(f'Saved {len(all_rows):,} rows and model bundle to {output_dir}')
        return output_dir

    def _plot_rows(self, max_rows_per_type, seed):
        """Sample rows per horizon type for display only; the latest stage's rows are used."""
        if max_rows_per_type < 1:
            raise ValueError('max_rows_per_type must be positive.')
        rows = next(frame for frame in (getattr(self, 'all_rows', None), getattr(self, 'fit_rows', None),
                                        self.projected_rows) if frame is not None)
        return pd.concat([group.sample(min(len(group), max_rows_per_type), random_state=seed)
                          for _, group in rows.groupby('horizon_type', sort=False)])

    def _class_order(self, rows):
        """Merged stakes classes by mean horizon-free arc length, as in `rotate_plane`."""
        order = self.projected_rows.groupby(STAKES)[TARGET].mean().sort_values(kind='stable').index.tolist()
        return order + sorted(set(rows[STAKES]) - set(order))

    def _add_class_scatter(self, fig, rows, columns, order, colors):
        import plotly.graph_objects as go

        for name in order:
            group = rows.loc[rows[STAKES].eq(name)]
            if group.empty:
                continue
            # Legend-only trace: the translucent scatter would render a washed-out swatch.
            fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', name=name,
                                       legendgroup=name, marker=dict(size=8, color=colors[name])))
            fig.add_trace(go.Scatter3d(
                x=group[columns[0]], y=group[columns[1]], z=group[columns[2]], mode='markers',
                name=name, legendgroup=name, showlegend=False, opacity=0.35,
                marker=dict(size=2, color=colors[name],
                            symbol=np.where(group['horizon_type'].eq('stated'), 'diamond', 'circle')),
                customdata=group[['source_file', 'source_row', 'horizon_type', TARGET]].to_numpy(),
                hovertemplate=(f'<b>{name}</b><br>%{{customdata[0]}} row %{{customdata[1]}}'
                               f'<br>%{{customdata[2]}}<br>bcpc_arc_length %{{customdata[3]:.3g}}'
                               f'<extra></extra>')))

    def plot_bcpc(self, max_rows_per_type=20_000, seed=42):
        """BCPC1-3 rows by merged stakes class, class medians, anchors, and the fitted spline."""
        import plotly.graph_objects as go

        bcpc = self.bcpc
        columns = bcpc.score_columns[:3]
        rows = self._plot_rows(max_rows_per_type, seed)
        order = self._class_order(rows)
        colors = stakes_colors(order)
        fig = go.Figure()
        self._add_class_scatter(fig, rows, columns, order, colors)
        centroids = bcpc.centroids.loc[[name for name in order if name in bcpc.centroids.index]]
        fig.add_trace(go.Scatter3d(
            x=centroids[columns[0]], y=centroids[columns[1]], z=centroids[columns[2]],
            mode='markers+text', name='Class medians', text=centroids.index.tolist(),
            textposition='top center', textfont=dict(color=INK, size=12),
            marker=dict(size=8, color=[colors[name] for name in centroids.index],
                        line=dict(color=INK, width=2)),
            hovertemplate='<b>%{text}</b> median<br>BCPC1 %{x:.3g}<br>BCPC2 %{y:.3g}'
                          '<br>BCPC3 %{z:.3g}<extra></extra>'))
        fig.add_trace(go.Scatter3d(
            x=bcpc.anchors[columns[0]], y=bcpc.anchors[columns[1]], z=bcpc.anchors[columns[2]],
            mode='markers+text', name='Anchors', text=bcpc.anchors.index.tolist(),
            textposition='bottom center', textfont=dict(color=INK_SECONDARY, size=11),
            marker=dict(size=6, symbol='diamond', color=MUTED, line=dict(color=INK, width=1)),
            hovertemplate='<b>%{text}</b><br>BCPC1 %{x:.3g}<br>BCPC2 %{y:.3g}'
                          '<br>BCPC3 %{z:.3g}<extra></extra>'))
        curve = bcpc.spline_curve
        fig.add_trace(go.Scatter3d(
            x=curve[columns[0]], y=curve[columns[1]], z=curve[columns[2]], mode='lines',
            name='BCPC spline', line=dict(color=INK, width=5),
            customdata=curve['bcpc_arc_length'].to_numpy(),
            hovertemplate='BCPC spline<br>arc length %{customdata:.3g}<extra></extra>'))
        return style_figure(fig, f'BCPC1-3: stakes classes, medians, and spline '
                                 f'(s=0 at {bcpc.zero_anchor})', columns)

    def plot_pls_centroids(self, max_rows_per_type=20_000, seed=42):
        """Rotated PLS1-3 rows by stakes class, per-file class means, and per-file splines."""
        import plotly.graph_objects as go

        columns = self.surface_columns
        rows = self._plot_rows(max_rows_per_type, seed)
        order = self._class_order(rows)
        colors = stakes_colors(order)
        fig = go.Figure()
        self._add_class_scatter(fig, rows, columns, order, colors)
        centroids = self.pls_centroids
        fig.add_trace(go.Scatter3d(
            x=centroids[columns[0]], y=centroids[columns[1]], z=centroids[columns[2]],
            mode='markers', name='File class means',
            marker=dict(size=6, color=centroids[STAKES].map(colors).tolist(),
                        line=dict(color=INK, width=1)),
            customdata=centroids[['file', STAKES]].to_numpy(),
            hovertemplate='<b>%{customdata[1]}</b> mean<br>%{customdata[0]}<extra></extra>'))
        parameters = np.linspace(0, 1, self.settings.arc_samples)
        lines, labels = [], []
        for rel, spline in self.file_splines.items():
            # One trace with gaps keeps many files to a single legend entry.
            lines.extend([spline(parameters), np.full((1, 3), np.nan)])
            labels.extend([PurePosixPath(rel).stem] * (len(parameters) + 1))
        line = np.vstack(lines)
        fig.add_trace(go.Scatter3d(
            x=line[:, 0], y=line[:, 1], z=line[:, 2], mode='lines', name='Per-file splines',
            line=dict(color=INK_SECONDARY, width=3), text=labels,
            hovertemplate='%{text}<extra>spline</extra>', connectgaps=False))
        return style_figure(fig, 'Rotated PLS1-3: stakes classes, per-file class means, and splines',
                            columns)

    def plot_histogram(self, column, title, bins=150):
        """Histogram of every row (no sampling) along one arc-length coordinate."""
        import plotly.graph_objects as go

        values = self.all_rows[column].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(f'No finite values in {column}.')
        low, high = float(values.min()), float(values.max())
        # Explicit edges give exactly `bins` bins; plotly's nbinsx is only a hint.
        fig = go.Figure(go.Histogram(
            x=values, xbins=dict(start=low, end=high, size=(high - low) / bins or 1.0),
            name='Rows', marker=dict(color='#2a78d6', line=dict(color=SURFACE_COLOR, width=0.5)),
            hovertemplate=f'{column} %{{x}}<br>%{{y:,}} rows<extra></extra>'))
        style_figure(fig, f'{title} ({values.size:,} rows, {bins} bins)', ('', '', ''))
        axis = dict(gridcolor=GRID, zerolinecolor=BASELINE, color=INK_SECONDARY)
        fig.update_layout(height=520, showlegend=False, bargap=0,
                          xaxis=dict(title=column, **axis), yaxis=dict(title='rows', **axis))
        return fig

    def plot(self, max_rows_per_type=20_000, seed=42, grid_points=65, margin=0.10):
        """Plotly graph of the surface, representative slices, and per-file splines."""
        import plotly.graph_objects as go

        if grid_points < 2 or margin < 0:
            raise ValueError('Invalid plotting controls.')
        all_rows, surface, coordinates = self.all_rows, self.surface, self.coordinates
        plot_rows = self._plot_rows(max_rows_per_type, seed)
        fig = go.Figure(go.Scatter3d(
            x=plot_rows['PLS1'], y=plot_rows['PLS2'], z=plot_rows['PLS3'], mode='markers',
            name='Rows', opacity=0.35,
            marker=dict(size=2, color=plot_rows[TARGET], colorscale=SEQUENTIAL_SCALE,
                        colorbar=dict(title=TARGET, thickness=12),
                        symbol=np.where(plot_rows['horizon_type'].eq('stated'), 'diamond', 'circle')),
            customdata=plot_rows[['source_file', 'source_row', STAKES, 'coordinate_status',
                                  'surface_extended', 'height_extrapolated', TARGET]].to_numpy(),
            hovertemplate=('<b>%{customdata[2]}</b><br>%{customdata[0]} row %{customdata[1]}'
                           '<br>bcpc_arc_length %{customdata[6]:.3g}<br>status %{customdata[3]}'
                           '<br>extended %{customdata[4]}, extrapolated %{customdata[5]}'
                           '<extra></extra>')))
        xlow = min(surface.x_bounds[0], all_rows.surface_u.min())
        xhigh = max(surface.x_bounds[1], all_rows.surface_u.max())
        pad = (xhigh - xlow) * margin
        xgrid = np.linspace(xlow - pad, xhigh + pad, grid_points)
        zgrid = np.linspace(coordinates.heights[0], coordinates.heights[-1], grid_points)
        u, v = np.meshgrid(xgrid, zgrid)
        mesh = surface.evaluate(np.column_stack([u.ravel(), v.ravel()])).reshape(grid_points, grid_points, 3)
        fig.add_trace(go.Surface(x=mesh[:, :, 0], y=mesh[:, :, 1], z=mesh[:, :, 2],
            opacity=0.35, showscale=False, colorscale=[[0, BASELINE], [1, BASELINE]],
            name='Graph with tangent extensions', showlegend=True, hoverinfo='skip'))
        for n, i in enumerate(np.unique(np.linspace(0, len(coordinates.heights) - 1, 9).astype(int))):
            line = surface.evaluate(np.column_stack([xgrid, np.full(len(xgrid), coordinates.heights[i])]))
            fig.add_trace(go.Scatter3d(x=line[:, 0], y=line[:, 1], z=line[:, 2], mode='lines',
                name='Height slices', legendgroup='slices', showlegend=n == 0,
                line=dict(color=INK_SECONDARY, width=3),
                hovertemplate=f'Slice {i}<br>PLS3 {coordinates.heights[i]:.3g}<extra></extra>'))
        for n, (rel, spline) in enumerate(self.file_splines.items()):
            line = spline(np.linspace(0, 1, self.settings.arc_samples))
            fig.add_trace(go.Scatter3d(x=line[:, 0], y=line[:, 1], z=line[:, 2], mode='lines',
                name='Per-file splines', legendgroup='files', showlegend=n == 0,
                line=dict(color=INK, width=4),
                hovertemplate=f'{PurePosixPath(rel).stem}<extra>spline</extra>'))
        return style_figure(fig, 'PLS2 = f(PLS1, PLS3) and height slices', self.surface_columns)

    def save_plots(self, max_rows_per_type=20_000, seed=42):
        """Write every diagnostic figure as a self-contained HTML file; return {name: figure}."""
        plots_dir = self.config.plots_dir
        plots_dir.mkdir(parents=True, exist_ok=True)
        figures = {
            'bcpc_centroid_spline': self.plot_bcpc(max_rows_per_type, seed),
            'pls_centroid_splines': self.plot_pls_centroids(max_rows_per_type, seed),
            'pls_surface': self.plot(max_rows_per_type, seed),
            'bcpc_arc_length_histogram': self.plot_histogram(
                TARGET, 'Rows along the BCPC spline'),
            'arc_length_parallel_histogram': self.plot_histogram(
                'arc_length_parallel', 'Rows along PLS1-3 surface slices'),
        }
        for name, fig in figures.items():
            path = plots_dir / f'{name}.html'
            temporary = plots_dir / f'{name}.html.tmp'
            # Embed plotly.js so the files open offline.
            fig.write_html(temporary, include_plotlyjs=True, full_html=True)
            os.replace(temporary, path)
            print(f'Saved {path}')
        return figures


def stakes_colors(order):
    """Ordered classes spread evenly over the ordinal ramp, lightest for the lowest stakes."""
    if len(order) > len(STAKES_RAMP):
        raise ValueError(f'At most {len(STAKES_RAMP)} stakes classes can be colored.')
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


def runtime_versions():
    import scipy
    import torch
    return dict(numpy=np.__version__, pandas=pd.__version__,
                scipy=scipy.__version__, torch=str(torch.__version__))
