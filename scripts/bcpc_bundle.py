"""Save and load the full-fit BCPC and arc-length spline that the pipeline uses.

One folder per model, `RunConfig.bcpc_dir` (`artifacts/<model>/bcpc/`):

- `model.npz`     BCPC mean, components, eigenvalues and class masses, the very_low ->
                  existential direction, the spline's class centres, end anchors, knots and
                  coefficients, and each class centre's refined stakes
- `model.json`    settings, orientation, the refined-stakes scale, cross-validation
                  summaries, the fitted cache files, code and file checksums
- `rows.parquet`  every horizon-free row scored by the full fit alone: BCPC scores,
                  `bcpc_arc_length`, `refined_stakes`, `refined_residual`, and `cv_fold`
- `plot_data/`    what the out-of-fold plots draw: `scores.parquet` (BCPC1-3 of the full
                  fit and of every fold's held-out rows), `anchors.parquet`,
                  `retention.parquet` and `refinement.parquet` (per-row held-out values)

Every row's arc length and refined stakes come from the one fit on all rows, so they share
one scale. The fold fits are diagnostics: only the plots and the cross-validation summaries
use them. Apply the saved model to new activations with `score_activations`.
"""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import subprocess

import numpy as np
import pandas as pd

from . import bcpc_cross_validation as cv
from .bcpc_arc_length import BcpcArcLength, BcpcSettings
from .cache_inventory import cache_signature
from .pipeline_config import ROOT

SCHEMA_VERSION = 1
ANCHOR_NAMES = ['negative_anchor', 'positive_anchor']
PLOT_DATA = ('scores', 'anchors', 'retention', 'refinement')
CODE_FILES = ('scripts/bcpc_bundle.py', 'scripts/bcpc_cross_validation.py',
              'scripts/bcpc_arc_length.py')


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, writer):
    """Write through a temporary file so a reader never sees a partial file."""
    temporary = path.with_name(path.name + '.tmp')
    writer(temporary)
    os.replace(temporary, path)


def _plain(value):
    """JSON-safe floats: NaN becomes null."""
    value = float(value)
    return None if np.isnan(value) else value


def _git_state():
    def git(*args):
        return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    try:
        return dict(commit=git('rev-parse', 'HEAD'),
                    uncommitted_code=bool(git('status', '--porcelain', '--', *CODE_FILES)))
    except (OSError, subprocess.CalledProcessError):
        return dict(commit=None, uncommitted_code=None)


def _runtime_versions():
    import pyarrow
    import scipy
    return dict(numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                pyarrow=pyarrow.__version__)


def _check_sources(sources):
    """Raise if any fitted cache changed since it was read."""
    changed = [s['source_file'] for s in sources
               if cache_signature(Path(s['path'])) != (s['size'], s['mtime_ns'])]
    if changed:
        raise RuntimeError(f'Caches changed after fitting: {changed}')


def _cross_validation_summary(result):
    retention = result.retention.groupby('dimensions')[['bcpc', 'difference_of_means']].agg(['mean', 'std'])
    return dict(
        **result.settings, held_out_unit='task', stratified_by='stakes',
        role='diagnostics only; no coordinate in rows.parquet comes from a fold fit',
        retained_geometry={int(k): {f'{a}_{b}': _plain(v) for (a, b), v in row.items()}
                           for k, row in retention.iterrows()},
        stability={key: _plain(v) for key, v in result.stability.items()},
        refinement={key: _plain(v) for key, v in cv.refinement_summary(result.refinement).items()})


def save_bcpc_bundle(result, directory):
    """Write the full fit, its row scores and the plot data of one model's cross-validation.

    The saved model is reloaded and must reproduce every row's arc length before this returns.
    """
    full, curve, provenance = result.full, result.full.curve, result.provenance
    if full is None or provenance is None:
        raise ValueError('The result lacks its full fit or provenance; rerun cv.cross_validate.')
    _check_sources(provenance['sources'])
    directory = Path(directory)
    (directory / 'plot_data').mkdir(parents=True, exist_ok=True)

    arrays = dict(
        bcpc_mean=full.bcpc.mean, bcpc_components=full.bcpc.components,
        bcpc_eigenvalues=full.bcpc.eigenvalues, class_mass=full.bcpc.class_mass,
        difference_of_means_direction=full.bcpc.dom_direction, class_refined=full.class_refined,
        spline_centroids=curve.centroids.loc[list(cv.LEVELS), curve.score_columns].to_numpy(),
        spline_anchors=curve.anchors.loc[ANCHOR_NAMES, curve.score_columns].to_numpy(),
        spline_knots=curve.curve.t, spline_coefficients=curve.curve.c)

    def write_npz(path):
        with path.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
    _write(directory / 'model.npz', write_npz)
    _write(directory / 'rows.parquet',
           lambda path: result.full_rows.to_parquet(path, engine='pyarrow', index=False))
    plot_frames = dict(scores=result.scores, anchors=result.anchors, retention=result.retention,
                       refinement=result.refinement)
    for name, frame in plot_frames.items():
        _write(directory / 'plot_data' / f'{name}.parquet',
               lambda path, frame=frame: frame.to_parquet(path, engine='pyarrow', index=False))

    files = ['model.npz', 'rows.parquet', *(f'plot_data/{name}.parquet' for name in PLOT_DATA)]
    metadata = dict(
        schema_version=SCHEMA_VERSION,
        created_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        **{key: provenance[key] for key in ('model_name', 'model_slug', 'layer_component',
                                            'position', 'rows', 'features')},
        fitting_rows='horizon_free', excluded_templates=list(cv.EXCLUDED_TEMPLATES),
        classes=list(cv.LEVELS), stakes_merges=cv.STAKES_MERGES,
        positive_anchor_class=cv.POSITIVE_ANCHOR, negative_anchor_class=cv.NEGATIVE_ANCHOR,
        weighting='equal total weight per template, split equally among its rows',
        bcpc=dict(n_components=int(full.bcpc.components.shape[1]),
                  eigenvalues=[float(v) for v in full.bcpc.eigenvalues],
                  orientation=f'each direction signed so the {cv.POSITIVE_ANCHOR} centroid scores positive',
                  transform="(X - bcpc_mean) @ bcpc_components"),
        spline=dict(settings=asdict(curve.settings), score_columns=curve.score_columns,
                    anchor_rows=ANCHOR_NAMES, zero_anchor=curve.zero_anchor,
                    reverse_arc_length=bool(curve.reverse_arc_length),
                    total_arc_length=float(curve.total_arc_length)),
        refined_stakes=dict(
            formula=f'1 + {cv.N_CLASSES - 1} * (bcpc_arc_length - arc_zero) / arc_scale',
            meaning=f'{cv.NEGATIVE_ANCHOR} class centre = 1, {cv.POSITIVE_ANCHOR} centre = '
                    f'{cv.N_CLASSES}, one unit per ladder step',
            arc_zero=float(full.arc_zero), arc_scale=float(full.arc_scale),
            class_centres={level: float(v) for level, v in zip(cv.LEVELS, full.class_refined)},
            residual='refined_stakes minus the refined stakes of the row\'s labelled class centre'),
        cross_validation=_cross_validation_summary(result),
        sources=[{key: s[key] for key in ('source_file', 'size', 'mtime_ns', 'rows')}
                 for s in provenance['sources']],
        files={name: _sha256(directory / name) for name in files},
        code=dict(**_git_state(), sha256={name: _sha256(ROOT / name) for name in CODE_FILES}),
        runtime_versions=_runtime_versions())
    _write(directory / 'model.json',
           lambda path: path.write_text(json.dumps(metadata, indent=2), encoding='utf-8'))

    fit, _ = load_bcpc_bundle(directory)
    saved = pd.read_parquet(directory / 'rows.parquet')
    arc = fit.curve.batch_arc_length(saved[fit.curve.score_columns].to_numpy())
    if not (np.allclose(arc, saved['bcpc_arc_length'], rtol=0, atol=1e-9)
            and np.allclose(fit.refined(arc), saved['refined_stakes'], rtol=0, atol=1e-9)):
        raise RuntimeError(f'The reloaded model does not reproduce the saved arc lengths in {directory}.')
    _check_sources(provenance['sources'])
    return directory


def load_bcpc_bundle(directory):
    """(fit, metadata): the saved full fit as a `cv.FoldFit`, without refitting.

    `fit.bcpc.mean` and `fit.bcpc.components` project activations, `fit.curve` gives arc
    length and `fit.refined` rescales it; `score_activations` does all three.
    """
    directory = Path(directory)
    metadata = json.loads((directory / 'model.json').read_text(encoding='utf-8'))
    if metadata.get('schema_version') != SCHEMA_VERSION:
        raise ValueError(f'Unsupported BCPC bundle version in {directory}.')
    if _sha256(directory / 'model.npz') != metadata['files']['model.npz']:
        raise ValueError(f'{directory / "model.npz"} does not match its metadata checksum.')
    if metadata['classes'] != list(cv.LEVELS):
        raise ValueError(f'The bundle classes {metadata["classes"]} differ from {list(cv.LEVELS)}.')
    with np.load(directory / 'model.npz', allow_pickle=False) as archive:
        a = {key: archive[key] for key in archive.files}

    columns = metadata['spline']['score_columns']
    bcpc = cv.BcpcFit(a['bcpc_mean'], a['bcpc_components'], a['bcpc_eigenvalues'],
                      a['class_mass'], a['difference_of_means_direction'])
    projection = dict(mean=bcpc.mean, components=bcpc.components, eigenvalues=bcpc.eigenvalues,
                      classes=np.array(cv.LEVELS), class_mass=bcpc.class_mass)
    curve = BcpcArcLength.restore(
        BcpcSettings(**metadata['spline']['settings']), projection,
        pd.DataFrame(a['spline_centroids'], index=pd.Index(cv.LEVELS, name='stakes'), columns=columns),
        pd.DataFrame(a['spline_anchors'], index=pd.Index(ANCHOR_NAMES, name='anchor'), columns=columns),
        a['spline_knots'], a['spline_coefficients'])
    if (curve.zero_anchor != metadata['spline']['zero_anchor']
            or not np.isclose(curve.total_arc_length, metadata['spline']['total_arc_length'],
                              rtol=1e-12, atol=0)):
        raise RuntimeError(f'The spline rebuilt from {directory} differs from the one saved.')
    scale = metadata['refined_stakes']
    return cv.FoldFit(bcpc, curve, scale['arc_zero'], scale['arc_scale'], a['class_refined']), metadata


def score_activations(fit, X):
    """BCPC scores, arc length and refined stakes of activations X under a saved fit."""
    scores = (np.asarray(X, dtype=np.float64) - fit.bcpc.mean) @ fit.bcpc.components
    arc = fit.curve.batch_arc_length(scores)
    return pd.DataFrame(scores, columns=fit.curve.score_columns).assign(
        bcpc_arc_length=arc, refined_stakes=fit.refined(arc))


def load_bcpc_result(directory):
    """A `cv.ModelResult` rebuilt from the saved files, enough to redraw every plot."""
    directory = Path(directory)
    fit, metadata = load_bcpc_bundle(directory)
    for name in ('rows.parquet', *(f'plot_data/{n}.parquet' for n in PLOT_DATA)):
        if _sha256(directory / name) != metadata['files'][name]:
            raise ValueError(f'{directory / name} does not match its metadata checksum.')
    rows = pd.read_parquet(directory / 'rows.parquet')
    frames = {name: pd.read_parquet(directory / 'plot_data' / f'{name}.parquet') for name in PLOT_DATA}
    summary = metadata['cross_validation']
    return cv.ModelResult(
        rows, frames['scores'], frames['retention'], frames['refinement'],
        pd.Series(summary['stability']), fit.curve, frames['anchors'], fit, rows,
        dict(metadata, run_dir=str(directory.parent)),
        {key: summary[key] for key in ('n_folds', 'seed', 'n_components')})


PLOTS = {
    'bcpc_scores_3d': lambda results, max_rows, seed: cv.plot_scores(results, max_rows, seed),
    'bcpc_retained_geometry': lambda results, max_rows, seed: cv.plot_retention(results),
    'bcpc_refined_stakes_held_out': lambda results, max_rows, seed: cv.plot_refined(results),
}


def save_bcpc_plots(directory, plots_dir, max_rows=None, seed=42):
    """Draw every plot from the saved bundle alone and write it as standalone HTML."""
    result = load_bcpc_result(directory)
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, draw in PLOTS.items():
        fig = draw({result.provenance['model_slug']: result}, max_rows, seed)
        # Embed plotly.js so the files open offline, as the surface pipeline's plots do.
        _write(plots_dir / f'{name}.html',
               lambda path: fig.write_html(path, include_plotlyjs=True, full_html=True))
        paths.append(plots_dir / f'{name}.html')
    return paths
