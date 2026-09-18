"""Out-of-fold BCPC geometry and arc-length stakes for the nine stakes classes.

Two goals are checked, neither of them class prediction:

1. Retained geometry. How much of the held-out stakes geometry a BCPC subspace fitted
   on the training rows keeps. The geometry is the between-class scatter of the held-out
   class means, estimated as the cross-product of two independent halves of the held-out
   rows, so sampling noise, which spreads over every dimension, cancels in expectation.
2. Refined stakes. Each row's arc length along the BCPC spline (`BcpcArcLength`, refitted
   in every fold), rescaled so the `very_low` class centre sits at 1 and the
   `existential` centre at 9, one unit per ladder step. A row's residual from its own class centre is the
   refinement. It is checked for pull (the residual when the row was in training against
   when it was held out) and for reproducibility across two halves of the templates.

Rows are held out by task: every fold's held-out tasks are unseen task content, while
every template is still seen in training.

The labelled stakes levels are kept as they are, with one exception: `near_existential`
is merged into `existential`, because their centroids sit much closer together than any
other pair of classes. Every BCPC direction is oriented so the `existential` centroid
scores positive, `existential` is the positive end of the refined scale and of the
difference-of-means line, and `very_low` is the negative end. Nothing is written to disk.

Every average across templates gives each dataset (register) equal total weight, split
equally among its templates, so a dataset with more templates does not dominate. This
covers the class means, the spline's class medians, the held-out geometry and the task
averages. The surface pipeline instead weights each template file equally.

Activations are held in float64, about 0.7 GB for a 5,376-wide model. The between-class
scatter is M^T M with M the K x d matrix of mass-weighted centred class means, so its
leading eigenvectors are M's right singular vectors: the same directions
`bcpc_arc_length.fit_between_class_pca` finds, without forming a d x d matrix.
"""
from dataclasses import dataclass
from pathlib import PurePosixPath

import numpy as np
import pandas as pd

from .bcpc_arc_length import BcpcArcLength, BcpcSettings
from .cache_inventory import CacheInventory
from .corpora.training.base_task_set import STAKES_LEVELS
from .pipeline_config import load_run_config

POSITIVE_ANCHOR = 'existential'
NEGATIVE_ANCHOR = 'very_low'
# The only merge: these two centroids are much closer than any other pair of classes.
STAKES_MERGES = {'near_existential': 'existential'}
LEVELS = tuple(level for level in STAKES_LEVELS if level not in STAKES_MERGES)
N_CLASSES = len(LEVELS)
CHUNK_ROWS = 1024


@dataclass
class ModelActivations:
    name: str
    rows: pd.DataFrame      # one row per activation, aligned with X
    X: np.ndarray           # (rows, features) float64


@dataclass
class BcpcFit:
    mean: np.ndarray            # weighted grand mean, (d,)
    components: np.ndarray      # (d, k), orthonormal columns
    eigenvalues: np.ndarray     # between-class variance per component, (k,)
    class_mass: np.ndarray      # training weight per class, (K,)
    dom_direction: np.ndarray   # unit very_low -> existential direction, (d,)


@dataclass
class FoldFit:
    bcpc: BcpcFit
    curve: BcpcArcLength
    arc_zero: float             # arc length of the very_low class centre
    arc_scale: float            # existential centre minus very_low centre, in arc length
    class_refined: np.ndarray   # refined stakes of each class centre, ladder order

    def refined(self, arc_length):
        """Arc length rescaled so very_low's centre is 1 and existential's is N_CLASSES."""
        return 1 + (N_CLASSES - 1) * (arc_length - self.arc_zero) / self.arc_scale


@dataclass
class ModelResult:
    rows: pd.DataFrame
    scores: pd.DataFrame        # BCPC1-3 and refined stakes, in-sample and held-out
    retention: pd.DataFrame     # retained held-out geometry per fold and dimension
    refinement: pd.DataFrame    # per-row held-out refined stakes and residuals
    stability: pd.Series
    curve: BcpcArcLength        # the in-sample spline, for plotting
    anchors: pd.DataFrame       # every fit's spline end anchors in BCPC1-3, for plotting


def load_model(run_dir):
    """Every horizon-free row of one model with merged stakes, activations in float64."""
    config = load_run_config(run_dir)
    # STAKES_MERGES replaces the run config's merges, whatever the caching pass recorded.
    inventory = CacheInventory(config.activations_dir, config.model_name, config.layer_component,
                               config.position, STAKES_MERGES, key_root=config.run_dir)
    frames = [inventory.entries[rel]['frame'] for rel in inventory.fit_files]
    rows = pd.concat(frames, ignore_index=True)
    X = np.empty((len(rows), inventory.feature_count), dtype=np.float64)
    offset = 0
    for rel, frame in zip(inventory.fit_files, frames):
        for start, stop, block in inventory.batches(rel):
            X[offset + start:offset + stop] = block
        offset += len(frame)

    unknown = set(rows['stakes_original']) - set(STAKES_LEVELS)
    if unknown:
        raise ValueError(f'Unknown stakes levels: {sorted(unknown)}')
    rows = rows[['source_file', 'source_row', 'template_id', 'task', 'stakes', 'stakes_original']].copy()
    rows['template'] = rows['source_file'].map(lambda rel: PurePosixPath(rel).stem)
    rows['register'] = rows['template'].str.split('--').str[0]
    rows['stakes_rank'] = rows['stakes'].map({level: i + 1 for i, level in enumerate(LEVELS)}).astype(int)
    rows['class_index'] = rows['stakes_rank'] - 1
    return ModelActivations(config.model_slug, rows, X)


def stratified_folds(keys, strata, n_folds, seed):
    """{key: fold}: each stratum's shuffled keys dealt round-robin, the deal running on
    across strata so fold sizes differ by at most one."""
    table = pd.DataFrame({'key': np.asarray(keys), 'stratum': np.asarray(strata)}).drop_duplicates()
    if table['key'].duplicated().any():
        raise ValueError('A fold key belongs to more than one stratum.')
    rng = np.random.default_rng(seed)
    folds, position = {}, 0
    for _, group in table.sort_values('key').groupby('stratum', sort=True):
        for key in rng.permutation(group['key'].to_numpy()):
            folds[key] = position % n_folds
            position += 1
    return folds


def task_folds(rows, n_folds, seed):
    """Fold per row, with whole tasks dealt into folds stratified by stakes level."""
    return rows['task'].map(stratified_folds(rows['task'], rows['stakes'], n_folds, seed)).to_numpy()


def fold_composition(rows, folds):
    """Held-out tasks per stakes level, by fold."""
    tasks = rows.assign(fold=folds).drop_duplicates('task')
    return pd.crosstab(tasks['stakes'], tasks['fold']).reindex(LEVELS)


def dataset_weights(frame, by=()):
    """Row weights summing to one within each `by` group, or over all rows if none: equal
    total per dataset (register), split equally among its templates present, then among
    each template's rows."""
    by = list(by)
    registers = (frame.groupby(by, sort=False)['register'].transform('nunique') if by
                 else frame['register'].nunique())
    templates = frame.groupby(by + ['register'], sort=False)['template'].transform('nunique')
    rows_per_template = frame.groupby(by + ['template'], sort=False)['template'].transform('size')
    return (1.0 / (registers * templates * rows_per_template)).to_numpy()


def _weighted_sums(values, weights, keys):
    """Per-group sums of weighted values; group means when weights sum to one per group."""
    return values.mul(weights, axis=0).groupby(keys, sort=False).sum()


def class_means(data, index, chunk_rows=CHUNK_ROWS):
    """Dataset-weighted means of every stakes class over `index`; (means, class mass)."""
    weights = dataset_weights(data.rows.iloc[index])
    class_index = data.rows['class_index'].to_numpy()[index]
    mass = np.bincount(class_index, weights=weights, minlength=N_CLASSES)
    if (mass <= 0).any():
        missing = [LEVELS[c] for c in np.flatnonzero(mass <= 0)]
        raise ValueError(f'Rows lack stakes levels {missing}.')
    sums = np.zeros((N_CLASSES, data.X.shape[1]))
    for start in range(0, len(index), chunk_rows):
        stop = min(start + chunk_rows, len(index))
        onehot = np.zeros((stop - start, N_CLASSES))
        onehot[np.arange(stop - start), class_index[start:stop]] = weights[start:stop]
        sums += onehot.T @ data.X[index[start:stop]]
    return sums / mass[:, None], mass


def fit_bcpc(data, index, n_components=None):
    """BCPC and difference-of-means direction from the rows in `index` alone."""
    n_components = N_CLASSES - 1 if n_components is None else n_components
    means, mass = class_means(data, index)
    mean = mass @ means
    centred = means - mean
    _, singular_values, vt = np.linalg.svd(np.sqrt(mass)[:, None] * centred, full_matrices=False)
    if n_components > np.sum(singular_values > singular_values[0] * 1e-10):
        raise ValueError(f'The classes support fewer than {n_components} BCPC directions.')
    components = vt[:n_components].T
    signs = np.sign(centred[LEVELS.index(POSITIVE_ANCHOR)] @ components)
    signs[signs == 0] = 1.0
    components = components * signs
    direction = (means[LEVELS.index(POSITIVE_ANCHOR)]
                 - means[LEVELS.index(NEGATIVE_ANCHOR)])
    return BcpcFit(mean, components, singular_values[:n_components] ** 2, mass,
                   direction / np.linalg.norm(direction))


def project(data, index, fit, chunk_rows=CHUNK_ROWS):
    """BCPC scores of the rows in `index`."""
    out = np.empty((len(index), fit.components.shape[1]))
    for start in range(0, len(index), chunk_rows):
        stop = min(start + chunk_rows, len(index))
        out[start:stop] = (data.X[index[start:stop]] - fit.mean) @ fit.components
    return out


def arc_length(curve, scores):
    return curve.batch_arc_length(scores)


def fit_fold(data, index, n_components=None):
    """BCPC, its arc-length spline, and the refined-stakes scale from `index` alone."""
    bcpc = fit_bcpc(data, index, n_components)
    scores = project(data, index, bcpc)
    projection = dict(mean=bcpc.mean, components=bcpc.components, eigenvalues=bcpc.eigenvalues,
                      classes=np.array(LEVELS), class_mass=bcpc.class_mass)
    curve = BcpcArcLength(BcpcSettings(n_components=scores.shape[1])).fit_curve(
        projection, scores, dataset_weights(data.rows.iloc[index]),
        data.rows['class_index'].to_numpy()[index])
    class_arc = arc_length(curve, curve.centroids.loc[list(LEVELS), curve.score_columns].to_numpy())
    zero = class_arc[LEVELS.index(NEGATIVE_ANCHOR)]
    scale = class_arc[LEVELS.index(POSITIVE_ANCHOR)] - zero
    if scale <= 0:
        raise ValueError(f'The {POSITIVE_ANCHOR} centre does not lie beyond {NEGATIVE_ANCHOR} on the arc.')
    fold = FoldFit(bcpc, curve, zero, scale, None)
    fold.class_refined = fold.refined(class_arc)
    return fold


def held_out_halves(rows, held_out, seed):
    """Split held-out rows in two by task, stratified by stakes level as the folds are."""
    held = rows.iloc[held_out]
    return held['task'].map(stratified_folds(held['task'], held['stakes'], 2, seed)).to_numpy()


def retained_geometry(data, held_out, half, fit):
    """Share of the held-out cross-half between-class scatter inside the training subspace.

    With A and B the two halves' centred class means and m the held-out class mass, the
    scatter is sum_c m_c a_c b_c^T, whose trace keeps only structure both halves share.
    Returns the share retained by BCPC1..k for every k and by the difference-of-means line,
    then the trace itself.
    """
    halves = []
    for h in (0, 1):
        means, mass = class_means(data, held_out[half == h])
        halves.append(means - mass @ means)
    A, B = halves
    _, mass = class_means(data, held_out)
    total = float(np.sum(mass * np.einsum('cd,cd->c', A, B)))
    a, b = A @ fit.components, B @ fit.components
    bcpc = np.cumsum(np.sum(mass[:, None] * a * b, axis=0)) / total
    u = fit.dom_direction
    dom = float(np.sum(mass * (A @ u) * (B @ u))) / total
    return bcpc, dom, total


def cross_validate(data, folds, seed, n_components=None):
    """In-sample fit, then every task fold, scoring all rows in every fit."""
    rows = data.rows
    everything = np.arange(len(rows))
    label = rows['class_index'].to_numpy()
    full = fit_fold(data, everything, n_components)
    scores = project(data, everything, full.bcpc)
    frames = [pd.DataFrame(scores[:, :3], columns=['BCPC1', 'BCPC2', 'BCPC3']).assign(
        refined=full.refined(arc_length(full.curve, scores)), fit='in_sample', fold=-1, row=everything)]
    anchors = [_anchor_frame(full.curve, 'in_sample', -1)]
    template_half = rows['source_file'].map(
        stratified_folds(rows['source_file'], rows['register'], 2, seed)).to_numpy()

    n_folds = folds.max() + 1
    refined = np.empty((n_folds, len(rows)))
    residual = np.empty((n_folds, len(rows)))
    retention, fold_components = [], []
    for fold in range(n_folds):
        held_out = np.flatnonzero(folds == fold)
        fit = fit_fold(data, np.flatnonzero(folds != fold), n_components)
        fold_components.append(fit.bcpc.components)
        anchors.append(_anchor_frame(fit.curve, 'held_out', fold))
        scores = project(data, everything, fit.bcpc)
        refined[fold] = fit.refined(arc_length(fit.curve, scores))
        residual[fold] = refined[fold] - fit.class_refined[label]

        bcpc, dom, total = retained_geometry(
            data, held_out, held_out_halves(rows, held_out, seed + fold), fit.bcpc)
        retention.extend(dict(fold=fold, dimensions=k + 1, bcpc=bcpc[k],
                              random=(k + 1) / data.X.shape[1],
                              difference_of_means=dom if k == 0 else np.nan,
                              held_out_scatter=total)
                         for k in range(len(bcpc)))
        frames.append(pd.DataFrame(scores[held_out, :3], columns=['BCPC1', 'BCPC2', 'BCPC3']).assign(
            refined=refined[fold, held_out], fit='held_out', fold=fold, row=held_out))

    own = folds[None, :] == np.arange(n_folds)[:, None]
    refinement = rows[['task', 'template', 'register', 'stakes', 'stakes_original', 'stakes_rank']].assign(
        refined=refined[folds, everything],
        residual=residual[folds, everything],
        residual_in_training=np.where(own, 0.0, residual).sum(axis=0) / (~own).sum(axis=0),
        template_half=template_half)
    return ModelResult(rows, pd.concat(frames, ignore_index=True), pd.DataFrame(retention),
                       refinement, component_stability(fold_components), full.curve,
                       pd.concat(anchors, ignore_index=True))


def _anchor_frame(curve, fit_label, fold):
    """A spline's two end anchors in BCPC1-3, tagged with the fit they belong to."""
    return (curve.anchors[['BCPC1', 'BCPC2', 'BCPC3']].reset_index()
            .assign(fit=fit_label, fold=fold))


def component_stability(bases, n_axes=3):
    """Mean over fold pairs of |cos| between matching components, and BCPC1-3 subspace
    overlap (squared Frobenius norm of V_f^T V_g over 3; 1 means identical subspaces)."""
    pairs = [(a, b) for i, a in enumerate(bases) for b in bases[i + 1:]]
    result = {f'BCPC{j + 1} |cos|': np.mean([abs(a[:, j] @ b[:, j]) for a, b in pairs])
              for j in range(n_axes)}
    result['BCPC1-3 subspace overlap'] = np.mean(
        [np.sum((a[:, :n_axes].T @ b[:, :n_axes]) ** 2) / n_axes for a, b in pairs])
    return pd.Series(result)


def retention_summary(retention, dimensions=(1, 2, 3, 5, N_CLASSES - 1)):
    """Mean and sd over folds of each retained share, at selected subspace sizes."""
    frame = retention.loc[retention['dimensions'].isin(dimensions)]
    columns = ['bcpc', 'difference_of_means', 'random']
    return frame.groupby('dimensions')[columns].agg(['mean', 'std'])


def task_refinement(frame):
    """Per task, dataset-weighted over its templates: held-out refined stakes, its residual
    from the class centre, the residual when the task's rows were in training, and how
    consistently templates agree on it. The standard error behind `t` uses the weights'
    effective sample size."""
    task = frame['task']
    weights = dataset_weights(frame, ['task'])
    tasks = frame.groupby('task', sort=False).agg(
        stakes=('stakes', 'first'), stakes_original=('stakes_original', 'first'),
        stakes_rank=('stakes_rank', 'first'), templates=('residual', 'size'))
    columns = ['refined', 'residual', 'residual_in_training']
    tasks[columns] = _weighted_sums(frame[columns], weights, task)
    deviation = (frame['residual'] - task.map(tasks['residual'])) ** 2
    effective_n = 1.0 / pd.Series(weights ** 2, index=frame.index).groupby(task, sort=False).sum()
    tasks['residual_sd'] = np.sqrt(_weighted_sums(deviation, weights, task)
                                   * effective_n / (effective_n - 1))
    agree = (np.sign(frame['residual']) == task.map(np.sign(tasks['residual']))).astype(float)
    tasks['same_sign_share'] = _weighted_sums(agree, weights, task)
    tasks['t'] = tasks['residual'] / (tasks['residual_sd'] / np.sqrt(effective_n))
    return tasks


def refinement_summary(frame):
    """Size, pull and reproducibility of the held-out refinements.

    - refinement_sd: spread of task residuals from their class centres, in ladder steps
    - pull_ratio: mean |residual| in training over mean |residual| held out; below 1
      means fitting on a row pulls it toward its own class centre
    - in_vs_held_out_corr: task residuals in training against held out
    - residual_half_corr / refined_half_corr: task means from one half of the templates
      against the other half, for the residual and for refined stakes itself
    """
    tasks = task_refinement(frame)
    halves = _weighted_sums(frame[['residual', 'refined']],
                            dataset_weights(frame, ['task', 'template_half']),
                            [frame['task'], frame['template_half']]).unstack()
    return pd.Series(dict(
        refinement_sd=tasks['residual'].std(),
        pull_ratio=tasks['residual_in_training'].abs().mean() / tasks['residual'].abs().mean(),
        in_vs_held_out_corr=np.corrcoef(tasks['residual_in_training'], tasks['residual'])[0, 1],
        residual_half_corr=np.corrcoef(halves[('residual', 0)], halves[('residual', 1)])[0, 1],
        refined_half_corr=np.corrcoef(halves[('refined', 0)], halves[('refined', 1)])[0, 1]))


def stable_disagreements(frame, n=10):
    """Tasks whose held-out refined stakes sit furthest from their class centre, ranked by
    the residual over its standard error across templates; n above and n below."""
    tasks = task_refinement(frame).sort_values('t')
    columns = ['stakes', 'stakes_original', 'refined', 'residual', 'residual_sd',
               'same_sign_share', 'templates']
    return pd.concat({'above its class': tasks.tail(n).iloc[::-1][columns],
                      'below its class': tasks.head(n)[columns]})


SCORE_AXES = ['BCPC1', 'BCPC2', 'BCPC3']
FIT_CAPTIONS = {'in_sample': 'in-sample fit', 'held_out': 'held-out tasks'}


def anchor_proximity(result, fit_label='held_out', levels=('very_high', 'catastrophic'),
                     reference='catastrophic'):
    """Per task labelled one of `levels`, in BCPC1-3 as one `plot_scores` panel draws them:
    the distance from the task's dataset-weighted mean to its fit's positive spline anchor,
    and to the panel's `reference` class mean. Held out, a task is placed in its own fold's
    basis against that fold's anchor, while the class mean pools every fold's held-out rows,
    as the plotted circle does."""
    frame = result.scores.loc[result.scores['fit'].eq(fit_label)].reset_index(drop=True)
    frame = frame.join(result.rows[['stakes', 'task', 'template', 'register']], on='row')
    means = _weighted_sums(frame[SCORE_AXES], dataset_weights(frame, ['stakes']), frame['stakes'])
    chosen = frame.loc[frame['stakes'].isin(levels)]
    points = _weighted_sums(chosen[SCORE_AXES], dataset_weights(chosen, ['task']), chosen['task'])
    tasks = chosen.groupby('task', sort=False)[['stakes', 'fold']].first().loc[points.index]

    anchors = result.anchors.loc[result.anchors['fit'].eq(fit_label)
                                 & result.anchors['anchor'].eq('positive_anchor')]
    anchor = anchors.set_index('fold').loc[tasks['fold'], SCORE_AXES].to_numpy()
    return pd.DataFrame({
        'stakes': tasks['stakes'],
        'to_positive_anchor': np.linalg.norm(points.to_numpy() - anchor, axis=1),
        f'to_{reference}_mean': np.linalg.norm(points.to_numpy() - means.loc[reference].to_numpy(),
                                               axis=1),
    }, index=points.index)


def _score_traces(result, fit_label, max_rows, seed, legend):
    """One BCPC1-3 panel: rows by stakes level, class means, every fit's spline anchors and,
    in-sample, the spline. Returns the traces, the rows drawn and the rows in the fit."""
    import plotly.graph_objects as go

    from .stakes_surface_pipeline import INK, stakes_colors

    frame = result.scores.loc[result.scores['fit'].eq(fit_label)].reset_index(drop=True)
    frame = frame.join(result.rows[['stakes', 'task', 'template', 'register']], on='row')
    colors = stakes_colors(LEVELS)
    means = _weighted_sums(frame[SCORE_AXES], dataset_weights(frame, ['stakes']),
                           frame['stakes']).reindex(LEVELS)
    shown = frame if max_rows is None or len(frame) <= max_rows else frame.sample(max_rows, random_state=seed)

    traces = []
    for level in LEVELS:
        group = shown.loc[shown['stakes'].eq(level)]
        if legend:
            # Legend-only trace: the translucent scatter would render a washed-out swatch.
            traces.append(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', name=level,
                                       legendgroup=level, marker=dict(size=8, color=colors[level])))
        traces.append(go.Scatter3d(
            x=group['BCPC1'], y=group['BCPC2'], z=group['BCPC3'], mode='markers', name=level,
            legendgroup=level, showlegend=False, opacity=0.35,
            marker=dict(size=2, color=colors[level]),
            customdata=group[['task', 'template', 'refined']].to_numpy(),
            hovertemplate=(f'<b>{level}</b><br>%{{customdata[0]}}<br>%{{customdata[1]}}'
                           '<br>refined stakes %{customdata[2]:.2f}<extra></extra>')))
    ends = [NEGATIVE_ANCHOR, POSITIVE_ANCHOR]
    traces.append(go.Scatter3d(
        x=means['BCPC1'], y=means['BCPC2'], z=means['BCPC3'], mode='markers+text',
        name='Class means', legendgroup='means', showlegend=legend,
        marker=dict(size=8, color=[colors[level] for level in LEVELS],
                    line=dict(color=INK, width=2)),
        text=[level if level in ends else '' for level in LEVELS],
        textposition='top center', textfont=dict(color=INK, size=12),
        customdata=np.array(LEVELS)[:, None],
        hovertemplate=('<b>%{customdata[0]}</b> mean<br>BCPC1 %{x:.3g}<br>BCPC2 %{y:.3g}'
                       '<br>BCPC3 %{z:.3g}<extra></extra>')))
    anchors = result.anchors.loc[result.anchors['fit'].eq(fit_label)]
    fit_names = np.where(anchors['fold'] < 0, 'in-sample fit', 'fold ' + anchors['fold'].astype(str))
    traces.append(go.Scatter3d(
        x=anchors['BCPC1'], y=anchors['BCPC2'], z=anchors['BCPC3'], mode='markers',
        name='Spline anchors', legendgroup='anchors', showlegend=legend,
        marker=dict(size=6, symbol='diamond', color=INK),
        customdata=np.column_stack([anchors['anchor'].str.replace('_', ' '), fit_names]),
        hovertemplate=('<b>%{customdata[0]}</b><br>%{customdata[1]}<br>BCPC1 %{x:.3g}'
                       '<br>BCPC2 %{y:.3g}<br>BCPC3 %{z:.3g}<extra></extra>')))
    if fit_label == 'in_sample':
        line = result.curve.spline_curve
        traces.append(go.Scatter3d(
            x=line['BCPC1'], y=line['BCPC2'], z=line['BCPC3'], mode='lines',
            name='Arc-length spline (fitted in all BCPCs)', legendgroup='spline', showlegend=legend,
            line=dict(color=INK, width=4),
            customdata=line['bcpc_arc_length'].to_numpy(),
            hovertemplate='spline<br>arc length %{customdata:.3g}<extra></extra>'))
    return traces, len(shown), len(frame)


def plot_scores(results, max_rows=None, seed=42):
    """BCPC1-3 by labelled stakes level: one column per model, the in-sample fit with its
    spline on top and the held-out tasks below, each panel with its fits' spline anchors."""
    from plotly.subplots import make_subplots

    from .stakes_surface_pipeline import BASELINE, GRID, INK, INK_SECONDARY, SURFACE_COLOR

    names = list(results)
    fig = make_subplots(rows=len(FIT_CAPTIONS), cols=len(names),
                        specs=[[{'type': 'scene'}] * len(names)] * len(FIT_CAPTIONS),
                        subplot_titles=[f'{name}<br>{caption}' for caption in FIT_CAPTIONS.values()
                                        for name in names],
                        horizontal_spacing=0.01, vertical_spacing=0.05)
    counts = set()
    for row, fit_label in enumerate(FIT_CAPTIONS, start=1):
        for col, name in enumerate(names, start=1):
            traces, shown, total = _score_traces(results[name], fit_label, max_rows, seed,
                                                 legend=(row, col) == (1, 1))
            counts.add((shown, total))
            for trace in traces:
                fig.add_trace(trace, row=row, col=col)

    note = ''
    if any(shown < total for shown, total in counts):
        shown, total = min(counts)
        note = f' ({shown:,} of {total:,} rows shown per panel)'
    axis = dict(backgroundcolor=SURFACE_COLOR, gridcolor=GRID, zerolinecolor=BASELINE,
                color=INK_SECONDARY, showbackground=True)
    fig.update_scenes(aspectmode='data', camera=dict(eye=dict(x=1.7, y=1.7, z=1.0)),
                      **{f'{key}axis': dict(title=name, **axis) for key, name in zip('xyz', SCORE_AXES)})
    fig.update_annotations(font=dict(color=INK, size=13))
    fig.update_layout(
        template='plotly_white', height=1100, paper_bgcolor=SURFACE_COLOR,
        title=dict(text='BCPC1-3 by labelled stakes level' + note, font=dict(color=INK, size=16)),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=INK_SECONDARY),
        # A horizontal legend below the scenes never collides with them.
        legend=dict(itemsizing='constant', bgcolor=SURFACE_COLOR, orientation='h',
                    yanchor='top', y=0, x=0),
        margin=dict(l=16, r=16, t=80, b=16))
    return fig


def _style_2d(fig, title, height=460):
    from .stakes_surface_pipeline import BASELINE, GRID, INK, INK_SECONDARY, SURFACE_COLOR

    fig.update_layout(
        template='plotly_white', height=height, paper_bgcolor=SURFACE_COLOR,
        plot_bgcolor=SURFACE_COLOR, title=dict(text=title, font=dict(color=INK, size=16)),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=INK_SECONDARY),
        legend=dict(bgcolor=SURFACE_COLOR, orientation='h', yanchor='bottom', y=1.08, x=0),
        margin=dict(l=64, r=16, t=110, b=56))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=BASELINE, linecolor=BASELINE)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=BASELINE, linecolor=BASELINE)
    return fig


def plot_retention(results):
    """Retained held-out geometry against subspace size, one panel per model."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from .stakes_surface_pipeline import INK

    names = list(results)
    fig = make_subplots(rows=1, cols=len(names), shared_yaxes=True, subplot_titles=names,
                        horizontal_spacing=0.04)
    for col, name in enumerate(names, start=1):
        frame = results[name].retention
        summary = frame.groupby('dimensions')['bcpc'].agg(['mean', 'std'])
        k = summary.index.to_numpy()
        fig.add_trace(go.Scatter(
            x=k, y=summary['mean'], mode='lines+markers', name='Training BCPC1-k',
            legendgroup='bcpc', showlegend=col == 1, line=dict(color='#2a78d6', width=2),
            marker=dict(size=8, color='#2a78d6'),
            error_y=dict(type='data', array=summary['std'], color='#2a78d6', thickness=1),
            hovertemplate='%{x} dims: %{y:.3f}<extra>training BCPC</extra>'), row=1, col=col)
        dom = frame['difference_of_means'].dropna()
        fig.add_trace(go.Scatter(
            x=[1], y=[dom.mean()], mode='markers', name='Difference of means (1 dim)',
            legendgroup='dom', showlegend=col == 1,
            marker=dict(size=10, symbol='diamond', color=INK),
            error_y=dict(type='data', array=[dom.std()], color=INK, thickness=1),
            hovertemplate='1 dim: %{y:.3f}<extra>difference of means</extra>'), row=1, col=col)
        fig.update_xaxes(title_text='subspace dimensions', dtick=1, row=1, col=col)
    fig.update_yaxes(range=[0, 1.05], row=1)
    fig.update_yaxes(title_text='share of held-out geometry', row=1, col=1)
    return _style_2d(fig, 'Held-out stakes geometry retained '
                          '(mean ± sd over task folds; a random k-dim subspace keeps about k/5,000)')


def plot_refined(results):
    """Held-out refined stakes per task against its labelled level, one panel per model."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from .stakes_surface_pipeline import INK, stakes_colors

    names = list(results)
    colors = stakes_colors(LEVELS)
    fig = make_subplots(rows=1, cols=len(names), shared_yaxes=True, subplot_titles=names,
                        horizontal_spacing=0.04)
    for col, name in enumerate(names, start=1):
        tasks = task_refinement(results[name].refinement)
        for level in LEVELS:
            group = tasks.loc[tasks['stakes'].eq(level)]
            fig.add_trace(go.Box(
                x=[level] * len(group), y=group['refined'], name=level, showlegend=False,
                boxpoints='all', jitter=0.5, pointpos=0,
                marker=dict(size=5, color=colors[level], line=dict(color=INK, width=0.5)),
                line=dict(color=INK, width=1), fillcolor='rgba(0,0,0,0)',
                customdata=np.column_stack([group.index, group['residual']]),
                hovertemplate=('<b>%{customdata[0]}</b><br>refined %{y:.2f}'
                               '<br>residual %{customdata[1]:+.2f}<extra>' + level + '</extra>')),
                row=1, col=col)
        fig.update_xaxes(categoryorder='array', categoryarray=list(LEVELS),
                         tickangle=-45, row=1, col=col)
    fig.update_yaxes(title_text='held-out refined stakes', row=1, col=1)
    return _style_2d(fig, 'Held-out refined stakes per task by labelled level '
                          f'(1 = very_low centre, {N_CLASSES} = existential centre)', height=560)
