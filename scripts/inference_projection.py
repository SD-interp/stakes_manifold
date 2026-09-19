"""Shared inference-only caching, frozen projection and CSV export.

The pipeline runs in two halves. `InferenceDataset.cache` is the GPU half: it writes
the prompts and one activation cache per template, and needs no surface. `project` is
the CPU half: it reads those caches and maps them through the saved surface bundle.
`run` is both, for a host that has the model and an exported surface at the same time.
"""
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from . import cache_activations
from .geometric_surface import OrthogonalCoordinates
from .stakes_surface_bundle import load_surface_bundle

COORDINATE_COLUMNS = ['arc_length_parallel', 'arc_length_orthogonal']


def _check_bundle(config, metadata):
    for key in ('model_name', 'layer_component', 'position'):
        if metadata[key] != getattr(config, key):
            raise ValueError(f'Surface/config {key} mismatch.')


def _map_scores(coordinates, pls_scores):
    """Surface coordinates of PLS score rows, and whether each lies outside the fitted range.

    The geometric surface (`bcpc_surface`) maps every PLS component; the legacy slice surface
    maps PLS1-3 and is outside its range beyond its saved PLS3 heights.
    """
    if isinstance(coordinates, OrthogonalCoordinates):
        mapped = coordinates.map_points(pls_scores[:, :coordinates.shape.dimension])
        return mapped, mapped['coordinate_status'] != 'interior'
    mapped = coordinates.map_points(pls_scores[:, :3], progress_seconds=None)
    return mapped, ((pls_scores[:, 2] < coordinates.heights[0])
                    | (pls_scores[:, 2] > coordinates.heights[-1]))


def project_caches(config, records, paths, *, row_fields, csv_columns, bundle=None, scores=None):
    """Return ordered CSV rows and diagnostics, joining by template/cache row identity.

    `scores`, if given, maps a float64 activation batch to {column: per-row values},
    added beside the surface coordinates and checked finite with them.
    """
    import torch
    arrays, metadata, coordinates = bundle or load_surface_bundle(config.surface_dir)
    _check_bundle(config, metadata)
    groups = cache_activations.template_groups(records)
    expected = {(tid, row): record for tid, group in groups.items() for row, record in enumerate(group)}
    projected = {}
    source_keys = set()
    extra_columns = []
    for path in paths:
        path = Path(path).resolve()
        cached = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        settings = cached['config']
        for key in ('model_name', 'layer_component', 'position'):
            if settings.get(key) != metadata[key]:
                raise ValueError(f'Cache/surface {key} mismatch: {path}')
        if (cached['positions'] != [-1] or cached['layer_component'] != config.layer_component
                or settings.get('system_prompt') != '' or settings.get('enable_thinking') is not False
                or settings.get('add_generation_prompt') is not True):
            raise ValueError(f'Incompatible cache extraction settings: {path}')
        rows = cached['prompt_metadata']
        if (cached['input_sha256'] != cache_activations.fingerprint(rows, settings)
                or cached['prompts'] != [r['text'] for r in rows]):
            raise ValueError(f'Cache metadata mismatch: {path}')
        tensor = cached['activations'][config.layer_component]
        if tensor.shape != (len(rows), 1, metadata['feature_count']) or not tensor.is_floating_point():
            raise ValueError(f'Activation width/shape mismatch: {path}')
        X = tensor[:, 0, :].to(torch.float64).numpy()
        pls_scores = (X - arrays['pls_mean']) @ arrays['pls_rotations']
        mapped, outside = _map_scores(coordinates, pls_scores)
        extra = scores(X) if scores is not None else {}
        extra_columns = list(extra)
        template_ids = {r['template_id'] for r in rows}
        if len(template_ids) != 1:
            raise ValueError(f'Expected one template per cache: {path}')
        for source_row, record in enumerate(rows):
            source_key = (str(path), source_row)
            key = (record['template_id'], source_row)
            if source_key in source_keys or key in projected or expected.get(key) != record:
                raise ValueError(f'Duplicate or misaligned cache row: {source_key}')
            source_keys.add(source_key)
            projected[key] = dict(
                **row_fields(record),
                source_file=str(path), source_row=source_row,
                outside_saved_height_range=bool(outside[source_row]),
                **{name: values[source_row] for name, values in mapped.items()},
                **{name: values[source_row] for name, values in extra.items()},
            )
    if projected.keys() != expected.keys():
        raise ValueError('Missing inference cache rows.')
    diagnostics = pd.DataFrame([projected[key] for key in expected])
    result = diagnostics[csv_columns].copy()
    if not np.isfinite(diagnostics[[*COORDINATE_COLUMNS, *extra_columns]].to_numpy(dtype=float)).all():
        raise RuntimeError('Nonfinite inference coordinates; CSV export aborted.')
    return result, diagnostics


@dataclass(frozen=True)
class InferenceDataset:
    """One inference corpus and the two halves the pipeline runs it in.

    `name` is the directory under `artifacts/<model>/inference/`, `label` is the
    heading the notebooks print, and `cache_namespace` keeps the activations of one
    corpus from ever being mistaken for another's or for a fitting cache.
    """

    name: str
    label: str
    cache_namespace: str
    csv_name: str
    build_records: Callable[[], list]
    row_fields: Callable[[dict], dict]
    csv_columns: Sequence[str]

    def directory(self, config):
        return config.run_dir / 'inference' / self.name

    def cache(self, config, model, tokenizer, force=False):
        """GPU half: write `prompts.json` and one activation cache per template.

        No surface is loaded, so this runs before anything has been fitted. Existing
        caches are reused after a fingerprint check; a mismatch raises unless `force`.
        """
        import torch
        directory = self.directory(config)
        directory.mkdir(parents=True, exist_ok=True)
        records = self.build_records()
        _write_json(directory / 'prompts.json', records)
        try:
            return cache_activations.run_inference(
                config, records, directory / 'activations', model=model,
                tokenizer=tokenizer, force=force, namespace=self.cache_namespace)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def project(self, config, bundle=None):
        """CPU half: map the cached activations through the saved surface bundle.

        Returns the CSV path, its rows and the projection diagnostics. No model, no
        fitting pipeline and no fitting inventory is involved; coordinates always come
        from the bundle currently in `config.surface_dir`.
        """
        records = self.build_records()
        directory = self.directory(config)
        paths = cache_activations.cached_paths(
            config, records, directory / 'activations', self.cache_namespace)
        result, diagnostics = self.project_caches(config, records, paths, bundle)
        csv_path = directory / self.csv_name
        temporary = csv_path.with_suffix('.csv.tmp')
        result.to_csv(temporary, index=False, encoding='utf-8')
        temporary.replace(csv_path)
        print(f'Saved {len(result)} inference rows to {csv_path}')
        return csv_path, result, diagnostics

    def project_caches(self, config, records, paths, bundle=None):
        return project_caches(config, records, paths, row_fields=self.row_fields,
                              csv_columns=self.csv_columns, bundle=bundle)

    def run(self, config, model, tokenizer, force=False):
        """Both halves, for a host holding the model and an exported surface.

        The bundle is checked before any GPU work, so an absent or mismatched surface
        fails immediately instead of after the corpus has been cached.
        """
        bundle = load_surface_bundle(config.surface_dir)
        _check_bundle(config, bundle[1])
        self.cache(config, model, tokenizer, force=force)
        return self.project(config, bundle=bundle)


def _write_json(path, payload):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)
    return path
