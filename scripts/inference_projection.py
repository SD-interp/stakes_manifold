"""Shared inference-only caching, frozen projection and CSV export."""
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import cache_activations
from .stakes_surface_bundle import load_surface_bundle

COORDINATE_COLUMNS = ['arc_length_parallel', 'arc_length_orthogonal']


def _check_bundle(config, metadata):
    for key in ('model_name', 'layer_component', 'position'):
        if metadata[key] != getattr(config, key):
            raise ValueError(f'Surface/config {key} mismatch.')


def project_caches(config, records, paths, *, row_fields, csv_columns, bundle=None):
    """Return ordered CSV rows and diagnostics, joining by template/cache row identity."""
    import torch
    arrays, metadata, coordinates = bundle or load_surface_bundle(config.surface_dir)
    _check_bundle(config, metadata)
    groups = cache_activations.template_groups(records)
    expected = {(tid, row): record for tid, group in groups.items() for row, record in enumerate(group)}
    projected = {}
    source_keys = set()
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
        scores = (X - arrays['pls_mean']) @ arrays['pls_rotations']
        mapped = coordinates.map_points(scores[:, :3], progress_seconds=None)
        outside = (scores[:, 2] < coordinates.heights[0]) | (scores[:, 2] > coordinates.heights[-1])
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
            )
    if projected.keys() != expected.keys():
        raise ValueError('Missing inference cache rows.')
    diagnostics = pd.DataFrame([projected[key] for key in expected])
    result = diagnostics[csv_columns].copy()
    if not np.isfinite(result[COORDINATE_COLUMNS].to_numpy()).all():
        raise RuntimeError('Nonfinite inference coordinates; CSV export aborted.')
    return result, diagnostics


def run(config, *, build_records, dataset_name, cache_namespace, csv_name,
        row_fields, csv_columns, force=False, model=None, tokenizer=None):
    """Generate/cache/project all prompts; return CSV path, rows and diagnostics.

    Activation caches are reusable; coordinates always use the current saved bundle.
    No fitting pipeline or fitting inventory is invoked. A caller that already holds
    a loaded model passes it (with its tokenizer) so several datasets share one load;
    the model is left loaded, and only caller-free memory is reclaimed here.
    """
    import torch
    bundle = load_surface_bundle(config.surface_dir)
    _check_bundle(config, bundle[1])
    directory = config.run_dir / 'inference' / dataset_name
    directory.mkdir(parents=True, exist_ok=True)
    records = build_records()
    prompt_path = directory / 'prompts.json'
    temporary = prompt_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(prompt_path)
    try:
        paths = cache_activations.run_inference(config, records, directory / 'activations',
                                               model=model, tokenizer=tokenizer,
                                               force=force, namespace=cache_namespace)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result, diagnostics = project_caches(config, records, paths, row_fields=row_fields,
                                         csv_columns=csv_columns, bundle=bundle)
    csv_path = directory / csv_name
    temporary = csv_path.with_suffix('.csv.tmp')
    result.to_csv(temporary, index=False, encoding='utf-8')
    temporary.replace(csv_path)
    print(f'Saved {len(result)} inference rows to {csv_path}')
    return csv_path, result, diagnostics
