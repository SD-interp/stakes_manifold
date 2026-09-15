"""Inventory activation caches and read their final-token rows in aligned batches."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def cache_signature(path):
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns)


class CacheInventory:
    """Metadata of every `.pt` cache under a directory, with checked activation reads.

    Each file must hold a single horizon type and share model, layer, position and
    width, with matching metadata, prompts and activation counts. Keys are paths
    relative to `key_root` plus the zero-based cache row. Source signatures are
    rechecked on every read, so a cache replaced mid-run cannot enter the fit.
    """

    def __init__(self, activations_dir, model_name, layer_component, position,
                 stakes_merges, batch_rows=1024, key_root=None):
        self.activations_dir = Path(activations_dir)
        self.key_root = Path(key_root) if key_root is not None else self.activations_dir
        self.model_name = model_name
        self.layer_component = layer_component
        self.position = position
        self.stakes_merges = dict(stakes_merges)
        self.batch_rows = batch_rows
        self.entries = {}
        self.feature_count = None
        for path in sorted(self.activations_dir.rglob('*.pt')):
            self._add(path)
        self.fit_files = [rel for rel, entry in self.entries.items()
                          if entry['horizon_type'] == 'horizon_free']
        self.stated_files = [rel for rel, entry in self.entries.items()
                             if entry['horizon_type'] == 'stated']
        if not self.fit_files:
            raise ValueError(f'No horizon-free caches found under {self.activations_dir}.')
        self.all_keys = pd.concat([entry['frame'][['source_file', 'source_row']]
                                   for entry in self.entries.values()], ignore_index=True)
        if self.all_keys.duplicated().any():
            raise ValueError('Duplicate source keys.')

    def _open(self, path):
        cache = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        if (cache['config']['model_name'] != self.model_name
                or cache['layer_component'] != self.layer_component):
            raise ValueError(f'Model/layer mismatch: {path}')
        if cache['positions'].count(self.position) != 1:
            raise ValueError(f'Token position missing or duplicated: {path}')
        tensor = cache['activations'][self.layer_component]
        if (tensor.ndim != 3 or not tensor.is_floating_point()
                or tensor.shape[1] != len(cache['positions'])
                or len(tensor) != len(cache['prompt_metadata'])
                or len(tensor) != len(cache['prompts']) or len(tensor) == 0):
            raise ValueError(f'Invalid cache shape or metadata counts: {path}')
        return cache, tensor, cache['positions'].index(self.position)

    def merged_stakes(self, metadata):
        return [self.stakes_merges.get(m['task_metadata']['stakes'], m['task_metadata']['stakes'])
                for m in metadata]

    def _add(self, path):
        signature = cache_signature(path)
        cache, tensor, _ = self._open(path)
        metadata = cache['prompt_metadata']
        free = [m['base_unit'] is None for m in metadata]
        if any(free) and not all(free):
            raise ValueError(f'Mixed stated and horizon-free rows: {path}')
        if self.feature_count is None:
            self.feature_count = tensor.shape[2]
        if tensor.shape[2] != self.feature_count:
            raise ValueError(f'Feature width mismatch: {path}')
        rel = path.relative_to(self.key_root).as_posix()
        frame = pd.DataFrame(metadata)
        frame['source_file'] = rel
        frame['source_row'] = np.arange(len(frame), dtype=np.int64)
        frame['prompt'] = cache['prompts']
        frame['horizon_type'] = 'horizon_free' if all(free) else 'stated'
        frame['stakes_original'] = [m['task_metadata']['stakes'] for m in metadata]
        frame['stakes'] = frame['stakes_original'].replace(self.stakes_merges)
        # This layout stores each source/template group as one .pt cache.
        frame['pls_weight_group'] = rel
        if frame[['prompt', 'stakes']].isna().any().any():
            raise ValueError(f'Missing prompt or stakes: {rel}')
        if cache_signature(path) != signature:
            raise RuntimeError(f'Cache changed during inventory: {rel}')
        self.entries[rel] = dict(path=path, signature=signature, frame=frame,
                                 horizon_type=frame['horizon_type'].iloc[0])

    def summary(self):
        return pd.DataFrame([
            dict(source_file=rel, horizon_type=entry['horizon_type'], rows=len(entry['frame']))
            for rel, entry in self.entries.items()
        ])

    def check_unchanged(self):
        for entry in self.entries.values():
            if cache_signature(entry['path']) != entry['signature']:
                raise RuntimeError('A source cache changed before export.')

    def batches(self, rel):
        """Yield (start, stop, float64 block) of final-token activations for one file."""
        entry = self.entries[rel]
        if cache_signature(entry['path']) != entry['signature']:
            raise RuntimeError(f'Cache changed since inventory: {rel}')
        cache, tensor, position_index = self._open(entry['path'])
        frame = entry['frame']
        if (tensor.shape[2] != self.feature_count or len(tensor) != len(frame)
                or list(cache['prompts']) != frame['prompt'].tolist()
                or not np.array_equal(frame['source_row'], np.arange(len(frame)))):
            raise ValueError(f'Cache row alignment changed: {rel}')
        if any((m['base_unit'] is None) != (entry['horizon_type'] == 'horizon_free')
               for m in cache['prompt_metadata']):
            raise ValueError(f'Cache horizon type changed: {rel}')
        if self.merged_stakes(cache['prompt_metadata']) != frame['stakes'].tolist():
            raise ValueError(f'Cache stakes alignment changed: {rel}')
        for start in range(0, len(tensor), self.batch_rows):
            stop = min(start + self.batch_rows, len(tensor))
            block = tensor[start:stop, position_index, :].to(torch.float64).numpy()
            if not np.isfinite(block).all():
                raise ValueError(f'Nonfinite activations: {rel}, rows {start}:{stop}')
            yield start, stop, block
        if cache_signature(entry['path']) != entry['signature']:
            raise RuntimeError(f'Cache changed during projection: {rel}')
