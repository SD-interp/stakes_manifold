import copy
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from string import Formatter
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from scripts import (cache_activations, context_inference, severity_flipped_inference,
                     severity_inference,
                     severity_pairwise_inference, severity_wording_inference)
from scripts.corpora.inference import (context_prompts, severity_flipped_prompts,
                            severity_pairwise_prompts,
                            severity_wording_prompts)
from scripts.cache_inventory import CacheInventory
from scripts.corpora.inference.severity_prompts import TEMPLATES, build_prompt_records
from scripts.pipeline_config import RunConfig, NO_TIME_CORPORA
from scripts.stakes_height_slices import SliceConfig, SliceSurface
from scripts.stakes_surface_bundle import load_surface_bundle


class SeverityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = RunConfig(artifact_root=Path(self.temp.name), device='cpu', batch_size=2)
        self.records = build_prompt_records()
        self.directory = self.config.run_dir / 'inference' / 'severity'
        surface = SliceSurface([0, 0], [1, 1], np.zeros((4, 4)), [-2, 2], [-1, 1],
                               0, SliceConfig(n_slices=5))
        surface.build_cache(np.array([-1., 1.]))
        self.config.surface_dir.mkdir(parents=True)
        arrays = dict(pls_mean=np.zeros(3), pls_rotations=np.eye(3), **surface.arrays())
        archive = self.config.surface_dir / 'model.npz'
        np.savez_compressed(archive, **arrays)
        metadata = dict(schema_version=2, geometry_type='pls2_cubic_height_slices',
                        model_name=self.config.model_name, layer_component=self.config.layer_component,
                        position=-1, feature_count=3, slice_config=asdict(surface.config),
                        model_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        (self.config.surface_dir / 'model.json').write_text(json.dumps(metadata))

    @contextmanager
    def mock_model(self):
        active = []

        @contextmanager
        def hooks(modules, specs, early_exit, hookloc_resolver=None):
            active.append(specs['fwd'][0])
            try:
                yield
            finally:
                active.pop()

        class Model:
            device = torch.device('cpu')

            def named_modules(self):
                return []

            def __call__(self, input_ids, use_cache):
                active[0](None, None, input_ids.float().unsqueeze(1))

        def tokenize(texts):
            return {'input_ids': torch.tensor([[len(t) / 10, 0.5, (len(t) % 7) - 3] for t in texts])}

        hook_module = SimpleNamespace(HookSpecPost=lambda spec, fn: fn, temporary_hooks=hooks)
        with patch.dict('sys.modules', {'utils.mech_interp_toolkit.hook_utils': hook_module}), \
             patch.object(cache_activations, 'load_model', return_value=(Model(), tokenize)) as loader:
            yield loader, tokenize

    def test_dataset_and_notebook(self):
        self.assertEqual(len(TEMPLATES), 35)
        self.assertEqual(len(self.records), sum(len(t[3]) for t in TEMPLATES))
        self.assertEqual(len(self.records), 171)
        for _, _, template, values in TEMPLATES:
            fields = [field for _, field, _, _ in Formatter().parse(template)
                      if field is not None]
            self.assertEqual(len(fields), 1)
            self.assertEqual(len(values), len(set(values)))
        self.assertEqual(len({r['text'] for r in self.records}), len(self.records))
        for record in self.records:
            self.assertNotIn('stakes', record['task_metadata'])
            for field in ('base_value', 'base_unit', 'unit_variant', 'number_format', 'value', 'value_text', 'unit'):
                self.assertIsNone(record[field])
        nb = json.loads(Path('notebooks/arc_length_pipeline.ipynb').read_text(encoding='utf-8'))
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), f'cell-{i}', 'exec')
        code = '\n'.join(''.join(c['source']) for c in nb['cells'])
        self.assertLess(code.index('pipeline.export('), code.index('severity_inference.run('))
        self.assertLess(code.index('severity_inference.run('), code.index('severity_flipped_inference.run('))
        self.assertLess(code.index('severity_flipped_inference.run('), code.index('severity_pairwise_inference.run('))
        self.assertLess(code.index('severity_pairwise_inference.run('), code.index('severity_wording_inference.run('))
        self.assertLess(code.index('severity_wording_inference.run('), code.index('context_inference.run('))

    def test_context_dataset_has_matched_editable_contexts(self):
        groups = context_prompts.TASK_GROUPS
        records = context_prompts.build_prompt_records()
        self.assertEqual(len(groups), 5)
        self.assertEqual(len(records), 40)
        self.assertEqual(len({record['text'] for record in records}), 40)
        for group in groups:
            rows = [record for record in records if record['task'] == group['id']]
            self.assertEqual([row['task_metadata']['context'] for row in rows],
                             list(context_prompts.CONTEXT_ORDER) * 2)
            self.assertEqual([row['text'] for row in rows], list(group['prompts'].values()) + list(group['after_prompts'].values()))
            self.assertEqual(len({row['template_metadata']['core_request'] for row in rows}), 1)
            self.assertEqual([row['task_metadata']['context_position'] for row in rows],
                             ['before'] * 4 + ['after'] * 4)
            for before, after in zip(rows[:4], rows[4:]):
                framing, _, scenario = before['text'].partition('. ')
                self.assertEqual(after['text'], scenario + ' ' + framing + '.')
            for row in rows:
                self.assertEqual(row['task_metadata']['usage'], 'inference_only')
                self.assertNotIn('stakes', row['task_metadata'])
                self.assertNotIn('expected_direction', row['task_metadata'])
                for field in ('base_value', 'base_unit', 'unit_variant', 'number_format',
                              'value', 'value_text', 'unit'):
                    self.assertIsNone(row[field])

        changed = copy.deepcopy(groups)
        changed[0]['prompts']['video_game'] += ' Nobody can be harmed.'
        with patch.object(context_prompts, 'TASK_GROUPS', changed):
            with self.assertRaisesRegex(ValueError, 'scenario differs'):
                context_prompts.build_prompt_records()

        changed = copy.deepcopy(groups)
        changed[0]['after_prompts']['video_game'] += ' Nobody can be harmed.'
        with patch.object(context_prompts, 'TASK_GROUPS', changed):
            with self.assertRaisesRegex(ValueError, 'only move the context sentence'):
                context_prompts.build_prompt_records()

    def test_context_inference_is_isolated_reusable_and_aligned(self):
        before = {path.name: path.read_bytes() for path in self.config.surface_dir.iterdir()}
        records = context_prompts.build_prompt_records()
        directory = self.config.run_dir / 'inference' / 'context'
        with self.mock_model() as (loader, tokenize):
            csv, result, diagnostics = context_inference.run(self.config)
            self.assertEqual(loader.call_count, 1)
            cache_times = {path: path.stat().st_mtime_ns
                           for path in (directory / 'activations').glob('*.pt')}
            self.assertEqual(len(cache_times), 5)
            self.assertTrue(all(path.name.startswith('context_inference--')
                                for path in cache_times))
            loader.reset_mock()
            context_inference.run(self.config)
            loader.assert_not_called()
            self.assertEqual(cache_times, {path: path.stat().st_mtime_ns
                                           for path in cache_times})

        self.assertEqual(before, {path.name: path.read_bytes()
                                  for path in self.config.surface_dir.iterdir()})
        self.assertEqual(csv, directory / 'context_arc_lengths.csv')
        self.assertEqual(list(pd.read_csv(csv).columns), context_inference.CSV_COLUMNS)
        self.assertEqual(result.task.tolist(), [record['task'] for record in records])
        self.assertEqual(result.context.tolist(),
                         [record['task_metadata']['context'] for record in records])
        self.assertEqual(result.context_position.tolist(),
                         [record['task_metadata']['context_position'] for record in records])
        self.assertEqual(result.prompt.tolist(), [record['text'] for record in records])
        arrays, _, coordinates = load_surface_bundle(self.config.surface_dir)
        X = tokenize([record['text'] for record in records])['input_ids'].float().double().numpy()
        expected = coordinates.map_points(
            ((X - arrays['pls_mean']) @ arrays['pls_rotations'])[:, :3],
            progress_seconds=None,
        )
        for column in context_inference.CSV_COLUMNS[-2:]:
            np.testing.assert_allclose(result[column], expected[column])
        paths = sorted(cache_times, reverse=True)
        reordered, _ = context_inference.project_caches(self.config, records, paths)
        pd.testing.assert_frame_equal(result, reordered)
        with self.assertRaisesRegex(ValueError, 'Missing'):
            context_inference.project_caches(self.config, records, paths[:-1])
        fitting_cache_paths = (list(self.config.activations_dir.rglob('*.pt'))
                               if self.config.activations_dir.exists() else [])
        self.assertTrue(all('context' not in path.name for path in fitting_cache_paths))

    def test_flipped_dataset_preserves_reference_contract(self):
        reference = {template[0]: template for template in TEMPLATES}
        flipped = {template[0]: template for template in severity_flipped_prompts.TEMPLATES}
        self.assertEqual(len(flipped), 20)
        self.assertLess(set(flipped), set(reference))
        self.assertEqual([template[0] for template in severity_flipped_prompts.TEMPLATES],
                         [template[0] for template in TEMPLATES[:20]])

        records = severity_flipped_prompts.build_prompt_records()
        self.assertEqual(len(records),
                         sum(len(t[3]) for t in severity_flipped_prompts.TEMPLATES))
        self.assertEqual(len(records), 111)
        self.assertEqual(len({record['text'] for record in records}), len(records))
        reference_texts = {record['text'] for record in self.records}
        for template_id, (_, domain, template, values) in flipped.items():
            _, reference_domain, reference_template, reference_values = reference[template_id]
            self.assertEqual(domain, reference_domain)
            self.assertEqual(values, reference_values)
            self.assertNotEqual(template, reference_template)
            reference_fields = [field for _, field, _, _ in Formatter().parse(reference_template)
                                if field is not None]
            flipped_fields = [field for _, field, _, _ in Formatter().parse(template)
                              if field is not None]
            self.assertEqual(flipped_fields, reference_fields)
            rows = [record for record in records if record['template_id'] == template_id]
            self.assertEqual([row['task_metadata']['severity_word'] for row in rows], values)
            for row in rows:
                self.assertNotIn('{', row['text'])
                self.assertNotIn('}', row['text'])
                self.assertNotIn(row['text'], reference_texts)
                self.assertEqual(row['task'], template_id)
                self.assertNotIn('stakes', row['task_metadata'])
                self.assertEqual(row['task_metadata']['usage'], 'inference_only')
                self.assertEqual(row['template_metadata'],
                                 {'template': template, 'domain': domain})
                for field in ('base_value', 'base_unit', 'unit_variant', 'number_format',
                              'value', 'value_text', 'unit'):
                    self.assertIsNone(row[field])

        changed = [list(template) for template in severity_flipped_prompts.TEMPLATES]
        changed[0][3] = list(changed[0][3]) + [changed[0][3][0]]
        with patch.object(severity_flipped_prompts, 'TEMPLATES',
                          [tuple(template) for template in changed]):
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                severity_flipped_prompts.build_prompt_records()

    def test_pairwise_dataset_contrasts_low_and_high_severity_fillers(self):
        reference = {template[0]: template for template in TEMPLATES}
        pairwise = {template[0]: template for template in severity_pairwise_prompts.TEMPLATES}
        self.assertEqual(len(pairwise), 10)
        # Each pair names the reference family it contrasts, and introduces its own ID.
        self.assertTrue({template[2] for template in severity_pairwise_prompts.TEMPLATES}
                        <= set(reference))
        self.assertFalse(set(pairwise) & set(reference))

        records = severity_pairwise_prompts.build_prompt_records()
        self.assertEqual(len(records), 20)
        self.assertEqual(len({record['text'] for record in records}), len(records))
        reference_texts = {record['text'] for record in self.records}
        for template_id, (_, domain, family, template, low, high) in pairwise.items():
            # Pairs carry their own domain label, independent of the family's.
            self.assertTrue(domain)
            fields = [field for _, field, _, _ in Formatter().parse(template)
                      if field is not None]
            self.assertEqual(len(fields), 1)
            rows = [record for record in records if record['template_id'] == template_id]
            self.assertEqual([row['task_metadata']['severity_word'] for row in rows],
                             [low, high])
            self.assertEqual([row['task_metadata']['severity_pole'] for row in rows],
                             ['low', 'high'])
            for row in rows:
                self.assertNotIn('{', row['text'])
                self.assertNotIn('}', row['text'])
                self.assertEqual(row['task'], template_id)
                self.assertEqual(row['task_metadata']['family'], family)
                self.assertNotIn('stakes', row['task_metadata'])
                self.assertEqual(row['task_metadata']['usage'], 'inference_only')
                self.assertEqual(row['template_metadata'],
                                 {'template': template, 'domain': domain})
                for field in ('base_value', 'base_unit', 'unit_variant', 'number_format',
                              'value', 'value_text', 'unit'):
                    self.assertIsNone(row[field])

        # The pairs are restated in declarative form; they must not collide with the
        # reference corpus.
        self.assertFalse({record['text'] for record in records} & reference_texts)

        changed = [tuple(template) for template in severity_pairwise_prompts.TEMPLATES][:-1]
        with patch.object(severity_pairwise_prompts, 'TEMPLATES', changed):
            with self.assertRaisesRegex(ValueError, '10 uniquely identified'):
                severity_pairwise_prompts.build_prompt_records()

    def test_wording_dataset_and_independent_variant_lists(self):
        groups = severity_wording_prompts.TASK_GROUPS
        records = severity_wording_prompts.build_prompt_records()
        self.assertEqual(len(groups), 20)
        self.assertEqual(len(records), 160)
        self.assertEqual(len({r['text'] for r in records}), 160)
        self.assertEqual(len({id(g['variants']) for g in groups}), 20)
        for group in groups:
            rows = [r for r in records if r['template_id'] == group['id']]
            self.assertEqual([r['text'] for r in rows],
                             [v.format(task=group['task']) for v in group['variants']])
            for row in rows:
                self.assertEqual(row['task'], group['task'])
                self.assertEqual(row['text'].count(group['task']), 1)
                self.assertEqual(row['task_metadata'], {'usage': 'inference_only'})
                for field in ('base_value', 'base_unit', 'unit_variant', 'number_format', 'value', 'value_text', 'unit'):
                    self.assertIsNone(row[field])
        changed = copy.deepcopy(groups)
        changed[0]['variants'].append('Can you help me {task}, please?')
        with patch.object(severity_wording_prompts, 'TASK_GROUPS', changed):
            self.assertEqual(len(severity_wording_prompts.build_prompt_records()), 161)
        changed[0]['variants'].append('Help me {different_task}.')
        with patch.object(severity_wording_prompts, 'TASK_GROUPS', changed), self.assertRaises(ValueError):
            severity_wording_prompts.build_prompt_records()

    def test_both_datasets_stay_separate_and_wording_projects_correctly(self):
        before = {p.name: p.read_bytes() for p in self.config.surface_dir.iterdir()}
        records = severity_wording_prompts.build_prompt_records()
        directory = self.config.run_dir / 'inference' / 'severity_wording'
        training = copy.deepcopy(self.records[:2])
        for record in training:
            record['task_metadata'] = {'stakes': 'very_low'}
        with self.mock_model() as (loader, tokenize):
            cache_activations.run(self.config, {NO_TIME_CORPORA[0]: training})
            original_csv, original, _ = severity_inference.run(self.config)
            original_files = {p: p.read_bytes() for p in self.directory.rglob('*') if p.is_file()}
            csv, result, diagnostics = severity_wording_inference.run(self.config)
            self.assertNotEqual(csv, original_csv)
            self.assertEqual(csv.name, 'severity_wording_arc_lengths.csv')
            self.assertEqual(original_files, {p: p.read_bytes() for p in original_files})
            cache_times = {p: p.stat().st_mtime_ns for p in (directory / 'activations').glob('*.pt')}
            self.assertEqual(len(cache_times), 20)
            self.assertTrue(all(p.name.startswith('severity_wording_inference--') for p in cache_times))
            loader.reset_mock()
            severity_inference.run(self.config)
            severity_wording_inference.run(self.config)
            loader.assert_not_called()
            self.assertEqual(cache_times, {p: p.stat().st_mtime_ns for p in cache_times})
            changed = copy.deepcopy(records)
            changed[0]['text'] += ' Please.'
            with self.assertRaisesRegex(RuntimeError, 'Stale'):
                cache_activations.run_inference(self.config, changed, directory / 'activations',
                                                namespace='severity_wording_inference')
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.config.surface_dir.iterdir()})
        self.assertEqual(list(pd.read_csv(csv).columns), severity_wording_inference.CSV_COLUMNS)
        self.assertEqual(list(pd.read_csv(original_csv).columns), severity_inference.CSV_COLUMNS)
        self.assertEqual(len(original), len(self.records))
        self.assertEqual(len(result), 160)
        self.assertEqual(result.task.tolist(), [r['task'] for r in records])
        self.assertEqual(result.prompt.tolist(), [r['text'] for r in records])
        self.assertTrue(diagnostics.outside_saved_height_range.any())
        arrays, _, coordinates = load_surface_bundle(self.config.surface_dir)
        X = tokenize([r['text'] for r in records])['input_ids'].float().double().numpy()
        expected = coordinates.map_points(((X - arrays['pls_mean']) @ arrays['pls_rotations'])[:, :3], progress_seconds=None)
        for column in severity_wording_inference.CSV_COLUMNS[2:]:
            np.testing.assert_allclose(result[column], expected[column])
        reversed_paths = sorted(cache_times, reverse=True)
        reordered, _ = severity_wording_inference.project_caches(self.config, records, reversed_paths)
        pd.testing.assert_frame_equal(result, reordered)
        with self.assertRaisesRegex(ValueError, 'Missing'):
            severity_wording_inference.project_caches(self.config, records, reversed_paths[:-1])
        for _ in range(2):
            inventory = CacheInventory(self.config.activations_dir, self.config.model_name,
                                       self.config.layer_component, -1, self.config.stakes_merges)
            self.assertEqual(len(inventory.fit_files), 1)
            self.assertEqual(len(inventory.all_keys), 2)
            self.assertTrue(all('severity' not in key for key in inventory.entries))

    def test_flipped_inference_is_isolated_aligned_and_uses_frozen_bundle(self):
        before = {path.name: path.read_bytes() for path in self.config.surface_dir.iterdir()}
        records = severity_flipped_prompts.build_prompt_records()
        directory = self.config.run_dir / 'inference' / 'severity_flipped'
        with self.mock_model() as (loader, tokenize):
            reference_csv, _, _ = severity_inference.run(self.config)
            loader.reset_mock()
            csv, result, diagnostics = severity_flipped_inference.run(self.config)
            self.assertEqual(loader.call_count, 1)
            loader.reset_mock()
            severity_flipped_inference.run(self.config)
            loader.assert_not_called()
        self.assertNotEqual(csv, reference_csv)
        self.assertEqual(csv.name, 'severity_flipped_arc_lengths.csv')
        self.assertEqual(list(pd.read_csv(csv).columns), severity_flipped_inference.CSV_COLUMNS)
        self.assertEqual(len(result), len(records))
        self.assertEqual(result.severity_word.tolist(),
                         [record['task_metadata']['severity_word'] for record in records])
        self.assertTrue(diagnostics.outside_saved_height_range.any())
        self.assertEqual(before, {path.name: path.read_bytes()
                                  for path in self.config.surface_dir.iterdir()})

        paths = sorted((directory / 'activations').glob('*.pt'), reverse=True)
        self.assertEqual(len(paths), len(severity_flipped_prompts.TEMPLATES))
        self.assertTrue(all(path.name.startswith('severity_flipped_inference--')
                            for path in paths))
        reordered, _ = severity_flipped_inference.project_caches(
            self.config, records, paths)
        pd.testing.assert_frame_equal(result, reordered)
        with self.assertRaisesRegex(ValueError, 'Missing'):
            severity_flipped_inference.project_caches(self.config, records, paths[:-1])

        arrays, _, coordinates = load_surface_bundle(self.config.surface_dir)
        X = tokenize([record['text'] for record in records])['input_ids'].float().double().numpy()
        expected = coordinates.map_points(
            ((X - arrays['pls_mean']) @ arrays['pls_rotations'])[:, :3],
            progress_seconds=None)
        for column in severity_flipped_inference.CSV_COLUMNS[2:]:
            np.testing.assert_allclose(result[column], expected[column])

        changed = copy.deepcopy(records)
        changed[0]['text'] += ' Please.'
        with self.mock_model(), self.assertRaisesRegex(RuntimeError, 'Stale'):
            cache_activations.run_inference(
                self.config, changed, directory / 'activations',
                namespace='severity_flipped_inference')
        training = copy.deepcopy(self.records[:2])
        for record in training:
            record['task_metadata'] = {'stakes': 'very_low'}
        with self.mock_model():
            cache_activations.run(self.config, {NO_TIME_CORPORA[0]: training})
        inventory = CacheInventory(self.config.activations_dir, self.config.model_name,
                                   self.config.layer_component, -1,
                                   self.config.stakes_merges)
        self.assertTrue(all('severity_flipped' not in key for key in inventory.entries))

    def test_streamlit_registry_includes_flipped_dataset(self):
        app = Path('arc_length_app.py').read_text(encoding='utf-8')
        self.assertIn('"severity_flipped": ("template", "severity_word")', app)
        self.assertIn('"severity_pairwise": ("template", "severity_word")', app)
        self.assertNotIn('use_container_width', app)

    def test_pairwise_inference_is_isolated_reusable_and_aligned(self):
        before = {path.name: path.read_bytes() for path in self.config.surface_dir.iterdir()}
        records = severity_pairwise_prompts.build_prompt_records()
        directory = self.config.run_dir / 'inference' / 'severity_pairwise'
        with self.mock_model() as (loader, tokenize):
            csv, result, diagnostics = severity_pairwise_inference.run(self.config)
            self.assertEqual(loader.call_count, 1)
            loader.reset_mock()
            severity_pairwise_inference.run(self.config)
            loader.assert_not_called()
        self.assertEqual(csv.name, 'severity_pairwise_arc_lengths.csv')
        self.assertEqual(list(pd.read_csv(csv).columns), severity_pairwise_inference.CSV_COLUMNS)
        self.assertEqual(len(result), len(records))
        self.assertEqual(len(result), 20)
        self.assertEqual(result.severity_word.tolist(),
                         [record['task_metadata']['severity_word'] for record in records])
        self.assertTrue(diagnostics.outside_saved_height_range.any())
        self.assertEqual(before, {path.name: path.read_bytes()
                                  for path in self.config.surface_dir.iterdir()})

        paths = sorted((directory / 'activations').glob('*.pt'), reverse=True)
        self.assertEqual(len(paths), len(severity_pairwise_prompts.TEMPLATES))
        self.assertTrue(all(path.name.startswith('severity_pairwise_inference--')
                            for path in paths))
        reordered, _ = severity_pairwise_inference.project_caches(self.config, records, paths)
        pd.testing.assert_frame_equal(result, reordered)
        with self.assertRaisesRegex(ValueError, 'Missing'):
            severity_pairwise_inference.project_caches(self.config, records, paths[:-1])

        arrays, _, coordinates = load_surface_bundle(self.config.surface_dir)
        X = tokenize([record['text'] for record in records])['input_ids'].float().double().numpy()
        expected = coordinates.map_points(
            ((X - arrays['pls_mean']) @ arrays['pls_rotations'])[:, :3],
            progress_seconds=None)
        for column in severity_pairwise_inference.CSV_COLUMNS[2:]:
            np.testing.assert_allclose(result[column], expected[column])

        changed = copy.deepcopy(records)
        changed[0]['text'] += ' Please.'
        with self.mock_model(), self.assertRaisesRegex(RuntimeError, 'Stale'):
            cache_activations.run_inference(
                self.config, changed, directory / 'activations',
                namespace='severity_pairwise_inference')

    def test_end_to_end_reuse_alignment_and_frozen_bundle(self):
        before = {p.name: p.read_bytes() for p in self.config.surface_dir.iterdir()}
        with self.mock_model() as (loader, tokenize):
            csv, result, diagnostics = severity_inference.run(self.config)
            self.assertEqual(loader.call_count, 1)
            loader.reset_mock()
            severity_inference.run(self.config)
            loader.assert_not_called()
        self.assertEqual(list(pd.read_csv(csv).columns), severity_inference.CSV_COLUMNS)
        self.assertEqual(len(result), len(self.records))
        self.assertTrue(diagnostics.outside_saved_height_range.any())
        self.assertFalse(list(self.config.activations_dir.rglob('*.pt')))
        self.assertNotIn('severity_inference', NO_TIME_CORPORA)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.config.surface_dir.iterdir()})
        arrays, metadata, coordinates = load_surface_bundle(self.config.surface_dir)
        X = tokenize([r['text'] for r in self.records])['input_ids'].float().double().numpy()
        expected = coordinates.map_points(((X - arrays['pls_mean']) @ arrays['pls_rotations'])[:, :3], progress_seconds=None)
        for column in severity_inference.CSV_COLUMNS[2:]:
            np.testing.assert_allclose(result[column], expected[column])
        paths = sorted((self.directory / 'activations').glob('*.pt'), reverse=True)
        reordered, _ = severity_inference.project_caches(self.config, self.records, paths)
        pd.testing.assert_frame_equal(result, reordered)
        self.assertEqual(result.severity_word.tolist(), [r['task_metadata']['severity_word'] for r in self.records])
        with self.assertRaisesRegex(ValueError, 'Missing'):
            severity_inference.project_caches(self.config, self.records, paths[:-1])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            severity_inference.project_caches(self.config, self.records, paths + paths[:1])
        changed = copy.deepcopy(self.records)
        changed[0]['text'] += ' Please.'
        with self.mock_model(), self.assertRaisesRegex(RuntimeError, 'Stale'):
            cache_activations.run_inference(self.config, changed, self.directory / 'activations')

    def test_isolation_and_mismatched_model(self):
        with self.assertRaisesRegex(ValueError, 'outside'):
            cache_activations.run_inference(self.config, self.records, self.config.activations_dir / 'inference')
        with self.assertRaisesRegex(ValueError, 'horizon-free'):
            cache_activations.run(self.config, {'severity_inference': self.records})
        arrays, metadata, coordinates = load_surface_bundle(self.config.surface_dir)
        metadata['model_name'] = 'different-model'
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            severity_inference.project_caches(self.config, self.records, [], (arrays, metadata, coordinates))

    def test_training_inventory_excludes_inference_and_width_is_checked(self):
        training = copy.deepcopy(self.records[:2])
        for record in training:
            record['task_metadata'] = {'stakes': 'very_low'}
        with self.mock_model():
            cache_activations.run(self.config, {NO_TIME_CORPORA[0]: training})
            paths = cache_activations.run_inference(self.config, self.records[:2], self.directory / 'activations')
        for _ in range(2):
            inventory = CacheInventory(self.config.activations_dir, self.config.model_name,
                                       self.config.layer_component, -1, self.config.stakes_merges)
            self.assertEqual(len(inventory.fit_files), 1)
            self.assertEqual(len(inventory.all_keys), 2)
            self.assertTrue(all('severity_inference' not in key for key in inventory.entries))
        payload = torch.load(paths[0], weights_only=False)
        payload['activations'][self.config.layer_component] = torch.zeros(2, 1, 4)
        torch.save(payload, paths[0])
        with self.assertRaisesRegex(ValueError, 'width/shape'):
            severity_inference.project_caches(self.config, self.records[:2], paths)


if __name__ == '__main__':
    unittest.main()
