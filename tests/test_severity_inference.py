import copy
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from string import Formatter
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from scripts import (cache_activations, context_inference, severity_composition_inference,
                     severity_flipped_inference,
                     severity_inference, severity_magnitude_inference, severity_null_inference,
                     severity_pairwise_inference, severity_wording_inference, stated_stakes)
from scripts.corpora.inference import (context_prompts, rating_phrasings,
                            severity_composition_prompts, severity_flipped_prompts,
                            severity_magnitude_prompts,
                            severity_null_prompts, severity_pairwise_prompts,
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

    @staticmethod
    def as_training_records(records):
        """Turn inference records into the fitting shape the inventory expects."""
        training = copy.deepcopy(records)
        for record in training:
            record['task_metadata'] = {'stakes': 'very_low'}
            # Only fitting records carry a horizon; None is the horizon-free marker.
            record['base_unit'] = None
        return training

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
                self.assertNotIn(field, record)
        nb = json.loads(Path('notebooks/arc_length_pipeline.ipynb').read_text(encoding='utf-8'))
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), f'cell-{i}', 'exec')
        code = '\n'.join(''.join(c['source']) for c in nb['cells'])
        self.assertLess(code.index('pipeline.export('), code.index('severity_inference.run('))
        self.assertLess(code.index('severity_inference.run('), code.index('severity_null_inference.run('))
        self.assertLess(code.index('severity_null_inference.run('), code.index('severity_magnitude_inference.run('))
        self.assertLess(code.index('severity_magnitude_inference.run('), code.index('severity_composition_inference.run('))
        self.assertLess(code.index('severity_composition_inference.run('), code.index('severity_flipped_inference.run('))
        self.assertLess(code.index('severity_flipped_inference.run('), code.index('severity_pairwise_inference.run('))
        self.assertLess(code.index('severity_pairwise_inference.run('), code.index('severity_wording_inference.run('))
        self.assertLess(code.index('severity_wording_inference.run('), code.index('context_inference.run('))
        # Generation runs last: it reloads the model, and by then every CSV is exported.
        self.assertLess(code.index('context_inference.run('), code.index('stated_stakes.run_all('))

    def test_context_dataset_is_a_constructed_ladder_with_controls(self):
        contexts, groups = context_prompts.CONTEXTS, context_prompts.TASK_GROUPS
        frames, controls = context_prompts.FRAME_ORDER, context_prompts.CONTROL_ORDER
        phrasings = context_prompts.PHRASINGS_PER_CONTEXT
        low, high = context_prompts.FRAMING_WORD_RANGE
        records = context_prompts.build_prompt_records()
        placed = len(context_prompts.CONTEXT_POSITIONS) * phrasings
        per_task = (len(frames) + 1) * placed + 1
        self.assertEqual((len(groups), len(frames), controls), (5, 8, ('none', 'filler')))
        self.assertEqual((phrasings, per_task), (3, 55))
        self.assertEqual(len(records), len(groups) * per_task)
        self.assertEqual(len({record['text'] for record in records}), len(records))

        # The ladder is carried by two decomposed properties, not one opaque label.
        self.assertEqual([contexts[name]['ladder_rank'] for name in frames],
                         list(range(1, len(frames) + 1)))
        self.assertTrue(all(contexts[name]['ladder_rank'] is None for name in controls))
        self.assertEqual([contexts[name]['reality_status'] for name in frames],
                         ['real'] * 3 + ['simulated'] * 2 + ['fictional'] * 3)
        self.assertEqual([contexts[name]['answer_use'] for name in frames],
                         ['acted_now', 'acted_later', 'none', 'rehearsal'] + ['none'] * 4)
        # The filler control matches the frames in length while framing nothing.
        self.assertEqual(len(contexts['filler']['phrasings']), phrasings)
        self.assertEqual(contexts['none']['phrasings'], ())
        for name in context_prompts.CONTEXT_ORDER:
            for text in contexts[name]['phrasings']:
                self.assertTrue(low <= len(text.split()) <= high, text)

        for group in groups:
            scenario = group['scenario']
            rows = [record for record in records if record['task'] == group['id']]
            self.assertEqual(len(rows), per_task)
            self.assertEqual([row['task_metadata']['context'] for row in rows],
                             [name for name in context_prompts.CONTEXT_ORDER
                              for _ in range(1 if name == 'none' else placed)])
            for row in rows:
                metadata, framing = row['task_metadata'], row['task_metadata']['framing']
                self.assertEqual(row['template_metadata']['core_request'], scenario)
                self.assertEqual(metadata['is_control'], metadata['context'] in controls)
                self.assertEqual(metadata['usage'], 'inference_only')
                self.assertNotIn('stakes', metadata)
                self.assertNotIn('expected_direction', metadata)
                for field in ('base_value', 'base_unit', 'unit_variant', 'number_format',
                              'value', 'value_text', 'unit'):
                    self.assertNotIn(field, row)
                if framing is None:
                    self.assertEqual(metadata['context'], 'none')
                    self.assertEqual(metadata['context_position'], context_prompts.BARE_POSITION)
                    self.assertEqual(row['text'], scenario)
                    continue
                # Every other prompt is the scenario verbatim plus one framing sentence.
                options = contexts[metadata['context']]['phrasings']
                self.assertEqual(options[metadata['phrasing_index']], framing)
                self.assertEqual(row['text'],
                                 {'before': framing + ' ' + scenario,
                                  'after': scenario + ' ' + framing}[metadata['context_position']])

        changed = copy.deepcopy(contexts)
        changed['game']['phrasings'] = changed['game']['phrasings'][:2]
        with patch.object(context_prompts, 'CONTEXTS', changed):
            with self.assertRaisesRegex(ValueError, 'distinct phrasings'):
                context_prompts.build_prompt_records()

        changed = copy.deepcopy(contexts)
        changed['game']['phrasings'] = ('This is a game.',) + changed['game']['phrasings'][1:]
        with patch.object(context_prompts, 'CONTEXTS', changed):
            with self.assertRaisesRegex(ValueError, f'{low}-{high} words'):
                context_prompts.build_prompt_records()

        changed = copy.deepcopy(contexts)
        changed['filler']['ladder_rank'] = len(frames) + 1
        with patch.object(context_prompts, 'CONTEXTS', changed):
            with self.assertRaisesRegex(ValueError, 'outside the ladder'):
                context_prompts.build_prompt_records()

        changed = copy.deepcopy(groups)
        changed[0]['scenario'] += ' Nobody can be harmed in {setting}.'
        with patch.object(context_prompts, 'TASK_GROUPS', changed):
            with self.assertRaisesRegex(ValueError, 'unparameterised scenario'):
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

    def test_null_dataset_reuses_reference_vocabulary_without_stakes(self):
        families = severity_null_prompts.FAMILIES
        frames = severity_null_prompts.FRAMES
        records = severity_null_prompts.build_prompt_records()
        reference = {template[0]: template[3] for template in TEMPLATES}
        self.assertEqual((len(families), len(frames)), (12, 4))
        self.assertEqual(len(records), len(frames) * sum(len(f[2]) for f in families))
        self.assertEqual(len({record['text'] for record in records}), len(records))
        # Every value is copied from the reference family it names, so the two
        # corpora differ only in the frame the phrase sits in.
        self.assertLess({family[0] for family in families}, set(reference))
        for family_id, _, values in families:
            self.assertEqual(values, reference[family_id])
        self.assertEqual([frame[1] for frame in frames], [False, False, False, True])
        for record in records:
            metadata = record['task_metadata']
            self.assertIn(metadata['severity_word'], record['text'])
            self.assertEqual(metadata['usage'], 'inference_only')
            self.assertNotIn('stakes', metadata)
            self.assertNotIn('expected_direction', metadata)
            for field in ('base_value', 'base_unit', 'unit_variant', 'number_format',
                          'value', 'value_text', 'unit'):
                self.assertNotIn(field, record)
        rows = [r for r in records if r['template_id'] == 'disease']
        self.assertEqual([r['task_metadata']['frame'] for r in rows],
                         [frame[0] for frame in frames for _ in reference['disease']])

        changed = [(frames[0][0], frames[0][1], 'Count the letters.')] + list(frames[1:])
        with patch.object(severity_null_prompts, 'FRAMES', changed):
            with self.assertRaisesRegex(ValueError, 'one {value} slot'):
                severity_null_prompts.build_prompt_records()

    def test_rating_phrasings_enforce_a_parseable_shape(self):
        rating_phrasings.check_phrasings()
        low, high = rating_phrasings.RATING_SCALE
        self.assertEqual((low, high), (0, 10))
        self.assertEqual(rating_phrasings.GENERATED_TOKENS, 10)
        formats = {entry[1] for entry in rating_phrasings.PHRASINGS}
        positions = {entry[2] for entry in rating_phrasings.PHRASINGS}
        self.assertEqual(len(formats), 5)
        self.assertEqual(positions, {'before', 'after'})
        replies = {'bare_number': '7', 'labelled_line': 'Stakes: 7',
                   'json_object': '{"stakes": 7}', 'fraction': '7/10',
                   'tagged': '<stakes>7</stakes>'}
        for phrasing_id, output_format, _, wrapper, pattern in rating_phrasings.PHRASINGS:
            # Every wrapper forbids reasoning, or a ten-token answer never arrives.
            self.assertRegex(wrapper.lower(), r'no (explanation|reasoning|other words)|nothing else')
            self.assertEqual(rating_phrasings.parse_rating(pattern, replies[output_format]), 7)
            self.assertIsNone(rating_phrasings.parse_rating(pattern, 'I would say seven'))
            self.assertIsNone(rating_phrasings.parse_rating(pattern, replies[output_format]
                                                            .replace('7', '99')))

        changed = [(entry[0], entry[1], entry[2], entry[3].replace('10', 'ten'), entry[4])
                   if index == 0 else entry
                   for index, entry in enumerate(rating_phrasings.PHRASINGS)]
        with patch.object(rating_phrasings, 'PHRASINGS', changed):
            with self.assertRaisesRegex(ValueError, 'scale bounds'):
                rating_phrasings.check_phrasings()

    @contextmanager
    def mock_rating_model(self):
        class Tokenizer:
            pad_token_id = 0

            def __init__(self):
                self.batch = []

            def __call__(self, texts):
                self.batch = list(texts)
                ids = torch.zeros((len(texts), 4), dtype=torch.long)
                return {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}

            def batch_decode(self, tokens, skip_special_tokens=True):
                return [self.reply(text) for text in self.batch]

            @staticmethod
            def reply(text):
                rating = 9 if 'Ebola' in text else 2
                if 'single integer and no explanation' in text:
                    return 'I would say seven'  # deliberate parse failure
                if '<stakes>' in text:
                    return f'<stakes>{rating}</stakes>'
                if 'JSON object' in text:
                    return f'{{"stakes": {rating}}}'
                if 'Stakes: <integer' in text:
                    return f'Stakes: {rating}'
                if 'N/10' in text:
                    return f'{rating}/10'
                return str(rating)

        class Model:
            device = torch.device('cpu')

            def generate(self, input_ids, attention_mask, max_new_tokens, min_new_tokens,
                         do_sample, pad_token_id):
                assert min_new_tokens == max_new_tokens and do_sample is False
                return torch.zeros((input_ids.shape[0], input_ids.shape[1] + max_new_tokens),
                                   dtype=torch.long)

        with patch.object(cache_activations, 'load_model',
                          return_value=(Model(), Tokenizer())) as loader:
            yield loader

    def test_stated_ratings_are_cached_parsed_and_row_aligned(self):
        records = severity_pairwise_prompts.build_prompt_records()
        build = severity_pairwise_prompts.build_prompt_records
        directory = self.config.run_dir / 'inference' / 'severity_pairwise'
        phrasings = len(rating_phrasings.PHRASINGS)
        with self.mock_rating_model() as loader:
            csv, summary_csv, table, summary = stated_stakes.run(
                self.config, 'severity_pairwise', build, max_new_tokens=4)
            self.assertEqual(loader.call_count, 1)
            caches = sorted((directory / 'ratings').glob('ratings--*.json'))
            times = {path: path.stat().st_mtime_ns for path in caches}
            loader.reset_mock()
            stated_stakes.run(self.config, 'severity_pairwise', build, max_new_tokens=4)
            loader.assert_not_called()
            self.assertEqual(times, {path: path.stat().st_mtime_ns for path in times})

        self.assertEqual(len(caches), len({r['template_id'] for r in records}))
        self.assertEqual(len(table), len(records) * phrasings)
        self.assertEqual(csv, directory / 'stated_stakes.csv')
        self.assertEqual(list(pd.read_csv(csv).columns), stated_stakes.CSV_COLUMNS)
        self.assertEqual(list(pd.read_csv(summary_csv).columns), stated_stakes.SUMMARY_COLUMNS)
        # source_row indexes build_prompt_records(), which is the arc-length CSV order.
        self.assertEqual(sorted(summary.source_row.tolist()), list(range(len(records))))
        self.assertEqual(table.groupby('source_row').size().unique().tolist(), [phrasings])
        for row in table.itertuples():
            self.assertEqual(row.prompt, records[row.source_row]['text'])
        # Five of six wrappers comply; the sixth is recorded as a parse failure.
        self.assertEqual(summary.ratings_parsed.unique().tolist(), [phrasings - 1])
        self.assertEqual(summary.rating_spread.unique().tolist(), [0.0])
        expected = [9.0 if 'Ebola' in record['text'] else 2.0
                    for record in records]
        self.assertEqual(summary.sort_values('source_row').rating_median.tolist(), expected)
        rates = stated_stakes.parse_rate(table)
        self.assertEqual(rates.set_index('phrasing').rate.min(), 0.0)
        self.assertEqual(len(rates), phrasings)

        cached = json.loads(caches[0].read_text(encoding='utf-8'))
        self.assertTrue(all(len(tokens) == 4 for tokens in cached['generated_tokens']))
        self.assertEqual(cached['config']['min_new_tokens'], 4)
        self.assertEqual(cached['config']['do_sample'], False)
        fitting = (list(self.config.activations_dir.rglob('*'))
                   if self.config.activations_dir.exists() else [])
        self.assertEqual([path for path in fitting if path.is_file()], [])

    def test_magnitude_dataset_sweeps_one_quantity_in_two_renderings(self):
        families = severity_magnitude_prompts.FAMILIES
        formats = severity_magnitude_prompts.NUMBER_FORMATS
        records = severity_magnitude_prompts.build_prompt_records()
        rungs = sum(len(family[4]) for family in families)
        self.assertEqual((len(families), formats), (5, ('numeric', 'words')))
        self.assertEqual(len(records), rungs * len(formats))
        self.assertEqual(len({record['text'] for record in records}), len(records))
        reference = {template[0]: template for template in TEMPLATES}
        # The carried-over family reproduces the reference prompts exactly, so the
        # two datasets can be checked against each other.
        self.assertIn('savings', reference)
        savings = [record for record in records
                   if record['task'] == 'savings' and record['number_format'] == 'numeric']
        expected = [reference['savings'][2].replace('{amount}', value)
                    for value in reference['savings'][3]]
        self.assertEqual([record['text'] for record in savings], expected)
        for family_id, _, unit, template, values in families:
            rows = [record for record in records if record['task'] == family_id]
            self.assertEqual(len(rows), len(values) * len(formats))
            self.assertEqual([row['number_format'] for row in rows],
                             [fmt for fmt in formats for _ in values])
            # Only the quantity moves: every prompt is the same sentence filled in.
            for row in rows:
                self.assertEqual(row['unit'], unit)
                self.assertEqual(row['text'], template.format(value=row['value_text']))
                for field in ('base_value', 'base_unit', 'unit_variant'):
                    self.assertNotIn(field, row)
                self.assertAlmostEqual(row['task_metadata']['log10_value'],
                                       math.log10(row['value']), places=5)
                self.assertNotIn('stakes', row['task_metadata'])
            self.assertEqual([row['value'] for row in rows],
                             [value[0] for _ in formats for value in values])
            spanned = math.log10(values[-1][0] / values[0][0])
            self.assertGreaterEqual(spanned, severity_magnitude_prompts.MINIMUM_DECADES)

        def edited(values):
            changed = copy.deepcopy(families)
            changed[1] = changed[1][:4] + (values,)
            return patch.object(severity_magnitude_prompts, 'FAMILIES', changed)

        rungs = list(families[1][4])
        with edited(rungs[::-1]):
            with self.assertRaisesRegex(ValueError, 'strictly increase'):
                severity_magnitude_prompts.build_prompt_records()
        with edited(rungs[:4]):
            with self.assertRaisesRegex(ValueError, 'at least five rungs'):
                severity_magnitude_prompts.build_prompt_records()
        # Five rungs, but crowded into two decades: too flat to fit a slope on.
        with edited([(1, '1 user', 'one user'), (2, '2 users', 'two users'),
                     (5, '5 users', 'five users'), (20, '20 users', 'twenty users'),
                     (100, '100 users', 'one hundred users')]):
            with self.assertRaisesRegex(ValueError, 'decades'):
                severity_magnitude_prompts.build_prompt_records()

    def test_composition_dataset_pairs_conjunctions_with_both_controls(self):
        sets = severity_composition_prompts.HARM_SETS
        roles = severity_composition_prompts.ROLES
        inert = severity_composition_prompts.INERT_CLAUSE
        records = severity_composition_prompts.build_prompt_records()
        self.assertEqual((len(sets), len(roles)), (6, 6))
        self.assertEqual(len(records), len(sets) * len(roles))
        self.assertLess(len(records), severity_composition_prompts.PROMPT_BUDGET)
        self.assertEqual(len({record['text'] for record in records}), len(records))
        self.assertEqual({entry[1] for entry in sets},
                         set(severity_composition_prompts.PAIR_TYPES))
        for set_id, pair_type, harm_a, harm_b in sets:
            rows = {record['task_metadata']['role']: record
                    for record in records if record['task'] == set_id}
            self.assertEqual(tuple(rows), roles)
            compose = severity_composition_prompts.compose
            # Every prompt is built from the same clauses; nothing new is introduced.
            self.assertEqual(rows['a']['text'], compose([harm_a]))
            self.assertEqual(rows['b']['text'], compose([harm_b]))
            self.assertEqual(rows['a_and_b']['text'], compose([harm_a, harm_b]))
            self.assertEqual(rows['b_and_a']['text'], compose([harm_b, harm_a]))
            self.assertEqual(rows['a_and_inert']['text'], compose([harm_a, inert]))
            self.assertEqual(rows['b_and_inert']['text'], compose([harm_b, inert]))
            # The order control is length-matched; the inert control is harm-matched.
            forward, reverse = rows['a_and_b']['text'], rows['b_and_a']['text']
            self.assertNotEqual(forward, reverse)
            self.assertEqual(len(forward), len(reverse))
            self.assertEqual(len(forward.split()), len(reverse.split()))
            for role, row in rows.items():
                metadata = row['task_metadata']
                self.assertEqual(metadata['pair_type'], pair_type)
                self.assertEqual(metadata['harm_count'], 2 if role in ('a_and_b', 'b_and_a') else 1)
                self.assertEqual(metadata['clause_count'], 1 if role in ('a', 'b') else 2)
                self.assertEqual(metadata['usage'], 'inference_only')
                self.assertNotIn('stakes', metadata)
                self.assertTrue(row['text'].endswith(severity_composition_prompts.REQUEST))
                # The final clause is what a final-token readout could be reading.
                leading = harm_a if role.startswith('a') else harm_b
                self.assertTrue(row['text'].startswith(leading))
            for role in ('a_and_inert', 'b_and_inert'):
                self.assertEqual(rows[role]['task_metadata']['final_clause'], 'inert')
            self.assertEqual(rows['a_and_b']['task_metadata']['final_clause'], 'b')
            self.assertEqual(rows['b_and_a']['task_metadata']['final_clause'], 'a')

        def edited(index, entry):
            changed = copy.deepcopy(sets)
            changed[index] = entry
            return patch.object(severity_composition_prompts, 'HARM_SETS', changed)

        with edited(0, sets[0][:2] + ('My umbrella is lost', sets[0][3])):
            with self.assertRaisesRegex(ValueError, "start with 'I'"):
                severity_composition_prompts.build_prompt_records()
        with edited(0, sets[0][:2] + (sets[0][2], 'I broke a mug')):
            with self.assertRaisesRegex(ValueError, 'reused elsewhere'):
                severity_composition_prompts.build_prompt_records()
        with edited(2, ('shirt_illness', 'medium_low') + sets[2][2:]):
            with self.assertRaisesRegex(ValueError, 'Every pair type'):
                severity_composition_prompts.build_prompt_records()

    def test_composition_inference_is_isolated_and_aligned(self):
        records = severity_composition_prompts.build_prompt_records()
        directory = self.config.run_dir / 'inference' / 'severity_composition'
        with self.mock_model() as (loader, _):
            csv, result, _ = severity_composition_inference.run(self.config)
            self.assertEqual(loader.call_count, 1)
            paths = sorted((directory / 'activations').glob('*.pt'))
        self.assertEqual(len(paths), len(severity_composition_prompts.HARM_SETS))
        self.assertTrue(all(path.name.startswith('severity_composition_inference--')
                            for path in paths))
        self.assertEqual(csv, directory / 'severity_composition_arc_lengths.csv')
        self.assertEqual(list(pd.read_csv(csv).columns),
                         severity_composition_inference.CSV_COLUMNS)
        self.assertEqual(result.role.tolist(),
                         [record['task_metadata']['role'] for record in records])
        self.assertEqual(result.prompt.tolist(), [record['text'] for record in records])
        reordered, _ = severity_composition_inference.project_caches(
            self.config, records, paths[::-1])
        pd.testing.assert_frame_equal(result, reordered)

    def test_magnitude_inference_is_isolated_and_aligned(self):
        records = severity_magnitude_prompts.build_prompt_records()
        directory = self.config.run_dir / 'inference' / 'severity_magnitude'
        with self.mock_model() as (loader, _):
            csv, result, _ = severity_magnitude_inference.run(self.config)
            self.assertEqual(loader.call_count, 1)
            paths = sorted((directory / 'activations').glob('*.pt'))
        self.assertEqual(len(paths), len(severity_magnitude_prompts.FAMILIES))
        self.assertTrue(all(path.name.startswith('severity_magnitude_inference--')
                            for path in paths))
        self.assertEqual(csv, directory / 'severity_magnitude_arc_lengths.csv')
        self.assertEqual(list(pd.read_csv(csv).columns),
                         severity_magnitude_inference.CSV_COLUMNS)
        self.assertEqual(result.value.tolist(), [record['value'] for record in records])
        self.assertEqual(result.number_format.tolist(),
                         [record['number_format'] for record in records])
        np.testing.assert_allclose(result.log10_value, np.log10(result.value.to_numpy()),
                                   atol=1e-5)
        reordered, _ = severity_magnitude_inference.project_caches(
            self.config, records, paths[::-1])
        pd.testing.assert_frame_equal(result, reordered)

    def test_null_inference_is_isolated_and_aligned(self):
        records = severity_null_prompts.build_prompt_records()
        directory = self.config.run_dir / 'inference' / 'severity_null'
        with self.mock_model() as (loader, _):
            csv, result, _ = severity_null_inference.run(self.config)
            self.assertEqual(loader.call_count, 1)
            paths = sorted((directory / 'activations').glob('*.pt'))
            self.assertEqual(len(paths), len(severity_null_prompts.FAMILIES))
            self.assertTrue(all(path.name.startswith('severity_null_inference--')
                                for path in paths))
        self.assertEqual(csv, directory / 'severity_null_arc_lengths.csv')
        self.assertEqual(list(pd.read_csv(csv).columns), severity_null_inference.CSV_COLUMNS)
        self.assertEqual(result.severity_word.tolist(),
                         [record['task_metadata']['severity_word'] for record in records])
        self.assertEqual(result.frame.tolist(),
                         [record['task_metadata']['frame'] for record in records])
        reordered, _ = severity_null_inference.project_caches(self.config, records, paths[::-1])
        pd.testing.assert_frame_equal(result, reordered)
        fitting = (list(self.config.activations_dir.rglob('*.pt'))
                   if self.config.activations_dir.exists() else [])
        self.assertTrue(all('null' not in path.name for path in fitting))

    def test_rating_pass_loads_the_model_once_for_every_corpus(self):
        datasets = ['severity_pairwise', 'severity_wording']
        with self.mock_rating_model() as loader:
            results = stated_stakes.run_all(self.config, datasets, max_new_tokens=4)
            self.assertEqual(loader.call_count, 1)
        self.assertEqual(list(results), datasets)
        for name in datasets:
            csv, summary_csv, table, summary = results[name]
            expected = len(stated_stakes.DATASETS[name]())
            self.assertEqual(csv, self.config.run_dir / 'inference' / name / 'stated_stakes.csv')
            self.assertEqual(len(summary), expected)
            self.assertEqual(len(table), expected * len(rating_phrasings.PHRASINGS))
            self.assertTrue(summary_csv.exists())

    def test_rating_records_cover_every_prompt_and_wrapper(self):
        records = severity_null_prompts.build_prompt_records()[:3]
        rating_records = stated_stakes.build_rating_records(records)
        self.assertEqual(len(rating_records), len(records) * len(rating_phrasings.PHRASINGS))
        self.assertEqual(len({row['text'] for row in rating_records}), len(rating_records))
        for row in rating_records:
            self.assertIn(records[row['source_row']]['text'], row['text'])
            self.assertEqual(row['template_id'], records[row['source_row']]['template_id'])
        with self.assertRaisesRegex(ValueError, 'must not be empty'):
            stated_stakes.build_rating_records([])
        with self.assertRaisesRegex(ValueError, 'Unknown rating datasets'):
            stated_stakes.run_all(self.config, ['not_a_dataset'])

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
                    self.assertNotIn(field, row)

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
                    self.assertNotIn(field, row)

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
                    self.assertNotIn(field, row)
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
        training = self.as_training_records(self.records[:2])
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
        training = self.as_training_records(self.records[:2])
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
        training = self.as_training_records(self.records[:2])
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
