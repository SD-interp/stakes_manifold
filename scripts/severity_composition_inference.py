"""Inference-only conjunction dataset with isolated outputs."""
from . import inference_projection
from .corpora.inference.severity_composition_prompts import build_prompt_records

CSV_COLUMNS = ['task', 'role', 'pair_type', 'harm_count', 'clause_count', 'final_clause',
               'prompt', 'arc_length_parallel', 'arc_length_orthogonal']

# Carried through so a conjunction can be compared against its own singles
# without rejoining the prompt file.
_METADATA_COLUMNS = ('role', 'pair_type', 'harm_count', 'clause_count', 'final_clause')


def _row_fields(record):
    metadata = record['task_metadata']
    return dict(task=record['task'],
                **{name: metadata[name] for name in _METADATA_COLUMNS},
                prompt=record['text'])


DATASET = inference_projection.InferenceDataset(
    name='severity_composition', label='Composed severity', cache_namespace='severity_composition_inference',
    csv_name='severity_composition_arc_lengths.csv', build_records=build_prompt_records,
    row_fields=_row_fields, csv_columns=CSV_COLUMNS)

# The corpus is one object; these are the names the notebooks and tests call it by.
cache = DATASET.cache
project = DATASET.project
run = DATASET.run
project_caches = DATASET.project_caches
