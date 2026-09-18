"""Inference-only situated-context dataset with isolated outputs."""
from . import inference_projection
from .corpora.inference.situated_context_prompts import build_prompt_records


CSV_COLUMNS = ['task', 'setting', 'stakes_rank', 'context_axis', 'is_control', 'prompt',
               'arc_length_parallel', 'arc_length_orthogonal']

# Carried through so a task's ladder can be read, and its two controls found,
# without rejoining the prompt file.
_METADATA_COLUMNS = ('setting', 'stakes_rank', 'is_control')


def _row_fields(record):
    metadata = record['task_metadata']
    return dict(
        task=record['task'],
        **{name: metadata[name] for name in _METADATA_COLUMNS},
        context_axis=record['template_metadata']['context_axis'],
        prompt=record['text'],
    )


DATASET = inference_projection.InferenceDataset(
    name='situated_context', label='Situated context',
    cache_namespace='situated_context_inference',
    csv_name='situated_context_arc_lengths.csv', build_records=build_prompt_records,
    row_fields=_row_fields, csv_columns=CSV_COLUMNS)

# The corpus is one object; these are the names the notebooks and tests call it by.
cache = DATASET.cache
project = DATASET.project
run = DATASET.run
project_caches = DATASET.project_caches
