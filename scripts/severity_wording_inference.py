"""Inference-only expressed-distress wording dataset and separate CSV."""
from . import inference_projection
from .corpora.inference.severity_wording_prompts import build_prompt_records

CSV_COLUMNS = ['task', 'prompt', 'arc_length_parallel', 'arc_length_orthogonal']


def _row_fields(record):
    return dict(task=record['task'], prompt=record['text'])


DATASET = inference_projection.InferenceDataset(
    name='severity_wording', label='Severity wording', cache_namespace='severity_wording_inference',
    csv_name='severity_wording_arc_lengths.csv', build_records=build_prompt_records,
    row_fields=_row_fields, csv_columns=CSV_COLUMNS)

# The corpus is one object; these are the names the notebooks and tests call it by.
cache = DATASET.cache
project = DATASET.project
run = DATASET.run
project_caches = DATASET.project_caches
