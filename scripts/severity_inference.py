"""Inference-only slot-substitution severity dataset (stable public interface)."""
from . import inference_projection
from .corpora.inference.severity_prompts import build_prompt_records

CSV_COLUMNS = ['template', 'severity_word', 'arc_length_parallel', 'arc_length_orthogonal']


def _row_fields(record):
    return dict(template=record['template_metadata']['template'],
                severity_word=record['task_metadata']['severity_word'])


DATASET = inference_projection.InferenceDataset(
    name='severity', label='Severity', cache_namespace='severity_inference',
    csv_name='severity_arc_lengths.csv', build_records=build_prompt_records,
    row_fields=_row_fields, csv_columns=CSV_COLUMNS)

# The corpus is one object; these are the names the notebooks and tests call it by.
cache = DATASET.cache
project = DATASET.project
run = DATASET.run
project_caches = DATASET.project_caches
