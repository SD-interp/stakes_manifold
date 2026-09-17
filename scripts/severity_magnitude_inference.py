"""Inference-only magnitude dose-response dataset with isolated outputs."""
from . import inference_projection
from .corpora.inference.severity_magnitude_prompts import build_prompt_records

CSV_COLUMNS = ['task', 'template', 'value', 'value_text', 'unit', 'number_format',
               'log10_value', 'arc_length_parallel', 'arc_length_orthogonal']


def _row_fields(record):
    return dict(task=record['task'],
                template=record['template_metadata']['template'],
                value=record['value'],
                value_text=record['value_text'],
                unit=record['unit'],
                number_format=record['number_format'],
                log10_value=record['task_metadata']['log10_value'])


DATASET = inference_projection.InferenceDataset(
    name='severity_magnitude', label='Magnitude severity', cache_namespace='severity_magnitude_inference',
    csv_name='severity_magnitude_arc_lengths.csv', build_records=build_prompt_records,
    row_fields=_row_fields, csv_columns=CSV_COLUMNS)

# The corpus is one object; these are the names the notebooks and tests call it by.
cache = DATASET.cache
project = DATASET.project
run = DATASET.run
project_caches = DATASET.project_caches
