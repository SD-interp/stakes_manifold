"""Inference-only severity-vocabulary null control with isolated outputs."""
from . import inference_projection
from .corpora.inference.severity_null_prompts import build_prompt_records

CSV_COLUMNS = ['template', 'severity_word', 'frame', 'mentions_referent',
               'arc_length_parallel', 'arc_length_orthogonal']


def _row_fields(record):
    metadata = record['task_metadata']
    return dict(template=record['template_metadata']['template'],
                severity_word=metadata['severity_word'],
                frame=metadata['frame'],
                mentions_referent=metadata['mentions_referent'])


DATASET = inference_projection.InferenceDataset(
    name='severity_null', label='Null severity', cache_namespace='severity_null_inference',
    csv_name='severity_null_arc_lengths.csv', build_records=build_prompt_records,
    row_fields=_row_fields, csv_columns=CSV_COLUMNS)

# The corpus is one object; these are the names the notebooks and tests call it by.
cache = DATASET.cache
project = DATASET.project
run = DATASET.run
project_caches = DATASET.project_caches
