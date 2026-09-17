"""Inference-only prompt-length control with isolated outputs."""
from . import inference_projection
from .corpora.inference.severity_length_prompts import build_prompt_records

CSV_COLUMNS = ['template', 'severity_word', 'pad_level', 'pad_variant', 'pad_position',
               'pad_words', 'arc_length_parallel', 'arc_length_orthogonal']


def _row_fields(record):
    metadata = record['task_metadata']
    return dict(template=record['template_metadata']['template'],
                severity_word=metadata['severity_word'],
                pad_level=metadata['pad_level'],
                pad_variant=metadata['pad_variant'],
                pad_position=metadata['pad_position'],
                pad_words=metadata['pad_words'])


DATASET = inference_projection.InferenceDataset(
    name='severity_length', label='Prompt length', cache_namespace='severity_length_inference',
    csv_name='severity_length_arc_lengths.csv', build_records=build_prompt_records,
    row_fields=_row_fields, csv_columns=CSV_COLUMNS)

# The corpus is one object; these are the names the notebooks and tests call it by.
cache = DATASET.cache
project = DATASET.project
run = DATASET.run
project_caches = DATASET.project_caches
