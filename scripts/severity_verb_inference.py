"""Inference-only verb-frame control with isolated outputs."""
from . import inference_projection
from .corpora.inference.severity_verb_prompts import build_prompt_records

CSV_COLUMNS = ['predicament', 'verb_frame', 'mood', 'object_pronoun', 'situation', 'prompt',
               'arc_length_parallel', 'arc_length_orthogonal']


def _row_fields(record):
    metadata = record['task_metadata']
    return dict(predicament=record['template_id'],
                verb_frame=metadata['verb_frame'],
                mood=metadata['mood'],
                object_pronoun=metadata['object_pronoun'],
                situation=record['template_metadata']['situation'],
                prompt=record['text'])


DATASET = inference_projection.InferenceDataset(
    name='severity_verb', label='Verb frame', cache_namespace='severity_verb_inference',
    csv_name='severity_verb_arc_lengths.csv', build_records=build_prompt_records,
    row_fields=_row_fields, csv_columns=CSV_COLUMNS)

# The corpus is one object; these are the names the notebooks and tests call it by.
cache = DATASET.cache
project = DATASET.project
run = DATASET.run
project_caches = DATASET.project_caches
