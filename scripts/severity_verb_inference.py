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


def project_caches(config, records, paths, bundle=None):
    return inference_projection.project_caches(
        config, records, paths, row_fields=_row_fields, csv_columns=CSV_COLUMNS, bundle=bundle)


def run(config, force=False, model=None, tokenizer=None):
    return inference_projection.run(
        config, build_records=build_prompt_records, dataset_name='severity_verb',
        cache_namespace='severity_verb_inference', csv_name='severity_verb_arc_lengths.csv',
        row_fields=_row_fields, csv_columns=CSV_COLUMNS, force=force, model=model, tokenizer=tokenizer)
