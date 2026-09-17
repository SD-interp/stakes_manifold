"""Inference-only prompt-length control with isolated outputs."""
from . import inference_projection
from .corpora.inference.severity_length_prompts import build_prompt_records

CSV_COLUMNS = ['template', 'severity_word', 'pad_level', 'pad_position', 'pad_words',
               'arc_length_parallel', 'arc_length_orthogonal']


def _row_fields(record):
    metadata = record['task_metadata']
    return dict(template=record['template_metadata']['template'],
                severity_word=metadata['severity_word'],
                pad_level=metadata['pad_level'],
                pad_position=metadata['pad_position'],
                pad_words=metadata['pad_words'])


def project_caches(config, records, paths, bundle=None):
    return inference_projection.project_caches(
        config, records, paths, row_fields=_row_fields, csv_columns=CSV_COLUMNS, bundle=bundle)


def run(config, force=False, model=None, tokenizer=None):
    return inference_projection.run(
        config, build_records=build_prompt_records, dataset_name='severity_length',
        cache_namespace='severity_length_inference', csv_name='severity_length_arc_lengths.csv',
        row_fields=_row_fields, csv_columns=CSV_COLUMNS, force=force, model=model, tokenizer=tokenizer)
