"""Inference-only curated pairwise severity flips with isolated outputs."""
from . import inference_projection
from .corpora.inference.severity_pairwise_prompts import build_prompt_records

CSV_COLUMNS = ['template', 'severity_word', 'arc_length_parallel', 'arc_length_orthogonal']


def _row_fields(record):
    return dict(template=record['template_metadata']['template'],
                severity_word=record['task_metadata']['severity_word'])


def project_caches(config, records, paths, bundle=None):
    return inference_projection.project_caches(
        config, records, paths, row_fields=_row_fields, csv_columns=CSV_COLUMNS, bundle=bundle)


def run(config, force=False):
    return inference_projection.run(
        config, build_records=build_prompt_records, dataset_name='severity_pairwise',
        cache_namespace='severity_pairwise_inference', csv_name='severity_pairwise_arc_lengths.csv',
        row_fields=_row_fields, csv_columns=CSV_COLUMNS, force=force)
