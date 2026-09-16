"""Inference-only expressed-distress wording dataset and separate CSV."""
from . import inference_projection
from .corpora.inference.severity_wording_prompts import build_prompt_records

CSV_COLUMNS = ['task', 'prompt', 'arc_length_parallel', 'arc_length_orthogonal']


def _row_fields(record):
    return dict(task=record['task'], prompt=record['text'])


def project_caches(config, records, paths, bundle=None):
    return inference_projection.project_caches(
        config, records, paths, row_fields=_row_fields, csv_columns=CSV_COLUMNS, bundle=bundle)


def run(config, force=False):
    return inference_projection.run(
        config, build_records=build_prompt_records, dataset_name='severity_wording',
        cache_namespace='severity_wording_inference', csv_name='severity_wording_arc_lengths.csv',
        row_fields=_row_fields, csv_columns=CSV_COLUMNS, force=force)
