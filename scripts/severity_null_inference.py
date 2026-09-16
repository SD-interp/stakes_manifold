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


def project_caches(config, records, paths, bundle=None):
    return inference_projection.project_caches(
        config, records, paths, row_fields=_row_fields, csv_columns=CSV_COLUMNS, bundle=bundle)


def run(config, force=False):
    return inference_projection.run(
        config, build_records=build_prompt_records, dataset_name='severity_null',
        cache_namespace='severity_null_inference', csv_name='severity_null_arc_lengths.csv',
        row_fields=_row_fields, csv_columns=CSV_COLUMNS, force=force)
