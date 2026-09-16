"""Inference-only matched context dataset with isolated outputs."""
from . import inference_projection
from .corpora.inference.context_prompts import build_prompt_records


CSV_COLUMNS = ['task', 'context', 'context_position', 'prompt',
               'arc_length_parallel', 'arc_length_orthogonal']


def _row_fields(record):
    return dict(
        task=record['task'],
        context=record['task_metadata']['context'],
        context_position=record['task_metadata']['context_position'],
        prompt=record['text'],
    )


def project_caches(config, records, paths, bundle=None):
    return inference_projection.project_caches(
        config, records, paths, row_fields=_row_fields,
        csv_columns=CSV_COLUMNS, bundle=bundle)


def run(config, force=False):
    return inference_projection.run(
        config,
        build_records=build_prompt_records,
        dataset_name='context',
        cache_namespace='context_inference',
        csv_name='context_arc_lengths.csv',
        row_fields=_row_fields,
        csv_columns=CSV_COLUMNS,
        force=force,
    )
