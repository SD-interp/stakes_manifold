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


def project_caches(config, records, paths, bundle=None):
    return inference_projection.project_caches(
        config, records, paths, row_fields=_row_fields, csv_columns=CSV_COLUMNS, bundle=bundle)


def run(config, force=False, model=None, tokenizer=None):
    return inference_projection.run(
        config, build_records=build_prompt_records, dataset_name='severity_magnitude',
        cache_namespace='severity_magnitude_inference',
        csv_name='severity_magnitude_arc_lengths.csv',
        row_fields=_row_fields, csv_columns=CSV_COLUMNS, force=force, model=model, tokenizer=tokenizer)
