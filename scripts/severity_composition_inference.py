"""Inference-only conjunction dataset with isolated outputs."""
from . import inference_projection
from .corpora.inference.severity_composition_prompts import build_prompt_records

CSV_COLUMNS = ['task', 'role', 'pair_type', 'harm_count', 'clause_count', 'final_clause',
               'prompt', 'arc_length_parallel', 'arc_length_orthogonal']

# Carried through so a conjunction can be compared against its own singles
# without rejoining the prompt file.
_METADATA_COLUMNS = ('role', 'pair_type', 'harm_count', 'clause_count', 'final_clause')


def _row_fields(record):
    metadata = record['task_metadata']
    return dict(task=record['task'],
                **{name: metadata[name] for name in _METADATA_COLUMNS},
                prompt=record['text'])


def project_caches(config, records, paths, bundle=None):
    return inference_projection.project_caches(
        config, records, paths, row_fields=_row_fields, csv_columns=CSV_COLUMNS, bundle=bundle)


def run(config, force=False, model=None, tokenizer=None):
    return inference_projection.run(
        config, build_records=build_prompt_records, dataset_name='severity_composition',
        cache_namespace='severity_composition_inference',
        csv_name='severity_composition_arc_lengths.csv',
        row_fields=_row_fields, csv_columns=CSV_COLUMNS, force=force, model=model, tokenizer=tokenizer)
