"""Inference-only matched context dataset with isolated outputs."""
from . import inference_projection
from .corpora.inference.context_prompts import build_prompt_records


CSV_COLUMNS = ['task', 'context', 'context_position', 'ladder_rank', 'reality_status',
               'answer_use', 'is_control', 'phrasing_index', 'prompt',
               'arc_length_parallel', 'arc_length_orthogonal']

# Carried through so a condition can be averaged over its interchangeable
# phrasings, and so the ladder can be read without rejoining the prompt file.
_METADATA_COLUMNS = ('context', 'context_position', 'ladder_rank', 'reality_status',
                     'answer_use', 'is_control', 'phrasing_index')


def _row_fields(record):
    metadata = record['task_metadata']
    return dict(
        task=record['task'],
        **{name: metadata[name] for name in _METADATA_COLUMNS},
        prompt=record['text'],
    )


def project_caches(config, records, paths, bundle=None):
    return inference_projection.project_caches(
        config, records, paths, row_fields=_row_fields,
        csv_columns=CSV_COLUMNS, bundle=bundle)


def run(config, force=False, model=None, tokenizer=None):
    return inference_projection.run(
        config,
        build_records=build_prompt_records,
        dataset_name='context',
        cache_namespace='context_inference',
        csv_name='context_arc_lengths.csv',
        row_fields=_row_fields,
        csv_columns=CSV_COLUMNS,
        force=force,
        model=model,
        tokenizer=tokenizer,
    )
