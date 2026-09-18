"""Ordered registry of the inference corpora both pipeline halves run.

The caching notebook walks this list on the GPU and the analysis notebook walks the
same list on the CPU, so a corpus added here is picked up by both without either
notebook being edited. The order is the order results are reported in: the reference
severity corpus first, then its controls, then the counterfactual sets, then the two
context sets: the attached framings, and the situations that carry their own stakes.
"""
from . import (context_inference, severity_composition_inference,
               severity_flipped_inference, severity_inference,
               severity_length_inference, severity_magnitude_inference,
               severity_null_inference, severity_pairwise_inference,
               severity_verb_inference, severity_wording_inference,
               situated_context_inference)

DATASETS = (
    severity_inference.DATASET,
    severity_null_inference.DATASET,
    severity_length_inference.DATASET,
    severity_magnitude_inference.DATASET,
    severity_composition_inference.DATASET,
    severity_flipped_inference.DATASET,
    severity_pairwise_inference.DATASET,
    severity_wording_inference.DATASET,
    severity_verb_inference.DATASET,
    context_inference.DATASET,
    situated_context_inference.DATASET,
)

BY_NAME = {dataset.name: dataset for dataset in DATASETS}
if len(BY_NAME) != len(DATASETS):
    raise ValueError('Two inference corpora share a dataset name.')
if len({dataset.cache_namespace for dataset in DATASETS}) != len(DATASETS):
    raise ValueError('Two inference corpora share a cache namespace.')


def by_name(name):
    if name not in BY_NAME:
        raise ValueError(f'Unknown inference dataset {name!r}; '
                         f'expected one of {sorted(BY_NAME)}')
    return BY_NAME[name]


def cached_datasets(config, datasets=DATASETS):
    """The corpora whose activations the caching half has written for this model.

    The two halves often run on different machines, so a corpus can be absent simply
    because it was added after that model was cached, or because only part of the
    artifact tree was copied across. The analysis notebook reports what it skipped
    rather than failing on the first gap.
    """
    return [dataset for dataset in datasets
            if any((dataset.directory(config) / 'activations')
                   .glob(f'{dataset.cache_namespace}--*.pt'))]
