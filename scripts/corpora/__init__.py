"""Prompt corpora, split by the role a corpus plays in the pipeline.

`training/` holds the horizon-free corpora that are generated, cached, and fitted
on: the task metadata schema (`base_task_set`), the task set built on it
(`horizon_free_task_set`), the register modules, and the generator that crosses
them (`generate_prompts`). They import one another freely.

`inference/` holds the inference-only evaluation corpora (severity, its flipped,
pairwise, and wording variants, and the context variations). Each of those is
self-contained and exposes the same `build_prompt_records` interface, so nothing
in `training/` is needed to project them onto an already-fitted manifold.

The generator entry points stay re-exported here so downstream code can keep
importing `scripts.corpora` without knowing which subpackage a corpus lives in.
"""

from scripts.corpora.training import DATASETS, generate_task_dataset

__all__ = ["DATASETS", "generate_task_dataset"]
