"""Horizon-free prompt corpora: prompts that never state a time horizon.

`base_task_set` is not a corpus here; it supplies the task metadata
schema and base task set that `horizon_free_task_set` builds on. Note the names of the
two original corpora are historical and describe the opposite of their contents:

    conversational_no_time -> bare prompts, 3 templates  ("Task: X\n\nWrite a plan.",
                              a planning brief, and a working-notes heading)
    task_only              -> conversational prompts, 12 templates
                              ("Could you help me plan how to X?")

`scripts.pipeline_config.REGISTERS` maps each to the register label used in the analysis, so
downstream code never has to rely on the filenames.
"""

from scripts.corpora.training.generate_prompts import DATASETS, generate_task_dataset

__all__ = ["DATASETS", "generate_task_dataset"]
