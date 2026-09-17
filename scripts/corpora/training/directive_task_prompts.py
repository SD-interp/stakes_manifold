"""Imperative planning prompts that state the task and nothing about time.

The carrier sentences are taken from :mod:`nu_d.corpora.named_period_prompts`
(dataset ID ``named_horizon``), whose prompts are built as a period sentence
followed by an instruction. Dropping the period sentence leaves the instruction,
which is already a complete horizon-free prompt. Two further imperatives in the
same voice fill the register out.

This is a distinct register from the existing horizon-free corpora: it addresses
the model directly in the imperative with the task inline, where
``conversational_no_time`` states the task in an unaddressed written brief and
``task_only`` asks a first-person question.

Because no template interpolates ``{value}`` or ``{unit}``, the time grid is a
single placeholder value; iterating a wider grid would emit byte-identical
duplicate prompts rather than new conditions.
"""

from __future__ import annotations

try:
    from . import horizon_free_task_set
except ImportError:  # Support running scripts/corpora/generate_prompts.py directly.
    import horizon_free_task_set  # type: ignore


templates = [
    {
        "id": "write_a_plan",
        "template": "Write a plan to {task}.",
        "request_style": "imperative",
        "source_corpus": "named_horizon",
    },
    {
        "id": "describe_how",
        "template": "Describe how you would {task}.",
        "request_style": "imperative",
        "source_corpus": "named_horizon",
    },
    {
        "id": "develop_a_plan",
        "template": "Develop a plan to {task}.",
        "request_style": "imperative",
        "source_corpus": "named_horizon",
    },
    {
        "id": "outline_realistic",
        "template": "Outline a realistic plan to {task}.",
        "request_style": "imperative",
        "source_corpus": "named_horizon",
    },
    {
        "id": "set_out_the_steps",
        "template": "Set out the steps required to {task}.",
        "request_style": "imperative",
        "source_corpus": "directive",
    },
    {
        "id": "explain_how",
        "template": "Explain how to {task}.",
        "request_style": "imperative",
        "source_corpus": "directive",
    },
]

# Every template is already time-free, so removing time constraints must leave
# the prompt untouched rather than fall back to line-stripping.
for template in templates:
    template["unconstrained_template"] = template["template"]


def _copy_task_config(config: dict[str, object]) -> dict[str, object]:
    """Copy a task config without sharing its mutable unit set."""
    copied = dict(config)
    copied["units"] = set(config["units"])  # type: ignore[arg-type]
    return copied


tasks = {task: _copy_task_config(config) for task, config in horizon_free_task_set.tasks.items()}

time_free = True
values = [1]


__all__ = ["tasks", "templates", "time_free", "values"]
