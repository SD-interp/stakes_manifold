"""Bare, unaddressed planning prompts with no time horizon.

The register is a written brief rather than a request from one person to another:
nobody is addressed, and the task arrives as a stated item followed by an
instruction to plan. That is what separates this dataset from ``task_only``, which
asks a first-person question, and from ``directive_no_horizon``, which addresses
the model in the imperative.

The three templates were once ``Task:``, ``Goal:`` and ``Objective:`` followed by
byte-identical text, which made them one template form wearing three labels. Since
every template carries equal mass in the fits downstream, that gave this single
form three times the weight it should have had. The two replacements keep the bare
register but differ in structure and in the instruction line, not just in the
label word.
"""

from __future__ import annotations

try:
    from . import horizon_free_task_set
except ImportError:  # Support running scripts/corpora/generate_prompts.py directly.
    import horizon_free_task_set  # type: ignore


templates = [
    {
        "id": "task",
        "template": "Task: {task}\n\nWrite a plan.",
        "prompt_framing": "task",
    },
    {
        "id": "brief",
        "template": (
            "PLANNING BRIEF\n\nWhat needs doing: {task}\nWhat is needed back: an "
            "ordered plan of action, with the first step named."
        ),
        "prompt_framing": "brief",
    },
    {
        "id": "notes_heading",
        "template": (
            "{task}\n\nNotes toward a plan\n-------------------\nSteps, in the "
            "order they would have to happen:"
        ),
        "prompt_framing": "working_notes",
    },
]


def _copy_task_config(config: dict[str, object]) -> dict[str, object]:
    """Copy a task config without sharing its mutable unit set."""
    copied = dict(config)
    copied["units"] = set(config["units"])  # type: ignore[arg-type]
    return copied


tasks = {task: _copy_task_config(config) for task, config in horizon_free_task_set.tasks.items()}

time_free = True
values = [1]


__all__ = ["tasks", "templates", "time_free", "values"]
