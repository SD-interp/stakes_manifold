"""Third-person planning prompts that state the task and nothing about time.

Every existing horizon-free corpus puts a person in the prompt: ``task_only`` and
the verbose register speak as "I", and the bare and directive registers address
the model as "you". If the readout is partly tracking who is being asked rather
than what is being planned, none of those could reveal it. These prompts remove
the speaker entirely - no first-person pronoun, no direct address - and describe
the work as something a third party undertakes.

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
        "id": "a_team_needs_to",
        "template": "A team needs to {task}. What would a sensible plan look like?",
        "perspective": "third_person",
        "request_style": "impersonal",
    },
    {
        "id": "consider_the_problem",
        "template": (
            "Consider the problem of how to {task}. What does a workable plan involve?"
        ),
        "perspective": "third_person",
        "request_style": "impersonal",
    },
    {
        "id": "what_does_it_take",
        "template": "What does it take to {task}?",
        "perspective": "third_person",
        "request_style": "impersonal",
    },
    {
        "id": "the_objective_is",
        "template": (
            "The objective is to {task}. Describe an approach that would achieve it."
        ),
        "perspective": "third_person",
        "request_style": "impersonal",
    },
    {
        "id": "an_organisation_wants",
        "template": (
            "An organisation wants to {task}. How should the work be approached?"
        ),
        "perspective": "third_person",
        "request_style": "impersonal",
    },
    {
        "id": "suppose_someone_set_out",
        "template": (
            "Suppose someone set out to {task}. What would a reasonable plan cover?"
        ),
        "perspective": "third_person",
        "request_style": "impersonal",
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
