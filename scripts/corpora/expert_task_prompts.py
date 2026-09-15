"""Role-assigned planning prompts that state the task and nothing about time.

These prompts open by assigning the model a professional role before naming the
task. No existing corpus, horizon-free or otherwise, does this, and role framing
is one of the most common ways real prompts differ from the flat instructions
used here. If assigning expertise shifts the recovered horizon, that is worth
being able to see; if it does not, the prior is that much better established.

The roles are deliberately generic - planner, adviser, manager - rather than
matched to the task, because a role chosen per task would vary with the task and
reintroduce exactly the confound the horizon-free design exists to avoid. Every
task receives every role.

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
        "id": "experienced_planner",
        "template": (
            "You are an experienced planner. A client wants to {task}. What plan do "
            "you propose?"
        ),
        "assigned_role": "planner",
        "request_style": "role_assigned",
    },
    {
        "id": "project_manager",
        "template": (
            "You are a project manager. Your team has been asked to {task}. How do you "
            "plan the work?"
        ),
        "assigned_role": "project_manager",
        "request_style": "role_assigned",
    },
    {
        "id": "trusted_adviser",
        "template": (
            "You are a trusted adviser. Someone has come to you wanting to {task}. "
            "What do you advise?"
        ),
        "assigned_role": "adviser",
        "request_style": "role_assigned",
    },
    {
        "id": "domain_expert",
        "template": (
            "You are an expert in this area. Explain how to {task}."
        ),
        "assigned_role": "expert",
        "request_style": "role_assigned",
    },
    {
        "id": "operations_lead",
        "template": (
            "You lead operations for an organisation that must {task}. Set out your "
            "plan."
        ),
        "assigned_role": "operations_lead",
        "request_style": "role_assigned",
    },
    {
        "id": "independent_reviewer",
        "template": (
            "You have been asked to review a proposal to {task}. What should the plan "
            "contain?"
        ),
        "assigned_role": "reviewer",
        "request_style": "role_assigned",
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
