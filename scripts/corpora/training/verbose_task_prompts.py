"""Elaborated planning prompts that state the task and nothing about time.

This is the horizon-free counterpart of :mod:`nu_d.corpora.verbose_time_prompts`
(dataset ID ``plain_long``), which had none. The eight voices, their IDs and their
metadata are carried over unchanged; only the time clause is gone, along with the
surrounding phrases that referred to it.

The register matters because the existing horizon-free corpora are both short:
three bare field-form templates and twelve one-sentence conversational ones. If
prompt length or elaboration shifts the readout, no existing control could show
it. These prompts are long and discursive while still supplying no duration.

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
        "id": "direct_help",
        "template": (
            "Could you help me plan how to {task}? Please keep the advice realistic, "
            "practical, and grounded. I want a clear, actionable plan I can actually "
            "follow, not just vague or generic suggestions, so feel free to be "
            "specific about what to do first, next, and last."
        ),
        "perspective": "first_person",
        "request_style": "direct",
    },
    {
        "id": "practical_walkthrough",
        "template": (
            "I need to {task}. Walk me through a practical, step-by-step way to pull "
            "it off, including any preparation, key milestones, and potential pitfalls "
            "I should watch out for along the way. Assume I'm starting from scratch "
            "and want a concrete, no-nonsense plan rather than abstract advice."
        ),
        "perspective": "first_person",
        "request_style": "walkthrough",
    },
    {
        "id": "hypothetical_advice",
        "template": (
            "Imagine you had to {task}. How would you approach it? What order would "
            "you tackle things in, what would you prioritize, and what would you "
            "deliberately skip or postpone?"
        ),
        "perspective": "second_person",
        "request_style": "hypothetical",
    },
    {
        "id": "starting_now",
        "template": (
            "If I started now and wanted to {task}, what would a sensible plan look "
            "like? I'd like a rough outline with clear checkpoints, so I can tell "
            "early on whether I'm on track or falling behind and need to adjust my "
            "approach."
        ),
        "perspective": "first_person",
        "request_style": "reflective",
    },
    {
        "id": "use_the_time",
        "template": (
            "What's the best way for me to {task}? I want to make the most efficient, "
            "effective use of my effort, avoiding wasted work, so please suggest how "
            "to sequence tasks and where to focus my energy."
        ),
        "perspective": "first_person",
        "request_style": "optimization",
    },
    {
        "id": "limited_availability",
        "template": (
            "I'd like to {task}. What should I focus on? I'd rather do a few important "
            "things well than spread myself thin, so help me identify the highest-"
            "impact priorities and what can safely be left out or deferred."
        ),
        "perspective": "first_person",
        "request_style": "prioritization",
    },
    {
        "id": "external_deadline",
        "template": (
            "Someone has asked me to {task}. Can you suggest a realistic approach, "
            "including how to pace myself, what to prioritize, and how to avoid "
            "a frantic scramble at the end?"
        ),
        "perspective": "first_person",
        "request_style": "advice",
    },
    {
        "id": "break_it_down",
        "template": (
            "My goal is to {task}. How would you break that down into smaller, "
            "manageable steps or phases? I want to understand how the pieces fit "
            "together and roughly how much each part should reasonably involve."
        ),
        "perspective": "first_person",
        "request_style": "decomposition",
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
