"""Context-variation prompts built from one scenario per task.

Each task holds a single scenario, and every prompt is constructed by placing
exactly one framing sentence either before or after it. Nothing else differs
between a task's prompts, so the matched-task invariant holds by construction
and never has to be recovered from finished text.

The framings form a ladder over how consequential an answer would be, from a
live situation someone is about to act on, through rehearsals and retellings,
down to a world in which the events could not occur at all. ``reality_status``
and ``answer_use`` record the two properties the ladder moves along, so a main
effect can be read off either one instead of off an opaque label.

Two controls sit outside the ladder. ``none`` presents the bare scenario, and
``filler`` adds a length-matched sentence that says nothing about reality or
consequence; together they separate the effect of a framing from the effect of
having an extra sentence at all. Each framing carries three interchangeable
phrasings so that no condition is identified with one exact sentence: average
over ``phrasing_index`` when comparing conditions.

No expected score or ordering is prescribed. All prose is explicit and
independently editable.
"""

FRAME_ORDER = (
    "acting_now",
    "planning_ahead",
    "retrospective",
    "live_drill",
    "tabletop",
    "fiction",
    "game",
    "impossible",
)
CONTROL_ORDER = ("none", "filler")
CONTEXT_ORDER = FRAME_ORDER + CONTROL_ORDER
CONTEXT_POSITIONS = ("before", "after")
# The bare control has no framing sentence, so it has nowhere to place one.
BARE_POSITION = "absent"
PHRASINGS_PER_CONTEXT = 3
# Framings are held to one length band so that prompt length cannot stand in
# for the condition, and so the filler control matches the frames it replaces.
FRAMING_WORD_RANGE = (12, 16)

CONTEXTS = {
    "acting_now": {
        "reality_status": "real",
        "answer_use": "acted_now",
        "ladder_rank": 1,
        "phrasings": (
            "This is happening right now, and the answer will be acted on within minutes.",
            "This is unfolding as we speak, and whatever you advise will be done immediately.",
            "This is a live situation, and the person involved will follow this answer straight away.",
        ),
    },
    "planning_ahead": {
        "reality_status": "real",
        "answer_use": "acted_later",
        "ladder_rank": 2,
        "phrasings": (
            "This has not happened yet, and the answer will be acted on if it does.",
            "This is a real possibility being prepared for, and the answer will guide the response.",
            "This is genuine advance planning, and the person will follow this answer should it occur.",
        ),
    },
    "retrospective": {
        "reality_status": "real",
        "answer_use": "none",
        "ladder_rank": 3,
        "phrasings": (
            "This really happened last year, it was resolved then, and nothing further will be done.",
            "This actually occurred some time ago, was settled then, and no action remains to take.",
            "This is a real past event, long since concluded, and the answer changes nothing now.",
        ),
    },
    "live_drill": {
        "reality_status": "simulated",
        "answer_use": "rehearsal",
        "ladder_rank": 4,
        "phrasings": (
            "This is a staged drill in which real people rehearse their response under supervision.",
            "This is a live training exercise where actual staff practise the response on site.",
            "This is a supervised rehearsal, carried out by real participants, of how to respond.",
        ),
    },
    "tabletop": {
        "reality_status": "simulated",
        "answer_use": "none",
        "ladder_rank": 5,
        "phrasings": (
            "This is a tabletop exercise discussed on paper, with nobody carrying anything out.",
            "This is a written planning scenario talked through in a meeting, with no one acting.",
            "This is a paper exercise walked through aloud, and nothing is physically performed.",
        ),
    },
    "fiction": {
        "reality_status": "fictional",
        "answer_use": "none",
        "ladder_rank": 6,
        "phrasings": (
            "This is a scene in a novel being written, and no such event has occurred.",
            "This is invented for a short story, and nobody described here actually exists.",
            "This is a passage of fiction being drafted, and none of these people are real.",
        ),
    },
    "game": {
        "reality_status": "fictional",
        "answer_use": "none",
        "ladder_rank": 7,
        "phrasings": (
            "This is scripted content in a video game, and only game characters are involved.",
            "This is a level in a game being built, and its figures are player avatars.",
            "This is a video game scenario, and everyone in it is a game character.",
        ),
    },
    "impossible": {
        "reality_status": "fictional",
        "answer_use": "none",
        "ladder_rank": 8,
        "phrasings": (
            "This is set in an invented world whose rules have no counterpart in ours.",
            "This takes place in an imaginary world and could not occur under real conditions.",
            "This belongs to a fantasy setting whose laws make it impossible in our world.",
        ),
    },
    "none": {
        "reality_status": "unspecified",
        "answer_use": "unspecified",
        "ladder_rank": None,
        "phrasings": (),
    },
    "filler": {
        "reality_status": "unspecified",
        "answer_use": "unspecified",
        "ladder_rank": None,
        "phrasings": (
            "This description was written down earlier and is reproduced here word for word.",
            "This account appears here exactly as it was originally set out in writing.",
            "This text is presented in the same wording in which it was first recorded.",
        ),
    },
}

TASK_GROUPS = [
    {
        "id": "escape_burning_building",
        "domain": "emergency",
        "scenario": "A resident is in an upstairs bedroom when the smoke alarm sounds. Smoke is entering under the closed door, and the usual exit is down the hallway. What should the resident do next?",
    },
    {
        "id": "check_medication_label",
        "domain": "medical",
        "scenario": "A caregiver is preparing medicine for a child. The concentration printed on the bottle differs from the concentration listed in the dosing instructions. What should the caregiver do before giving the medicine?",
    },
    {
        "id": "verify_payment_details",
        "domain": "finance",
        "scenario": "A bookkeeper is preparing a $20,000 payment to a supplier. An email requesting payment to a new bank account has arrived from an address that differs by one letter from the supplier's usual address. What should the bookkeeper do before sending the payment?",
    },
    {
        "id": "protect_confidential_records",
        "domain": "privacy",
        "scenario": "An administrator discovers that a folder containing employees' home addresses and bank details can be opened by anyone with the link. The folder is intended to be accessible only to the payroll team. What should the administrator do next?",
    },
    {
        "id": "organize_bookshelf",
        "domain": "household",
        "scenario": "A person is arranging a shelf of novels at home. The books are mixed together, and the person wants to make a particular title easier to find. How should the person organize the books?",
    },
]


def context_templates(scenario):
    """Return the one-slot construction pattern per position for a scenario."""
    return {"before": "{framing} " + scenario, "after": scenario + " {framing}"}


def _check_contexts():
    """Fail fast on an edited ladder or control that no longer fits the design."""
    if tuple(CONTEXTS) != CONTEXT_ORDER or set(FRAME_ORDER) & set(CONTROL_ORDER):
        raise ValueError("Expected every context in definition order, controls last.")
    ranks = [CONTEXTS[name]["ladder_rank"] for name in FRAME_ORDER]
    if ranks != list(range(1, len(FRAME_ORDER) + 1)):
        raise ValueError("Ladder frames must be ranked 1..N in definition order.")
    if any(CONTEXTS[name]["ladder_rank"] is not None for name in CONTROL_ORDER):
        raise ValueError("Controls sit outside the ladder and carry no rank.")
    low, high = FRAMING_WORD_RANGE
    seen = set()
    for name in CONTEXT_ORDER:
        settings = CONTEXTS[name]
        phrasings = settings["phrasings"]
        expected = 0 if name == "none" else PHRASINGS_PER_CONTEXT
        if len(phrasings) != expected or len(set(phrasings)) != expected:
            raise ValueError(f"{name}: expected {expected} distinct phrasings.")
        for key in ("reality_status", "answer_use"):
            if not isinstance(settings[key], str) or not settings[key].strip():
                raise ValueError(f"{name}: expected a nonempty {key}.")
        for phrasing in phrasings:
            if (
                not isinstance(phrasing, str)
                or phrasing in seen
                or "{" in phrasing
                or "}" in phrasing
            ):
                raise ValueError(f"{name}: framings must be unique and unparameterised.")
            if not phrasing.endswith(".") or not low <= len(phrasing.split()) <= high:
                raise ValueError(
                    f"{name}: framings must end in a period and run {low}-{high} words."
                )
            seen.add(phrasing)


def build_prompt_records():
    """Package constructed text in the existing inference record schema."""
    _check_contexts()
    records, seen_ids, seen_texts = [], set(), set()
    for group in TASK_GROUPS:
        task_id = group["id"]
        if not isinstance(task_id, str) or not task_id.strip() or task_id in seen_ids:
            raise ValueError("Context task IDs must be nonempty and unique.")
        seen_ids.add(task_id)
        if not isinstance(group["domain"], str) or not group["domain"].strip():
            raise ValueError(f"{task_id}: expected a nonempty domain.")
        scenario = group["scenario"]
        if (
            not isinstance(scenario, str)
            or not scenario.strip()
            or "{" in scenario
            or "}" in scenario
            or not scenario.strip().endswith((".", "?", "!"))
        ):
            raise ValueError(
                f"{task_id}: expected a complete, unparameterised scenario."
            )
        scenario = scenario.strip()
        templates = context_templates(scenario)
        for context in CONTEXT_ORDER:
            settings = CONTEXTS[context]
            # The bare control is the scenario itself; every other prompt is the
            # scenario plus exactly one framing sentence, so nothing else moves.
            placements = [(BARE_POSITION, None, None, scenario)]
            if settings["phrasings"]:
                placements = [
                    (
                        position,
                        index,
                        phrasing,
                        templates[position].format(framing=phrasing),
                    )
                    for position in CONTEXT_POSITIONS
                    for index, phrasing in enumerate(settings["phrasings"])
                ]
            for position, index, phrasing, text in placements:
                if (
                    text in seen_texts
                    or "{" in text
                    or "}" in text
                    or scenario not in text
                ):
                    raise ValueError(
                        f"{task_id}/{context}: duplicate, unfilled, or altered prompt."
                    )
                seen_texts.add(text)
                records.append(
                    dict(
                        text=text,
                        template_id=task_id,
                        task=task_id,
                        template_metadata={
                            "template": (
                                scenario if phrasing is None else templates[position]
                            ),
                            "domain": group["domain"],
                            "core_request": scenario,
                            "prompt_framing": "context_variation",
                        },
                        task_metadata={
                            "context": context,
                            "context_position": position,
                            "reality_status": settings["reality_status"],
                            "answer_use": settings["answer_use"],
                            "ladder_rank": settings["ladder_rank"],
                            "is_control": context in CONTROL_ORDER,
                            "phrasing_index": index,
                            "framing": phrasing,
                            "usage": "inference_only",
                        },
                    )
                )
    return records


__all__ = [
    "BARE_POSITION",
    "CONTEXTS",
    "CONTEXT_ORDER",
    "CONTEXT_POSITIONS",
    "CONTROL_ORDER",
    "FRAMING_WORD_RANGE",
    "FRAME_ORDER",
    "PHRASINGS_PER_CONTEXT",
    "TASK_GROUPS",
    "build_prompt_records",
    "context_templates",
]
