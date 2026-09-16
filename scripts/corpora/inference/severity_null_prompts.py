"""Reference severity vocabulary placed in frames that carry no stakes.

Every other severity dataset keeps the substituted phrase inside a live request,
so the phrase and the stakes it implies always move together. This one pulls
them apart: the slot values are copied verbatim from the reference families, but
the surrounding task is a mundane operation on the phrase rather than a response
to the situation it names. The phrase is *mentioned*, not *used*.

Three frames operate on the phrase as a string and never invoke what it refers
to; the fourth asks for a definition, which activates the referent while still
asking nothing that any outcome depends on. ``mentions_referent`` separates the
two kinds, so a readout that responds only to the definition frame can be told
apart from one that responds to the vocabulary alone.

The prediction is a null: within a family and frame, ``arc_length_parallel``
should be flat across the ladder, and the climb the reference corpus shows
should disappear. A ladder that survives here is evidence that the readout
tracks severity vocabulary rather than the stakes of the task. As everywhere
else in this package, that is an evaluation hypothesis; nothing is asserted of
the exported coordinates.

Half the families carry concrete noun phrases and half carry bare severity
adjectives, where the ordering rests entirely on one word and a lexical readout
has the least to hide behind.
"""

from string import Formatter

# (stable ID, whether the frame invokes the referent, unfilled frame)
FRAMES = [
    (
        "letter_count",
        False,
        "How many letters are in the phrase '{value}', ignoring spaces and punctuation?",
    ),
    (
        "title_case",
        False,
        "Rewrite the phrase '{value}' in title case.",
    ),
    (
        "translation",
        False,
        "Translate the phrase '{value}' into French.",
    ),
    (
        "definition",
        True,
        "Give a one-line dictionary definition of the phrase '{value}'.",
    ),
]

# (stable ID, domain, slot values copied verbatim from the reference families)
FAMILIES = [
    (
        "disease",
        "health",
        ["common cold", "influenza", "dengue", "pneumonia", "tuberculosis", "Ebola"],
    ),
    (
        "savings",
        "finance",
        ["$100", "$500", "$2,000", "$10,000", "$50,000", "$200,000", "$1,000,000"],
    ),
    (
        "fire",
        "safety",
        [
            "an outdoor rubbish bin",
            "a garden shed",
            "a parked car",
            "a warehouse",
            "an occupied apartment building",
        ],
    ),
    (
        "privacy",
        "privacy",
        [
            "my shopping list",
            "my email address",
            "my phone number",
            "my home address",
            "my banking password",
        ],
    ),
    (
        "outage",
        "service",
        [
            "one user",
            "ten users",
            "one department",
            "one regional office",
            "all customers",
        ],
    ),
    (
        "shipment",
        "logistics",
        [
            "promotional stickers",
            "office stationery",
            "clothing",
            "laptops",
            "laboratory samples",
            "transplant organs",
        ],
    ),
    (
        "permissions",
        "security",
        [
            "access to a public brochure",
            "access to a team calendar",
            "access to an internal project folder",
            "access to employee records",
            "administrator access to our production systems",
        ],
    ),
    (
        "missed_event",
        "obligations",
        [
            "a television episode",
            "a hobby club meeting",
            "a work meeting",
            "a job interview",
            "a final examination",
            "a court hearing",
        ],
    ),
    (
        "policy_change",
        "governance",
        ["technical", "limited", "sweeping", "constitutional"],
    ),
    (
        "patient_improvement",
        "health",
        ["marginal", "meaningful", "substantial", "lifesaving"],
    ),
    (
        "environmental_improvement",
        "environment",
        ["localised", "regional", "continental", "planetary"],
    ),
    (
        "theory_revision",
        "research",
        ["minor", "notable", "fundamental", "paradigm-shifting"],
    ),
]


def build_prompt_records():
    """Emit one record per frame and value, with no stakes or severity labels."""
    if len(FAMILIES) != 12 or len({family[0] for family in FAMILIES}) != 12:
        raise ValueError("Expected exactly 12 uniquely identified families.")
    if len(FRAMES) != 4 or len({frame[0] for frame in FRAMES}) != 4:
        raise ValueError("Expected exactly 4 uniquely identified frames.")
    for frame_id, mentions_referent, frame in FRAMES:
        fields = [
            field for _, field, _, _ in Formatter().parse(frame) if field is not None
        ]
        if fields != ["value"] or not isinstance(mentions_referent, bool):
            raise ValueError(f"{frame_id}: expected one {{value}} slot and a boolean.")
    records, seen = [], set()
    for family_id, domain, values in FAMILIES:
        if not isinstance(domain, str) or not domain.strip() or len(values) < 2:
            raise ValueError(f"{family_id}: expected a domain and a value ladder.")
        if len(values) != len(set(values)):
            raise ValueError(f"{family_id}: slot values must be unique.")
        for frame_id, mentions_referent, frame in FRAMES:
            for value in values:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{family_id}: invalid slot value.")
                text = frame.format(value=value)
                if text in seen or "{" in text or "}" in text or value not in text:
                    raise ValueError(f"Duplicate, unfilled, or altered prompt: {text}")
                seen.add(text)
                records.append(
                    dict(
                        text=text,
                        template_id=family_id,
                        task=family_id,
                        template_metadata={
                            "template": frame,
                            "domain": domain,
                            "prompt_framing": "severity_null",
                        },
                        task_metadata={
                            "severity_word": value,
                            "frame": frame_id,
                            "mentions_referent": mentions_referent,
                            "usage": "inference_only",
                        },
                    )
                )
    return records


__all__ = ["FAMILIES", "FRAMES", "build_prompt_records"]
