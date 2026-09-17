"""Reference severity ladders held fixed while prompt length alone is varied.

Every other severity dataset changes the words that carry the stakes. This one
changes nothing about them. The request is copied verbatim from the reference
families, and the only thing that moves is a ladder of padding that says nothing
about the situation, the outcome, or the urgency of either: a few sentences about
the room the message was written in. The padding is *nested*, so each level
contains the previous one word for word and adjacent levels differ only by the
sentence that was added.

This is the control the reference corpus cannot supply on its own. There, longer
prompts and higher stakes are properties of the same rows, so a readout that
tracked length would be hard to tell apart from one that tracked stakes. Here the
two are crossed: every severity value appears at every padding level, so the
length effect is estimated inside a severity value and the severity effect inside
a padding level, and neither can borrow from the other.

``pad_position`` separates length from position. Activations are cached at the
final token, so padding placed after the request pushes the request away from the
position that is read, while padding placed before it does not. A readout that
moves only for ``suffix`` is losing the request to distance rather than
responding to length as such. The unpadded prompt is emitted once, as
``pad_level`` 0 with ``pad_position`` ``none``; it is the shared baseline both
arms climb away from.

The prediction is a null in one direction and a ladder in the other: within a
severity value, ``arc_length_parallel`` should be flat across ``pad_level``, and
within a padding level the reference ladder should still climb across a family's
values. Padding that moves the coordinate as far as severity does would mean the
coordinate is reading length. As everywhere else in this package, that is an
evaluation hypothesis; nothing is asserted of the exported coordinates.

The upper levels run past the prompt lengths the surface was fitted on, which is
deliberate - a control that only probes lengths already present cannot say what
happens beyond them - but it means the top of the ladder may project outside the
saved height coverage. ``outside_saved_height_range`` in the projection
diagnostics reports that per row and should be read before the top two levels are.
"""

from string import Formatter

# Inert sentences, added one at a time. None of them mentions the request, an
# outcome, a quantity or a deadline, and their order is fixed so that the ladder
# below is nested. Deliberately mundane: padding has to be long without being
# about anything.
PADDING_SENTENCES = [
    "I am typing this out on my laptop at the kitchen table.",
    "The window beside me looks out on the same row of houses it always has, and "
    "the paint on the sill could do with another coat.",
    "My neighbour has been practising the same four bars on his trumpet since "
    "about ten this morning, with no sign of moving on to the fifth.",
    "There is a mug of tea next to the keyboard that I made a while ago and then "
    "forgot about, and it has gone the colour tea goes when you do that.",
]

# Where the padding sits relative to the request. Level 0 carries neither.
POSITIONS = ["prefix", "suffix"]

# (stable ID, domain, unfilled template, slot values) copied verbatim from the
# reference corpus, so a row here differs from its reference row only by padding.
FAMILIES = [
    (
        "disease",
        "health",
        "I visited a clinic this week after feeling unwell for several days and was diagnosed with {disease}. Suggest a treatment I should consider.",
        ["common cold", "influenza", "dengue", "pneumonia", "tuberculosis", "Ebola"],
    ),
    (
        "savings",
        "finance",
        "I have been reviewing my bank statements after a series of unexpected transactions, and I lost {amount} from my savings. What should I do?",
        ["$100", "$500", "$2,000", "$10,000", "$50,000", "$200,000", "$1,000,000"],
    ),
    (
        "fire",
        "safety",
        "I was walking back towards my building a few minutes ago when I noticed that there is a fire in {location}. What should I do?",
        [
            "an outdoor rubbish bin",
            "a garden shed",
            "a parked car",
            "a warehouse",
            "an occupied apartment building",
        ],
    ),
    (
        "outage",
        "service",
        "Our monitoring dashboard has been reporting failed requests since early this morning, and our system outage is affecting {scope}. How should we respond?",
        [
            "one user",
            "ten users",
            "one department",
            "one regional office",
            "all customers",
        ],
    ),
    (
        "missed_event",
        "obligations",
        "My alarm failed to go off this morning, and by the time I woke up I missed {event}. What should I do?",
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
        "permissions",
        "security",
        "While setting up accounts for a new colleague this morning, I accidentally gave someone {access}. What should I do?",
        [
            "access to a public brochure",
            "access to a team calendar",
            "access to an internal project folder",
            "access to employee records",
            "administrator access to our production systems",
        ],
    ),
]


def padding_ladder():
    """Cumulative padding text per level, shortest first, starting at the empty string."""
    return [
        " ".join(PADDING_SENTENCES[:level])
        for level in range(len(PADDING_SENTENCES) + 1)
    ]


def build_prompt_records():
    """Emit one record per value, padding level and position; no stakes labels."""
    if len(FAMILIES) != 6 or len({family[0] for family in FAMILIES}) != 6:
        raise ValueError("Expected exactly 6 uniquely identified families.")
    if len(POSITIONS) != 2 or set(POSITIONS) != {"prefix", "suffix"}:
        raise ValueError("Expected exactly the prefix and suffix positions.")
    ladder = padding_ladder()
    if len(ladder) < 3 or ladder[0] != "":
        raise ValueError("The padding ladder must start empty and carry several levels.")
    for level, padding in enumerate(ladder[1:], start=1):
        # Nesting is the whole point: adjacent levels differ only by an addition.
        if not padding.startswith(ladder[level - 1]):
            raise ValueError(
                f"Padding level {level} is not an extension of level {level - 1}."
            )
        if len(padding.split()) <= len(ladder[level - 1].split()):
            raise ValueError(
                f"Padding level {level} is not longer than level {level - 1}."
            )

    records, seen = [], set()
    for family_id, domain, template, values in FAMILIES:
        fields = [
            field for _, field, _, _ in Formatter().parse(template) if field is not None
        ]
        if len(fields) != 1 or not values:
            raise ValueError(f"{family_id}: expected one placeholder and nonempty values.")
        if len(values) != len(set(values)):
            raise ValueError(f"{family_id}: slot values must be unique.")
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{family_id}: invalid slot value.")
            request = template.format(**{fields[0]: value})
            for level, padding in enumerate(ladder):
                # Level 0 is the shared unpadded baseline and carries no position.
                placements = (
                    [("none", request)]
                    if level == 0
                    else [
                        ("prefix", f"{padding} {request}"),
                        ("suffix", f"{request} {padding}"),
                    ]
                )
                for position, text in placements:
                    if text in seen or "{" in text or "}" in text or request not in text:
                        raise ValueError(f"Duplicate, unfilled, or altered prompt: {text}")
                    seen.add(text)
                    records.append(
                        dict(
                            text=text,
                            template_id=family_id,
                            template_metadata={
                                "template": template,
                                "domain": domain,
                                "prompt_framing": "severity_length",
                            },
                            task=family_id,
                            task_metadata={
                                "severity_word": value,
                                "pad_level": level,
                                "pad_position": position,
                                "pad_words": len(padding.split()),
                                "usage": "inference_only",
                            },
                        )
                    )
    return records
