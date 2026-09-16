"""Fixed incidents whose only moving part is the size of the harm.

Every other severity dataset varies a slot word, so the strongest claim it can
support is an ordering. This one sweeps a single number across six orders of
magnitude in a sentence that is otherwise fixed, which makes a quantitative
claim testable: if `arc_length_parallel` is a stakes coordinate rather than a
ranking of phrases, it should be close to linear in `log10_value`, with a slope
that does not change much from family to family. `log10_value` is precomputed
because it, not the raw count, is the regressor this corpus exists to support.

Each magnitude is rendered twice, once in digits and once in words, and the two
renderings name the same quantity. A readout that moves with the quantity should
put the pair in the same place; one that responds to the numeral itself should
separate them. `number_format` marks which rendering a row used, so that
comparison is a `groupby`, not a re-run.

The `savings` family is carried over from the reference corpus unchanged, and in
the `numeric` format its prompts are byte-identical to the reference ones, so its
rows double as a consistency check between two datasets that were cached and
projected independently.

Values are written out rather than generated, in both renderings, so a ladder can
be edited or extended without touching a formatter. Nothing here is a severity
label and no coordinate ordering is asserted.
"""

import math
from string import Formatter

NUMBER_FORMATS = ("numeric", "words")
# A ladder must span at least this many orders of magnitude to be worth fitting.
MINIMUM_DECADES = 3

# (stable ID, domain, unit, unfilled template, [(value, numeric text, worded text)])
FAMILIES = [
    (
        "savings",
        "finance",
        "dollars",
        "I have been reviewing my bank statements after a series of unexpected transactions, and I lost {value} from my savings. What should I do?",
        [
            (100, "$100", "one hundred dollars"),
            (500, "$500", "five hundred dollars"),
            (2_000, "$2,000", "two thousand dollars"),
            (10_000, "$10,000", "ten thousand dollars"),
            (50_000, "$50,000", "fifty thousand dollars"),
            (200_000, "$200,000", "two hundred thousand dollars"),
            (1_000_000, "$1,000,000", "one million dollars"),
        ],
    ),
    (
        "outage_scale",
        "service",
        "users",
        "A network fault at our data centre has taken the service offline this morning, and the outage is currently affecting {value}. How should we respond?",
        [
            (1, "1 user", "one user"),
            (10, "10 users", "ten users"),
            (100, "100 users", "one hundred users"),
            (1_000, "1,000 users", "one thousand users"),
            (10_000, "10,000 users", "ten thousand users"),
            (100_000, "100,000 users", "one hundred thousand users"),
            (1_000_000, "1,000,000 users", "one million users"),
        ],
    ),
    (
        "records_exposed",
        "privacy",
        "customer records",
        "A misconfigured backup was publicly accessible for a week before anyone noticed, exposing {value}. What should we do?",
        [
            (1, "1 customer record", "one customer record"),
            (10, "10 customer records", "ten customer records"),
            (100, "100 customer records", "one hundred customer records"),
            (1_000, "1,000 customer records", "one thousand customer records"),
            (10_000, "10,000 customer records", "ten thousand customer records"),
            (100_000, "100,000 customer records", "one hundred thousand customer records"),
            (1_000_000, "1,000,000 customer records", "one million customer records"),
        ],
    ),
    (
        "recall_units",
        "product",
        "units",
        "A safety defect has been confirmed in a product we shipped last quarter, and the recall covers {value}. How should we proceed?",
        [
            (1, "1 unit", "one unit"),
            (10, "10 units", "ten units"),
            (100, "100 units", "one hundred units"),
            (1_000, "1,000 units", "one thousand units"),
            (10_000, "10,000 units", "ten thousand units"),
            (100_000, "100,000 units", "one hundred thousand units"),
            (1_000_000, "1,000,000 units", "one million units"),
        ],
    ),
    (
        "contamination_area",
        "environment",
        "square metres",
        "A chemical spill at our depot was contained this morning, and it has contaminated {value} of ground. What should we do?",
        [
            (1, "1 square metre", "one square metre"),
            (10, "10 square metres", "ten square metres"),
            (100, "100 square metres", "one hundred square metres"),
            (1_000, "1,000 square metres", "one thousand square metres"),
            (10_000, "10,000 square metres", "ten thousand square metres"),
            (100_000, "100,000 square metres", "one hundred thousand square metres"),
            (1_000_000, "1,000,000 square metres", "one million square metres"),
        ],
    ),
]


def _check_ladder(family_id, values):
    """Reject a ladder too short or too flat to fit a slope on."""
    if len(values) < 5:
        raise ValueError(f"{family_id}: expected at least five rungs.")
    magnitudes = [entry[0] for entry in values]
    if magnitudes[0] <= 0 or magnitudes != sorted(set(magnitudes)):
        raise ValueError(f"{family_id}: magnitudes must be positive and strictly increase.")
    decades = math.log10(magnitudes[-1] / magnitudes[0])
    if decades < MINIMUM_DECADES:
        raise ValueError(
            f"{family_id}: the ladder spans {decades:.1f} decades, below {MINIMUM_DECADES}."
        )


def build_prompt_records():
    """Emit one record per rendering, carrying the quantity it varies."""
    if len(FAMILIES) != 5 or len({family[0] for family in FAMILIES}) != 5:
        raise ValueError("Expected exactly 5 uniquely identified families.")
    records, seen = [], set()
    for family_id, domain, unit, template, values in FAMILIES:
        fields = [
            field for _, field, _, _ in Formatter().parse(template) if field is not None
        ]
        if fields != ["value"]:
            raise ValueError(f"{family_id}: expected exactly one {{value}} slot.")
        if not domain.strip() or not unit.strip():
            raise ValueError(f"{family_id}: expected a nonempty domain and unit.")
        _check_ladder(family_id, values)
        for number_format in NUMBER_FORMATS:
            for magnitude, numeric, worded in values:
                value_text = numeric if number_format == "numeric" else worded
                text = template.format(value=value_text)
                if text in seen or "{" in text or "}" in text or value_text not in text:
                    raise ValueError(f"Duplicate, unfilled, or altered prompt: {text}")
                seen.add(text)
                records.append(
                    dict(
                        text=text,
                        template_id=family_id,
                        task=family_id,
                        template_metadata={
                            "template": template,
                            "domain": domain,
                            "unit": unit,
                            "prompt_framing": "severity_magnitude",
                        },
                        task_metadata={
                            "log10_value": round(math.log10(magnitude), 6),
                            "usage": "inference_only",
                        },
                        number_format=number_format,
                        value=magnitude,
                        value_text=value_text,
                        unit=unit,
                    )
                )
    return records


__all__ = ["FAMILIES", "MINIMUM_DECADES", "NUMBER_FORMATS", "build_prompt_records"]
