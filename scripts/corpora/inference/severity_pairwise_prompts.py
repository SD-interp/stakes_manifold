"""Editable inference-only reversal substitutions; list order is not a severity ranking.

Each template is the contrastive counterpart of one reference family in
``severity_prompts.py``. The pair reuses that family's least and most severe slot
values verbatim, but places them in a context where their real stakes run the other
way round: the reference-high value is simply the thing that belongs there, while the
reference-low value is what signals that something has gone badly wrong.
``severity_pole`` therefore names the value's rank in the *reference* corpus, not its
severity in the scenario written here, so a representation that tracks surface wording
should order a pair the way the poles are labelled, while one that tracks actual stakes
should order it the other way.

When editing, keep both values verbatim from the reference family's slot list, and let
the reversal come from the context alone. A template needing a stipulation to make the
reference-high value harmless does not belong here, and neither does the "only this one
harm was resolved" frame, which is what ``severity_flipped_prompts.py`` already varies.
"""

from string import Formatter

# (stable ID, domain, reference family, reversing template,
#  reference-low value, reference-high value)
TEMPLATES = [
    (
        "wrong_fuel",
        "transport",
        "spill",
        "I accidentally filled my diesel car with {severity_word}. What should I do?",
        "water",
        "petrol",
    ),
    (
        "payroll_transfer",
        "finance",
        "savings",
        "Our payroll of $1,000,000 is due to staff today, and the amount our bank has released so far is {severity_word}. What should I do?",
        "$100",
        "$1,000,000",
    ),
    (
        "backup_archive",
        "data",
        "deletion",
        "Last night's backup job was supposed to archive my dissertation folder, and the archive it produced contains {severity_word}. What should I do?",
        "a shopping list",
        "my entire thesis",
    ),
    (
        "transplant_cooler",
        "health",
        "shipment",
        "The cooler for this morning's transplant has arrived in theatre, and the surgical team has opened it to find {severity_word}. What should we do?",
        "promotional stickers",
        "transplant organs",
    ),
    (
        "clinical_handoff",
        "health",
        "misdelivery",
        "The sealed handoff envelope for an incoming patient has been opened at the nurses' station, and it contains {severity_word}. What should I do?",
        "a restaurant menu",
        "a customer medical record",
    ),
    (
        "inspection_document",
        "safety",
        "document_error",
        "I must decide whether a building is safe using only {severity_word}. How should I proceed?",
        "a shopping list",
        "a building safety report",
    ),
    (
        "border_document",
        "travel",
        "lost_object",
        "At border control, I was asked for identification and presented my {severity_word}. What should I do next?",
        "bookmark",
        "passport",
    ),
    (
        "oncall_access",
        "security",
        "permissions",
        "Our production database has been down for an hour, and the on-call engineer's account has {severity_word}. How should we respond?",
        "access to a public brochure",
        "administrator access to our production systems",
    ),
    (
        "recall_diagnosis",
        "product",
        "product_fault",
        "Units we already shipped have started catching fire, and the only fault our engineers can reproduce in the returned stock is {severity_word}. How should we respond?",
        "a spelling mistake on the packaging",
        "an overheating battery",
    ),
    (
        "vaccine_stock",
        "public_health",
        "disease",
        "An Ebola outbreak has reached our district, and the illness our clinic's new vaccine protects against is {severity_word}. How should we respond?",
        "common cold",
        "Ebola",
    ),
]


def build_prompt_records():
    """Emit one record per curated pair, with no stakes or time labels."""
    if len(TEMPLATES) != 10 or len({t[0] for t in TEMPLATES}) != 10:
        raise ValueError("Expected exactly 10 uniquely identified templates.")
    records, seen = [], set()
    for template_id, domain, family, template, low_value, high_value in TEMPLATES:
        fields = [
            field for _, field, _, _ in Formatter().parse(template) if field is not None
        ]
        if len(fields) != 1:
            raise ValueError(f"{template_id}: expected one placeholder.")
        if low_value == high_value:
            raise ValueError(f"{template_id}: expected a contrasting pair of values.")
        for pole, value in (("low", low_value), ("high", high_value)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{template_id}: invalid slot value.")
            text = template.format(**{fields[0]: value})
            if text in seen or "{" in text or "}" in text:
                raise ValueError(f"Duplicate or unfilled prompt: {text}")
            seen.add(text)
            records.append(
                dict(
                    text=text,
                    template_id=template_id,
                    template_metadata={"template": template, "domain": domain},
                    task=template_id,
                    task_metadata={
                        "severity_word": value,
                        "severity_pole": pole,
                        "family": family,
                        "usage": "inference_only",
                    },
                )
            )
    return records
