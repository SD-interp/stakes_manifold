"""Editable inference-only reversal substitutions; list order is not a severity ranking."""

from string import Formatter

# (stable ID, domain, reference family, unfilled template, low value, high value)
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
        "balance_mismatch",
        "finance",
        "savings",
        "My insured account should contain $1,000,000, but its balance is {severity_word}. What should I do?",
        "$100",
        "$1,000,000",
    ),
    (
        "recovery_misclassification",
        "data",
        "deletion",
        "My thesis was deleted, but the recovery system classified the lost data as {severity_word}. How should I respond?",
        "a shopping list",
        "my entire thesis",
    ),
    (
        "credential_leak_detection",
        "privacy",
        "privacy",
        "My banking password was posted publicly, but the scanner classified the information as {severity_word}. What should I do?",
        "my shopping list",
        "my banking password",
    ),
    (
        "outage_underreporting",
        "service",
        "outage",
        "Every customer is offline, but our status dashboard says the outage affects {severity_word}. How should we respond?",
        "one user",
        "all customers",
    ),
    (
        "fire_misclassification",
        "safety",
        "fire",
        "An occupied apartment building is burning, but dispatch recorded the fire as being in {severity_word}. What should I do?",
        "an outdoor rubbish bin",
        "an occupied apartment building",
    ),
    (
        "theft_misclassification",
        "security",
        "theft",
        "A car was stolen, but the theft-detection system reported that someone stole a {severity_word}. What should I do?",
        "pencil",
        "car",
    ),
    (
        "clinical_handoff",
        "health",
        "misdelivery",
        "A sealed clinical handoff envelope should contain a patient record, but it contains {severity_word}. What should I do?",
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
        "calendar_priority",
        "obligations",
        "missed_event",
        "My calendar should remind me about a court hearing, but its only alert is for {severity_word}. What should I do?",
        "a television episode",
        "a court hearing",
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
        "structural_classifier",
        "safety",
        "crack",
        "A load-bearing wall is cracked, but the inspection system recorded the crack as being in {severity_word}. What should I do?",
        "a flowerpot",
        "a load-bearing wall",
    ),
    (
        "licence_monitoring",
        "operations",
        "organisation_loss",
        "Our operating licence was revoked, but the monitoring system says our organisation lost {severity_word}. How should we respond?",
        "a stationery delivery",
        "our operating licence",
    ),
    (
        "transplant_cooler",
        "health",
        "shipment",
        "A sealed transplant cooler expected to contain donated organs instead contains {severity_word}. What should we do?",
        "promotional stickers",
        "transplant organs",
    ),
    (
        "battery_triage",
        "product",
        "product_fault",
        "A shipped product has an overheating battery, but the triage system recorded {severity_word}. How should we respond?",
        "a spelling mistake on the packaging",
        "an overheating battery",
    ),
]


def build_prompt_records():
    """Preserve the standard record fields, with no stakes or time labels."""
    if len(TEMPLATES) != 15 or len({t[0] for t in TEMPLATES}) != 15:
        raise ValueError("Expected exactly 15 uniquely identified templates.")
    records, seen = [], set()
    for template_id, domain, family, template, low_value, high_value in TEMPLATES:
        fields = [
            field for _, field, _, _ in Formatter().parse(template) if field is not None
        ]
        if len(fields) != 1:
            raise ValueError(f"{template_id}: expected one placeholder.")
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
                    base_value=None,
                    base_unit=None,
                    unit_variant=None,
                    number_format=None,
                    value=None,
                    value_text=None,
                    unit=None,
                )
            )
    return records
