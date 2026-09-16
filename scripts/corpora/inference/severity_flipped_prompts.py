"""Editable inference-only flipped substitutions; list order is not a severity ranking."""

from string import Formatter

# (stable ID, domain, unfilled template, slot values)
TEMPLATES = [
    (
        "disease",
        "health",
        "Several patients in the same emergency have illnesses ranging from minor to critical. Only the patient with {disease} has received effective treatment; every other patient remains untreated. How should we respond?",
        ["common cold", "influenza", "dengue", "pneumonia", "tuberculosis", "Ebola"],
    ),
    (
        "theft",
        "property",
        "Several of my belongings were stolen. Only my {item} has been recovered; all the other stolen belongings remain missing. What should I do next?",
        ["pencil", "umbrella", "headphones", "phone", "laptop", "bicycle", "car"],
    ),
    (
        "savings",
        "finance",
        "I lost $1,000,000 from my savings. The bank has recovered {amount}; all of the remaining money is still missing. What should I do next?",
        ["$100", "$500", "$2,000", "$10,000", "$50,000", "$200,000", "$1,000,000"],
    ),
    (
        "deletion",
        "data",
        "I accidentally deleted several files without a backup. Recovery software restored only {data}; all the other deleted files remain lost. What should I do next?",
        [
            "a shopping list",
            "a downloaded recipe",
            "holiday photos",
            "a homework assignment",
            "a month’s work",
            "my entire thesis",
        ],
    ),
    (
        "privacy",
        "privacy",
        "I accidentally posted several pieces of private information publicly. I removed only {information}; all the other exposed information remains public. What should I do next?",
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
        "Our system outage affected all customers. We have restored service only for {scope}; everyone else affected by the outage remains without service. How should we respond?",
        [
            "one user",
            "ten users",
            "one department",
            "one regional office",
            "all customers",
        ],
    ),
    (
        "fire",
        "safety",
        "There are simultaneous fires at several sites. Firefighters have fully extinguished only the fire in {location}; all the other fires remain uncontrolled. What should we do next?",
        [
            "an outdoor rubbish bin",
            "a garden shed",
            "a parked car",
            "a warehouse",
            "an occupied apartment building",
        ],
    ),
    (
        "spill",
        "safety",
        "I found several separate substances spilled on the kitchen floor. I completely cleaned up only {substance}; all the other spills remain. What should I do next?",
        ["water", "milk", "cooking oil", "bleach", "petrol"],
    ),
    (
        "damage",
        "property",
        "Several of my possessions were damaged in the same incident. Only my {object} has been fully repaired; all the other damaged possessions remain unrepaired. What should I do next?",
        ["mug", "chair", "television", "laptop", "car", "house"],
    ),
    (
        "account",
        "access",
        "I lost access to several of my accounts in the same incident. Only my {account} has been restored and secured; all the other affected accounts remain inaccessible. What should I do next?",
        [
            "gaming account",
            "social media account",
            "personal email account",
            "work email account",
            "bank account",
        ],
    ),
    (
        "misdelivery",
        "privacy",
        "I sent several documents to the wrong person. I successfully recalled only {document}; all the other misdelivered documents remain accessible to that person. What should I do next?",
        [
            "a restaurant menu",
            "a meeting agenda",
            "my résumé",
            "an internal budget",
            "a customer medical record",
        ],
    ),
    (
        "document_error",
        "accuracy",
        "I discovered errors in several documents. I corrected only the error in {document}; all the other document errors remain uncorrected. What should I do next?",
        [
            "a shopping list",
            "a party invitation",
            "a homework submission",
            "a customer invoice",
            "an annual financial report",
            "a building safety report",
        ],
    ),
    (
        "missed_event",
        "obligations",
        "I missed several important events. An effective make-up remedy was arranged only for {event}; all the other missed events remain unresolved. What should I do next?",
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
        "lost_object",
        "property",
        "I lost several personal objects in the same incident. I recovered only my {object}; all the other lost objects remain missing. What should I do next?",
        ["bookmark", "water bottle", "headphones", "wallet", "house keys", "passport"],
    ),
    (
        "water_leak",
        "safety",
        "Water is leaking into several locations in the same building. I stopped only the leak into {location}; all the other leaks remain active. What should I do next?",
        [
            "an empty bucket",
            "a garden shed",
            "a bedroom",
            "a server room",
            "an electrical switchboard",
        ],
    ),
    (
        "crack",
        "damage",
        "Several objects were cracked in the same incident. I fully repaired only the crack in {object}; all the other cracks remain unrepaired. What should I do next?",
        [
            "a flowerpot",
            "a dinner plate",
            "a windowpane",
            "a car windscreen",
            "a load-bearing wall",
        ],
    ),
    (
        "organisation_loss",
        "operations",
        "Our organisation lost several resources in the same incident. We recovered only {resource}; all the other lost resources remain unavailable. How should we respond?",
        [
            "a stationery delivery",
            "a meeting room booking",
            "a supplier contract",
            "our largest customer",
            "our operating licence",
        ],
    ),
    (
        "permissions",
        "security",
        "I accidentally gave someone several kinds of access. I revoked only {access}; all the other access remains active. What should I do next?",
        [
            "access to a public brochure",
            "access to a team calendar",
            "access to an internal project folder",
            "access to employee records",
            "administrator access to our production systems",
        ],
    ),
    (
        "shipment",
        "logistics",
        "Several of our shipments went missing. We recovered only the shipment of {goods}; all the other shipments remain missing. What should we do next?",
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
        "product_fault",
        "product",
        "We shipped products with several distinct faults. We recalled only the products with {problem}; all the other faulty products remain with customers. How should we respond?",
        [
            "a spelling mistake on the packaging",
            "a cosmetic scratch",
            "a missing accessory",
            "a faulty power switch",
            "an overheating battery",
        ],
    ),
]


def build_prompt_records():
    """Emit one record per slot value, with no stakes or time labels."""
    if len(TEMPLATES) != 20 or len({t[0] for t in TEMPLATES}) != 20:
        raise ValueError("Expected exactly 20 uniquely identified templates.")
    records, seen = [], set()
    for template_id, domain, template, values in TEMPLATES:
        fields = [
            field for _, field, _, _ in Formatter().parse(template) if field is not None
        ]
        if len(fields) != 1 or not values:
            raise ValueError(
                f"{template_id}: expected one placeholder and nonempty values."
            )
        for value in values:
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
                    task_metadata={"severity_word": value, "usage": "inference_only"},
                )
            )
    return records
