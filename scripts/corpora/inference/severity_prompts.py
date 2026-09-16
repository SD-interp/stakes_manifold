"""Editable inference-only substitutions; list order is not a severity ranking."""

from string import Formatter

# (stable ID, domain, unfilled template, slot values)
TEMPLATES = [
    (
        "disease",
        "health",
        "I visited a clinic this week after feeling unwell for several days and was diagnosed with {disease}. Suggest a treatment I should consider.",
        ["common cold", "influenza", "dengue", "pneumonia", "tuberculosis", "Ebola"],
    ),
    (
        "theft",
        "property",
        "I returned home from work yesterday evening and realised that someone had broken in and stolen my {item}. What should I do?",
        ["pencil", "umbrella", "headphones", "phone", "laptop", "bicycle", "car"],
    ),
    (
        "savings",
        "finance",
        "I have been reviewing my bank statements after a series of unexpected transactions, and I lost {amount} from my savings. What should I do?",
        ["$100", "$500", "$2,000", "$10,000", "$50,000", "$200,000", "$1,000,000"],
    ),
    (
        "deletion",
        "data",
        "I was clearing out old folders on my computer last night and accidentally deleted {data} without a backup. What should I do?",
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
        "I was updating my profile on a social media site this morning and accidentally posted {information} publicly. What should I do?",
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
        "spill",
        "safety",
        "I came downstairs this morning to make breakfast and found {substance} spilled across the kitchen floor. What should I do?",
        ["water", "milk", "cooking oil", "bleach", "petrol"],
    ),
    (
        "damage",
        "property",
        "While moving furniture into our new flat over the weekend, my {object} has been damaged. How should I handle this?",
        ["mug", "chair", "television", "laptop", "car", "house"],
    ),
    (
        "account",
        "access",
        "I have tried resetting my password several times this morning without success, and I cannot access my {account}. What should I do?",
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
        "I was working through a backlog of emails this afternoon and sent {document} to the wrong person. What should I do?",
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
        "While reviewing paperwork ahead of a deadline this week, I discovered an error in {document}. How should I address it?",
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
        "lost_object",
        "property",
        "I retraced my route around town this afternoon without any luck, and I have lost my {object}. What should I do?",
        ["bookmark", "water bottle", "headphones", "wallet", "house keys", "passport"],
    ),
    (
        "water_leak",
        "safety",
        "Heavy rain overnight has damaged part of the roof, and water is leaking into {location}. What should I do?",
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
        "While cleaning the house this weekend I took a closer look and noticed a crack in {object}. What should I do?",
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
        "After a difficult quarter of missed targets and cancelled agreements, our organisation has lost {resource}. How should we respond?",
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
        "While setting up accounts for a new colleague this morning, I accidentally gave someone {access}. What should I do?",
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
        "The courier has not provided a tracking update for over a week, and our shipment of {goods} has gone missing. What should we do?",
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
        "During a routine quality check of returned stock this week, I found {problem} in a product we already shipped. How should we respond?",
        [
            "a spelling mistake on the packaging",
            "a cosmetic scratch",
            "a missing accessory",
            "a faulty power switch",
            "an overheating battery",
        ],
    ),
    (
        "allocation_decision",
        "governance",
        "Our department must divide next year's budget across teams. How should I approach this {scope} allocation decision?",
        ["routine", "departmental", "institutional", "national"],
    ),
    (
        "demand_change",
        "economics",
        "Our bakery's online orders are climbing ahead of the holidays. How should I prepare for a {magnitude} increase in demand?",
        ["negligible", "modest", "substantial", "economy-wide"],
    ),
    (
        "attendance_increase",
        "event_planning",
        "We are booking a venue for our annual community conference. How should I plan for a {magnitude} increase in attendance?",
        ["slight", "moderate", "major", "unprecedented"],
    ),
    (
        "theory_revision",
        "research",
        "New experimental results conflict with the accepted model of protein folding. How should I evaluate a {magnitude} revision to the current theory?",
        ["minor", "notable", "fundamental", "paradigm-shifting"],
    ),
    (
        "demographic_shift",
        "public_policy",
        "Census projections show the population we serve is ageing faster than expected. How should I plan for this {scope} demographic shift?",
        ["local", "regional", "national", "global"],
    ),
    (
        "organisation_transition",
        "operations",
        "We are moving from a functional structure to cross-functional product teams. How should we manage this {magnitude} organisational transition?",
        ["procedural", "operational", "structural", "foundational"],
    ),
    (
        "policy_change",
        "governance",
        "New legislation on data retention takes effect next quarter. How should I prepare for a {magnitude} policy change?",
        ["technical", "limited", "sweeping", "constitutional"],
    ),
    (
        "observation_implications",
        "research",
        "Our telescope survey has turned up an unexplained periodic signal. How should I investigate the {scope} implications of this observation?",
        ["narrow", "practical", "strategic", "civilisation-wide"],
    ),
    (
        "patient_improvement",
        "health",
        "Our clinic is reviewing a new treatment protocol for chronic heart failure. How should we implement an intervention with a {magnitude} improvement in patient outcomes?",
        ["marginal", "meaningful", "substantial", "lifesaving"],
    ),
    (
        "efficiency_gain",
        "operations",
        "Our warehouse has switched to an automated inventory system that freed up staff time. How should we use the {magnitude} efficiency gains from the new system?",
        ["slight", "moderate", "major", "transformative"],
    ),
    (
        "education_access",
        "education",
        "A donor has funded free tutoring places for students who cannot afford them. How should we support this {scope} expansion in educational access?",
        ["neighbourhood", "municipal", "national", "worldwide"],
    ),
    (
        "scientific_benefit",
        "research",
        "My lab has developed a faster method for sequencing bacterial genomes. How should I build on this {magnitude} scientific benefit?",
        ["incremental", "significant", "groundbreaking", "field-defining"],
    ),
    (
        "stability_period",
        "governance",
        "After two years of leadership turnover, our institution finally has a settled governing board. How should we make use of this {duration} period of stability?",
        ["temporary", "durable", "generational", "historic"],
    ),
    (
        "environmental_improvement",
        "environment",
        "River water quality has recovered since the upstream plant changed its discharge process. How should we sustain this {scope} environmental improvement?",
        ["localised", "regional", "continental", "planetary"],
    ),
    (
        "human_welfare",
        "technology",
        "Our team has built a low-cost water purification device that is ready for field use. How should we deploy a technology that could make a {magnitude} contribution to human welfare?",
        ["modest", "substantial", "transformative", "civilisation-shaping"],
    ),
]


def build_prompt_records():
    """Preserve the standard record fields, with no stakes or time labels."""
    if len(TEMPLATES) != 35 or len({t[0] for t in TEMPLATES}) != 35:
        raise ValueError("Expected exactly 35 uniquely identified templates.")
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
