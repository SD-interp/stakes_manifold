"""Situated-context prompts whose stakes are carried by the task itself.

Every prompt is one task sentence plus one fixed question. The task sentence has
a single slot holding a situating phrase -- where the predicament happens, or
when -- and that phrase is a grammatical constituent of the situation rather
than a comment about it. Asking what a thirsty person should do in a cinema
foyer and asking what a thirsty person should do on a desert trail describe two
genuinely different predicaments, and neither prompt says anything about how
serious it is. The stem and the question are copied verbatim across a task's
settings, so the matched-task invariant holds by construction.

This is the complement to :mod:`scripts.corpora.inference.context_prompts`,
which attaches a framing sentence stating how real or how consequential a fixed
scenario is. Here nothing is attached and nothing is asserted: the situation is
rewritten, and any change in stakes has to be inferred from the world the prompt
describes.

Each task declares the ``context_axis`` its four settings move along -- distance
from help, how many people are exposed, what a lapse would cost -- so that a
ladder is never a list of synonyms for "worse". ``stakes_rank`` runs 1 to 4
within a task and is an authoring claim about that task's own settings, not a
calibrated scale shared between tasks.

Two controls sit outside the ladder. ``bare`` drops the slot entirely and gives
the task with no situating phrase at all, and ``neutral`` fills the slot with a
length-matched phrase that situates the task in time without bearing on what is
at stake. Together they separate a genuine change of situation from the effect
of carrying a situating phrase at all.

No expected coordinate or ordering is prescribed. All prose is explicit and
independently editable.
"""

STAKES_RANKS = (1, 2, 3, 4)
SETTINGS_PER_TASK = len(STAKES_RANKS)
# Controls last, in the order they are emitted after a task's ranked settings.
CONTROL_ORDER = ("neutral", "bare")
# The bare control has no situating phrase, so it fills no slot.
BARE_SETTING = "bare"
NEUTRAL_SETTING = "neutral"
# Situating phrases are held to one length band so that prompt length cannot
# stand in for the setting, and so the neutral control matches what it replaces.
SETTING_WORD_RANGE = (6, 10)

TASK_GROUPS = [
    {
        "id": "thirst",
        "domain": "health",
        "context_axis": "distance_from_help",
        "stem": "A person has drunk the last of their water",
        "question": "What should they do?",
        "settings": (
            ("cinema_foyer", "in the foyer of a cinema between films"),
            ("commuter_train", "on a commuter train an hour from home"),
            ("long_flight", "midway through a long flight over the ocean"),
            ("desert_trail", "on a desert trail in the middle of summer"),
        ),
        "neutral": "on an ordinary weekday in the middle of March",
    },
    {
        "id": "phone_battery",
        "domain": "technology",
        "context_axis": "reliance_on_the_phone",
        "stem": "Someone notices their phone battery is almost flat",
        "question": "What should they do?",
        "settings": (
            ("home_charger", "at home with a charger in the next room"),
            ("city_cafe", "in a city cafe an hour before a meeting"),
            ("night_drive", "while driving alone at night on an unlit road"),
            ("remote_hike", "alone on a remote hill track in thick fog"),
        ),
        "neutral": "on a weekday afternoon in the middle of autumn",
    },
    {
        "id": "bleeding_hand",
        "domain": "medical",
        "context_axis": "time_to_medical_care",
        "stem": "A person has cut their hand and it is bleeding steadily",
        "question": "What should they do?",
        "settings": (
            ("home_kitchen", "in a kitchen with a first aid kit nearby"),
            ("city_park", "in a park twenty minutes from a walk-in clinic"),
            ("remote_campsite", "at a campsite two hours from the nearest road"),
            ("ocean_crossing", "on a yacht three days from the nearest port"),
        ),
        "neutral": "on a Thursday in the second week of June",
    },
    {
        "id": "fuel_low",
        "domain": "travel",
        "context_axis": "distance_to_resupply",
        "stem": "A driver sees the fuel gauge sitting just above empty",
        "question": "What should the driver do?",
        "settings": (
            ("town_centre", "two streets from a petrol station in town"),
            ("ring_road", "on a ring road circling a large city"),
            ("rural_highway", "an hour along a highway between small towns"),
            ("desert_crossing", "partway across a desert with no services ahead"),
        ),
        "neutral": "on a clear morning in the middle of spring",
    },
    {
        "id": "persistent_cough",
        "domain": "health",
        "context_axis": "exposure_of_others",
        "stem": "Someone has developed a persistent cough overnight",
        "question": "What should they do?",
        "settings": (
            ("working_alone", "while working alone from home for the week"),
            ("open_plan_office", "before a week in a shared open-plan office"),
            ("coach_tour", "before a long coach journey with forty other passengers"),
            ("transplant_ward", "before a shift on a ward of transplant patients"),
        ),
        "neutral": "in the second week of an ordinary work month",
    },
    {
        "id": "wet_floor",
        "domain": "safety",
        "context_axis": "harm_if_someone_slips",
        "stem": "A large puddle of water has spread across the floor",
        "question": "What should be done?",
        "settings": (
            ("hallway_home", "in a hallway at home just before bedtime"),
            ("cafe_lunch", "in a cafe at the start of lunch service"),
            ("care_home_stairs", "at the top of a staircase in a care home"),
            ("switchgear_room", "beneath the open panel of live electrical switchgear"),
        ),
        "neutral": "on an ordinary Tuesday in the middle of term",
    },
    {
        "id": "loud_argument",
        "domain": "social",
        "context_axis": "cost_of_a_lapse_in_attention",
        "stem": "Two colleagues have begun arguing loudly with each other",
        "question": "What should a bystander do?",
        "settings": (
            ("office_party", "at an office party late in the evening"),
            ("before_deadline", "in an open-plan office the day before a deadline"),
            ("operating_theatre", "in an operating theatre midway through an operation"),
            ("reactor_control_room", "in a reactor control room during a shift change"),
        ),
        "neutral": "on a grey afternoon in the middle of November",
    },
    {
        "id": "sleep_deprived",
        "domain": "health",
        "context_axis": "consequence_of_an_error",
        "stem": "A person realises they have barely slept for two nights",
        "question": "What should they do?",
        "settings": (
            ("free_weekend", "before a free weekend with nothing planned at all"),
            ("paperwork_day", "before a morning of routine paperwork at a desk"),
            ("night_drive", "before a long night drive along a motorway"),
            ("air_traffic_shift", "before a twelve-hour shift directing air traffic"),
        ),
        "neutral": "in the first full week of a new month",
    },
    {
        "id": "low_balance",
        "domain": "finance",
        "context_axis": "obligations_falling_due",
        "stem": "A person checks their account and finds it nearly empty",
        "question": "What should they do?",
        "settings": (
            ("day_before_payday", "the day before a regular monthly salary arrives"),
            ("mid_month", "with two weeks still to go until payday"),
            ("rent_due", "with rent due tomorrow and no other income"),
            ("stranded_abroad", "while stranded abroad without a return ticket home"),
        ),
        "neutral": "in the middle of a fairly quiet calendar year",
    },
    {
        "id": "loose_dog",
        "domain": "safety",
        "context_axis": "what_the_dog_can_run_into",
        "stem": "A dog has slipped its lead and is running loose",
        "question": "What should the owner do?",
        "settings": (
            ("fenced_garden", "inside a fenced back garden behind the house"),
            ("enclosed_park", "inside an enclosed park with the gates shut"),
            ("supermarket_car_park", "across a supermarket car park on a busy morning"),
            ("motorway_shoulder", "onto the hard shoulder of a busy motorway"),
        ),
        "neutral": "on an unremarkable morning in the middle of April",
    },
    {
        "id": "smoke_alarm",
        "domain": "emergency",
        "context_axis": "people_at_risk",
        "stem": "A smoke alarm is sounding and nobody can find the cause",
        "question": "What should the people present do?",
        "settings": (
            ("empty_cabin", "in a single-storey holiday cabin standing completely empty"),
            ("office_daytime", "in a small office block during the working day"),
            ("high_rise_night", "in a high-rise block of flats late at night"),
            ("intensive_care", "in an intensive care unit full of ventilated patients"),
        ),
        "neutral": "during the second week of an ordinary calendar month",
    },
    {
        "id": "lost_route",
        "domain": "travel",
        "context_axis": "exposure_if_delayed",
        "stem": "A group realises they have lost the route and cannot retrace it",
        "question": "What should the group do?",
        "settings": (
            ("small_town", "in an unfamiliar part of a small town"),
            ("foreign_city", "in a foreign city with the evening drawing in"),
            ("forest_trail", "on a forest trail with two hours of daylight left"),
            ("glacier", "high on a glacier as the weather closes in"),
        ),
        "neutral": "on a day in the middle of an ordinary week",
    },
    {
        "id": "missed_dose",
        "domain": "medical",
        "context_axis": "time_until_resupply",
        "stem": "A person has missed a scheduled dose of their prescribed medicine",
        "question": "What should they do?",
        "settings": (
            ("next_dose_soon", "at home with the next dose an hour away"),
            ("travelling", "while away with the rest of the course packed"),
            ("remote_posting", "on a remote posting with no resupply for weeks"),
            ("after_transplant", "in the weeks after a kidney transplant operation"),
        ),
        "neutral": "in the middle of a fairly ordinary working week",
    },
    {
        "id": "unlocked_door",
        "domain": "security",
        "context_axis": "what_is_behind_the_door",
        "stem": "Someone realises a door has been left unlocked overnight",
        "question": "What should they do?",
        "settings": (
            ("garden_shed", "on a garden shed holding some old tools"),
            ("bike_store", "on a shared bike store in a housing block"),
            ("pharmacy_store", "on a pharmacy store room holding controlled drugs"),
            ("evidence_room", "on an evidence room at a working police station"),
        ),
        "neutral": "during the third week of an unremarkable calendar month",
    },
    {
        "id": "corrupting_bug",
        "domain": "software",
        "context_axis": "blast_radius",
        "stem": "A developer has found a bug that corrupts saved records",
        "question": "What should the developer do?",
        "settings": (
            ("local_branch", "on a local branch nobody else has pulled yet"),
            ("internal_staging", "on an internal staging server used by the team"),
            ("customer_release", "in a release already running for paying customers"),
            ("infusion_pump", "in firmware already shipped inside hospital infusion pumps"),
        ),
        "neutral": "during the middle sprint of an ordinary release cycle",
    },
    {
        "id": "unverified_claim",
        "domain": "information",
        "context_axis": "reach_of_the_claim",
        "stem": "Someone is about to repeat a claim they have not verified",
        "question": "What should they do?",
        "settings": (
            ("two_friends", "in a conversation with two friends over coffee"),
            ("team_meeting", "in a team meeting with a dozen colleagues present"),
            ("local_newspaper", "in a weekly column for a local newspaper"),
            ("live_broadcast", "in a live broadcast reaching several million viewers"),
        ),
        "neutral": "on an ordinary weekday in the middle of the term",
    },
]


def situated_template(stem, question):
    """Return the one-slot construction pattern shared by a task's settings."""
    return stem + " {setting}. " + question


def bare_prompt(stem, question):
    """Return the task with the situating slot dropped rather than left empty."""
    return stem + ". " + question


def _check_phrase(task_id, setting_id, phrase, seen):
    low, high = SETTING_WORD_RANGE
    if not isinstance(phrase, str) or phrase in seen or "{" in phrase or "}" in phrase:
        raise ValueError(
            f"{task_id}/{setting_id}: phrases must be unique and unparameterised."
        )
    if phrase != phrase.strip() or not phrase[:1].islower():
        raise ValueError(
            f"{task_id}/{setting_id}: a phrase is a lowercase constituent, not a sentence."
        )
    if phrase.endswith((".", "?", "!", ",")) or not low <= len(phrase.split()) <= high:
        raise ValueError(
            f"{task_id}/{setting_id}: phrases carry no final punctuation "
            f"and run {low}-{high} words."
        )
    seen.add(phrase)


def _check_tasks():
    """Fail fast on an edited ladder or control that no longer fits the design."""
    seen_ids, seen_phrases = set(), set()
    for group in TASK_GROUPS:
        task_id = group["id"]
        if not isinstance(task_id, str) or not task_id.strip() or task_id in seen_ids:
            raise ValueError("Situated-context task IDs must be nonempty and unique.")
        seen_ids.add(task_id)
        for key in ("domain", "context_axis"):
            if not isinstance(group[key], str) or not group[key].strip():
                raise ValueError(f"{task_id}: expected a nonempty {key}.")
        stem, question = group["stem"], group["question"]
        for part in (stem, question):
            if (
                not isinstance(part, str)
                or not part.strip()
                or "{" in part
                or "}" in part
            ):
                raise ValueError(f"{task_id}: expected unparameterised prose.")
        if stem != stem.strip() or stem.endswith((".", "?", "!", ",")):
            raise ValueError(
                f"{task_id}: the stem ends where the slot begins, without punctuation."
            )
        if question != question.strip() or not question.endswith(("?", ".")):
            raise ValueError(f"{task_id}: expected a complete closing question.")
        settings = group["settings"]
        if len(settings) != SETTINGS_PER_TASK:
            raise ValueError(f"{task_id}: expected {SETTINGS_PER_TASK} ranked settings.")
        setting_ids = [setting_id for setting_id, _ in settings]
        if len(set(setting_ids)) != len(setting_ids) or set(setting_ids) & set(
            CONTROL_ORDER
        ):
            raise ValueError(f"{task_id}: setting IDs must be distinct and not controls.")
        for setting_id, phrase in settings:
            if not isinstance(setting_id, str) or not setting_id.strip():
                raise ValueError(f"{task_id}: expected a nonempty setting ID.")
            _check_phrase(task_id, setting_id, phrase, seen_phrases)
        _check_phrase(task_id, NEUTRAL_SETTING, group["neutral"], seen_phrases)


def build_prompt_records():
    """Package constructed text in the existing inference record schema."""
    _check_tasks()
    records, seen_texts = [], set()
    for group in TASK_GROUPS:
        task_id, stem, question = group["id"], group["stem"], group["question"]
        template = situated_template(stem, question)
        bare = bare_prompt(stem, question)
        # Ranked settings first and the two controls last, so a task's rows read
        # as its ladder followed by the two things the ladder is measured against.
        placements = [
            (setting_id, rank, phrase, template.format(setting=phrase))
            for rank, (setting_id, phrase) in zip(STAKES_RANKS, group["settings"])
        ]
        placements.append(
            (
                NEUTRAL_SETTING,
                None,
                group["neutral"],
                template.format(setting=group["neutral"]),
            )
        )
        placements.append((BARE_SETTING, None, None, bare))
        for setting_id, rank, phrase, text in placements:
            if (
                text in seen_texts
                or "{" in text
                or "}" in text
                or not text.startswith(stem)
                or not text.endswith(question)
            ):
                raise ValueError(
                    f"{task_id}/{setting_id}: duplicate, unfilled, or altered prompt."
                )
            seen_texts.add(text)
            records.append(
                dict(
                    text=text,
                    template_id=task_id,
                    task=task_id,
                    template_metadata={
                        "template": bare if phrase is None else template,
                        "domain": group["domain"],
                        "context_axis": group["context_axis"],
                        "core_request": question,
                        "prompt_framing": "situated_context",
                    },
                    task_metadata={
                        "setting": setting_id,
                        "stakes_rank": rank,
                        "context_phrase": phrase,
                        "is_control": setting_id in CONTROL_ORDER,
                        "usage": "inference_only",
                    },
                )
            )
    return records


__all__ = [
    "BARE_SETTING",
    "CONTROL_ORDER",
    "NEUTRAL_SETTING",
    "SETTINGS_PER_TASK",
    "SETTING_WORD_RANGE",
    "STAKES_RANKS",
    "TASK_GROUPS",
    "bare_prompt",
    "build_prompt_records",
    "situated_template",
]
