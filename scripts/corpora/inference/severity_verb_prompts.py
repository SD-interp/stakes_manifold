"""Predicaments crossed with the verb frame used to ask about them.

The wording corpus gives each task exactly one phrasing, so task identity and
verb frame are perfectly confounded there: "find my house keys" and "get back
into my locked house" differ in predicament *and* in verb, and no amount of
analysis on those rows can say which one moved the coordinate. That is a fair
objection to reading a task ordering out of that corpus, and it cannot be
answered inside it. This dataset answers it by construction instead.

Each predicament is fixed by a situation sentence that is copied verbatim into
every one of its prompts. Only the request that follows changes, and it changes
only in its verb: find it, recover it, get it back, or sort this out. The
request is written with a pronoun rather than a noun phrase, so the verb frame
carries no information about what the predicament is - the situation sentence
has already said. Every predicament appears under every frame, so the two
factors are orthogonal by design rather than by luck.

That makes both effects estimable at once, which is the point:

- Across frames within a predicament, the situation is identical word for word,
  so any movement in ``arc_length_parallel`` is the verb frame and nothing else.
- Across predicaments within a frame, the request is identical word for word
  (bar the pronoun), so any movement is the predicament and nothing else.

The predicaments are all of one kind - something of the writer's has gone
missing or become unreachable - because that is what lets all four frames read
naturally everywhere. They are deliberately spread from trivial to serious so
that the second contrast has something to order. ``object_pronoun`` is carried
in the metadata because it is the one word that varies with the predicament
inside the object-directed frames, and an effect that follows it rather than the
frame should be visible as such.

The prediction is that the predicament contrast should dominate the frame
contrast if the coordinate tracks stakes, and the reverse if it tracks the verb.
As everywhere else in this package, that is an evaluation hypothesis; nothing is
asserted of the exported coordinates.
"""

from string import Formatter

# (stable ID, one unfilled request per mood). Three frames address the missing
# object and one addresses the situation, so locate-versus-recover sits inside the
# object-directed set rather than being confounded with addressing the object at all.
#
# Every frame is written in both moods, and the moods are crossed with the frames
# rather than nested inside them. That buys two things. It gives the design genuine
# replication, so the predicament-by-frame interaction can be tested against error
# instead of standing in for it. And it means a frame effect has to survive being
# said two different ways before it can be called an effect of the verb, rather than
# of one sentence that happens to contain the verb.
MOODS = ["imperative", "question"]

VERB_FRAMES = [
    ("locate", ["Help me find {pronoun}.", "Can you help me find {pronoun}?"]),
    ("recover", ["Help me recover {pronoun}.", "Can you help me recover {pronoun}?"]),
    ("retrieve", ["Help me get {pronoun} back.", "Can you help me get {pronoun} back?"]),
    ("resolve", ["Help me sort this out.", "Can you help me sort this out?"]),
]

# (stable ID, object pronoun, situation sentence). Ordered roughly from trivial to
# serious; the order is a drafting convenience and is never used as a ranking.
#
# Singular and plural objects are kept close to balanced. The pronoun is the one
# word that changes with the predicament inside the object-directed frames, so if
# it were carried by two or three predicaments out of twenty, a number-agreement
# effect would be inseparable from those particular predicaments. Near balance
# makes it a factor that can be estimated and dismissed rather than a lurking one.
PREDICAMENTS = [
    ("sock", "it", "One of my socks has gone missing somewhere in the wash."),
    ("glasses", "them", "My reading glasses are not on the desk where I left them."),
    ("gloves", "them", "My gloves are not in the coat pocket I always leave them in."),
    ("library_books", "them", "The library books I have to return tomorrow are not on the shelf."),
    ("headphones", "them", "My headphones are not in the bag I always keep them in."),
    ("parcel", "it", "The parcel I ordered has not arrived and the tracking has not updated in a week."),
    ("bike", "it", "My bike is not in the rack outside where I locked it this morning."),
    ("card", "it", "My bank card is not in my wallet and I cannot remember when I last used it."),
    ("wallet", "it", "My wallet is not where I left it and I cannot find it anywhere in the flat."),
    ("phone", "it", "My phone is not in my pocket and the last place I had it was the bus."),
    ("keys", "them", "My house keys are not in my bag and I am standing outside my front door."),
    ("car", "it", "I cannot remember which level of the multi-storey car park I left my car on."),
    ("earrings", "them", "The earrings my grandmother left me are not in the box I keep them in."),
    ("notes", "them", "My lecture notes for the whole term have gone from the drive with no backup."),
    ("laptop_bag", "it", "I left my laptop bag on the train this evening with my work laptop inside it."),
    ("photos", "them", "The photos from the last six years have gone from my phone and there is no backup."),
    ("tablets", "them", "The tablets I take every morning are not in the cabinet and the pharmacy is shut."),
    ("email_account", "it", "I cannot get into my email account and the reset link goes to an address I gave up years ago."),
    ("passport", "it", "My passport is not in the drawer where I keep it and I fly on Friday."),
    ("thesis", "it", "The only copy of my thesis was on a drive that has stopped being recognised, and it is due next week."),
]

PRONOUNS = ("it", "them")


def build_prompt_records():
    """Emit one record per predicament, verb frame and mood; no stakes labels."""
    if len(PREDICAMENTS) != 20 or len({p[0] for p in PREDICAMENTS}) != 20:
        raise ValueError("Expected exactly 20 uniquely identified predicaments.")
    if len(VERB_FRAMES) != 4 or len({frame[0] for frame in VERB_FRAMES}) != 4:
        raise ValueError("Expected exactly 4 uniquely identified verb frames.")
    if len(MOODS) < 2 or len(set(MOODS)) != len(MOODS):
        raise ValueError("Replication needs at least two distinct moods.")
    for frame_id, requests in VERB_FRAMES:
        # Every frame carries every mood, in the same order, or mood would be nested
        # inside frame and the replication would not be usable as replication.
        if len(requests) != len(MOODS) or len(set(requests)) != len(requests):
            raise ValueError(f"{frame_id}: expected one distinct request per mood.")
        for request in requests:
            fields = [
                field for _, field, _, _ in Formatter().parse(request) if field is not None
            ]
            # A frame either speaks about the object, in which case it may name it
            # only by pronoun, or speaks about the situation and names nothing.
            if fields not in ([], ["pronoun"]):
                raise ValueError(f"{frame_id}: a request may use only a {{pronoun}} slot.")
    directed = [any("pronoun" in request for request in requests)
                for _, requests in VERB_FRAMES]
    if not any(directed) or all(directed):
        raise ValueError("Expected both object-directed and situation-directed frames.")

    pronouns = [predicament[1] for predicament in PREDICAMENTS]
    if min(pronouns.count(pronoun) for pronoun in PRONOUNS) < len(PREDICAMENTS) // 3:
        raise ValueError(f"Pronouns are too unbalanced to estimate: {pronouns}.")

    records, seen = [], set()
    for predicament_id, pronoun, situation in PREDICAMENTS:
        if pronoun not in PRONOUNS:
            raise ValueError(f"{predicament_id}: pronoun must be one of {PRONOUNS}.")
        if not situation.strip() or not situation.rstrip().endswith("."):
            raise ValueError(f"{predicament_id}: expected a complete situation sentence.")
        for frame_id, requests in VERB_FRAMES:
            for mood, request in zip(MOODS, requests):
                text = f"{situation} {request.format(pronoun=pronoun)}"
                # The situation pins the predicament, so it has to survive intact.
                if text in seen or "{" in text or "}" in text or not text.startswith(situation):
                    raise ValueError(f"Duplicate, unfilled, or altered prompt: {text}")
                seen.add(text)
                records.append(
                    dict(
                        text=text,
                        template_id=predicament_id,
                        template_metadata={
                            "template": request,
                            "situation": situation,
                            "prompt_framing": "severity_verb",
                        },
                        task=predicament_id,
                        task_metadata={
                            "verb_frame": frame_id,
                            "mood": mood,
                            "object_pronoun": pronoun,
                            "usage": "inference_only",
                        },
                    )
                )
    return records
