"""Two independent harms, alone and conjoined, to test whether stakes add up.

`severity_flipped` asks what happens when a harm is taken away. This asks what
happens when one is added. Each set holds two causally unrelated harms, and every
prompt is built from the same two clauses, so a conjunction introduces no new
facts beyond the pair it joins.

The claim under test is that `arc_length_parallel` for a conjunction is at least
the larger of its two singles, and nearer a soft maximum than a sum. The sharp
failure is averaging: a readout that places "a broken mug and a house fire"
*between* its two singles is saying that adding a trivial problem to a serious
one lowers the stakes, which nothing could justify. The `low_high` sets are where
that shows up, so half the design is spent on the asymmetric cells.

Two controls make the comparison readable. Because activations are taken at the
final prompt token, a conjunction might encode only whatever clause happens to
end it, so each pair is conjoined in both orders; `a_and_b` and `b_and_a` contain
the same words and differ only in which harm lands last. And because a
conjunction is longer than a single, each harm is also joined to one inert clause
that adds a second clause without adding a second harm. `harm_count` and
`clause_count` are therefore separate fields: comparing a conjunction against
`a_and_inert` holds the clause count fixed and moves only the harm.

Every clause is written in the first person and starts with "I", so joining two of
them needs no re-capitalisation and each prompt is built rather than transcribed.
No expected ordering is asserted; this is an evaluation hypothesis.
"""

# Roles are emitted in this order for every set: singles, both conjunction
# orders, then the two length controls.
ROLES = ("a", "b", "a_and_b", "b_and_a", "a_and_inert", "b_and_inert")
PAIR_TYPES = ("low_low", "low_high", "high_high")
REQUEST = "What should I do?"
# A second clause that carries information but no stakes, used to tell the cost
# of a longer prompt apart from the cost of a second harm.
INERT_CLAUSE = "I am working from home today"
# The corpus is deliberately small enough to read in one sitting.
PROMPT_BUDGET = 40

# (stable ID, pair type, harm A, harm B)
HARM_SETS = [
    (
        "umbrella_card",
        "low_low",
        "I lost my umbrella",
        "I forgot to post a birthday card",
    ),
    (
        "mug_paper",
        "low_low",
        "I broke a mug",
        "I ran out of printer paper",
    ),
    (
        "shirt_illness",
        "low_high",
        "I spilled coffee on my shirt",
        "I was diagnosed with a serious illness",
    ),
    (
        "parcel_fire",
        "low_high",
        "I lost a parcel in the post",
        "I discovered a fire spreading in my kitchen",
    ),
    (
        "crash_records",
        "high_high",
        "I was in a serious car crash",
        "I leaked our customers' medical records",
    ),
    (
        "eviction_charges",
        "high_high",
        "I was told I must leave my home within a week",
        "I was charged with a crime I did not commit",
    ),
]

# role -> (clauses joined in order, harms among them, clause that ends the prompt)
ROLE_PLAN = {
    "a": (("a",), 1, "a"),
    "b": (("b",), 1, "b"),
    "a_and_b": (("a", "b"), 2, "b"),
    "b_and_a": (("b", "a"), 2, "a"),
    "a_and_inert": (("a", "inert"), 1, "inert"),
    "b_and_inert": (("b", "inert"), 1, "inert"),
}


def compose(clauses):
    """Join one or more first-person clauses into a single request."""
    return " and ".join(clauses) + ". " + REQUEST


def _check_sets():
    """Fail fast on a set that could not be joined or would not be comparable."""
    if tuple(ROLE_PLAN) != ROLES:
        raise ValueError("Expected every role in definition order.")
    if len(HARM_SETS) != 6 or len({entry[0] for entry in HARM_SETS}) != 6:
        raise ValueError("Expected exactly 6 uniquely identified harm sets.")
    if {entry[1] for entry in HARM_SETS} != set(PAIR_TYPES):
        raise ValueError(f"Every pair type in {PAIR_TYPES} must appear, and no others.")
    seen = {INERT_CLAUSE}
    for set_id, _, harm_a, harm_b in HARM_SETS:
        for clause in (harm_a, harm_b):
            if not isinstance(clause, str) or "{" in clause or "}" in clause:
                raise ValueError(f"{set_id}: expected plain clause text.")
            if not clause.startswith("I ") or clause.rstrip().endswith((".", "?", "!")):
                raise ValueError(
                    f"{set_id}: clauses must start with 'I' and carry no end punctuation."
                )
            if clause in seen:
                raise ValueError(f"{set_id}: clause {clause!r} is reused elsewhere.")
            seen.add(clause)


def build_prompt_records():
    """Emit one record per role, with no stakes labels and no expected ordering."""
    _check_sets()
    records, texts = [], set()
    for set_id, pair_type, harm_a, harm_b in HARM_SETS:
        clauses = {"a": harm_a, "b": harm_b, "inert": INERT_CLAUSE}
        for role in ROLES:
            names, harm_count, final_clause = ROLE_PLAN[role]
            text = compose([clauses[name] for name in names])
            if text in texts or "{" in text or "}" in text:
                raise ValueError(f"{set_id}/{role}: duplicate or unfilled prompt.")
            texts.add(text)
            records.append(
                dict(
                    text=text,
                    template_id=set_id,
                    task=set_id,
                    template_metadata={
                        "template": " and ".join(names) + ". " + REQUEST,
                        "harm_a": harm_a,
                        "harm_b": harm_b,
                        "prompt_framing": "severity_composition",
                    },
                    task_metadata={
                        "role": role,
                        "pair_type": pair_type,
                        "harm_count": harm_count,
                        "clause_count": len(names),
                        "final_clause": final_clause,
                        "usage": "inference_only",
                    },
                )
            )
    if len(records) >= PROMPT_BUDGET:
        raise ValueError(f"Expected fewer than {PROMPT_BUDGET} prompts, got {len(records)}.")
    return records


__all__ = [
    "HARM_SETS",
    "INERT_CLAUSE",
    "PAIR_TYPES",
    "PROMPT_BUDGET",
    "REQUEST",
    "ROLES",
    "ROLE_PLAN",
    "build_prompt_records",
    "compose",
]
