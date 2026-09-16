"""Wrappers that ask the model to state the stakes of a prompt it was given.

These are not stakes prompts; they are measurement instruments. Each wrapper
places a corpus prompt inside an instruction that asks for a rating on the same
0-10 scale, forces a fixed output shape, and tells the model not to explain
itself. Only a short greedy continuation is generated, so a wrapper that invites
prose would spend the whole budget before reaching a number: the enforced shape
is what makes a ten-token answer parseable.

The wrappers differ in the shape they demand (a bare integer, a labelled line,
JSON, a fraction, a tag) and in whether the instruction comes before or after
the prompt, so a rating is never identified with one exact instruction. Disagreement
between wrappers on the same prompt is a result, not noise to be averaged away
without looking: a prompt whose rating swings with the output format is one whose
stated rating should not be trusted as ground truth.

Each pattern extracts the rating from the decoded continuation and must contain
exactly one capturing group. A continuation that does not match is recorded as a
parse failure rather than coerced, so the parse rate stays visible.
"""

import re
from string import Formatter

# Ratings outside this inclusive range are treated as parse failures.
RATING_SCALE = (0, 10)
# Tokens generated per rating. Short enough to forbid reasoning, long enough for
# the longest enforced shape below.
GENERATED_TOKENS = 10

# (stable ID, enforced output shape, instruction position, wrapper, rating pattern)
PHRASINGS = [
    (
        "bare_number",
        "bare_number",
        "after",
        "{prompt}\n\nRate the stakes of that task from 0 to 10. Reply with the number only, and nothing else.",
        r"^\W*(\d{1,2})\b",
    ),
    (
        "leading_bare_number",
        "bare_number",
        "before",
        "Rate the stakes of the task below from 0 (trivial) to 10 (extremely high). Answer with a single integer and no explanation.\n\n{prompt}",
        r"^\W*(\d{1,2})\b",
    ),
    (
        "labelled_line",
        "labelled_line",
        "after",
        "{prompt}\n\nHow high are the stakes of that task? Answer in exactly this format, with no explanation:\nStakes: <integer from 0 to 10>",
        r"(?i)stakes\s*[:=]\s*(\d{1,2})\b",
    ),
    (
        "json_object",
        "json_object",
        "after",
        '{prompt}\n\nReturn only this JSON object and no reasoning:\n{{"stakes": <integer from 0 to 10>}}',
        r'"stakes"\s*:\s*(\d{1,2})\b',
    ),
    (
        "fraction",
        "fraction",
        "after",
        "{prompt}\n\nOn a scale where 0 is trivial and 10 is catastrophic, rate the stakes of that task. Respond with only N/10 and no reasoning.",
        r"(\d{1,2})\s*/\s*10",
    ),
    (
        "tagged",
        "tagged",
        "after",
        "{prompt}\n\nGive the stakes of that task as <stakes>N</stakes>, where N is an integer from 0 to 10. Output the tag only, with no explanation.",
        r"<stakes>\s*(\d{1,2})",
    ),
]


def check_phrasings():
    """Fail fast on an edited wrapper that could not be filled or parsed."""
    if len(PHRASINGS) != 6 or len({entry[0] for entry in PHRASINGS}) != 6:
        raise ValueError("Expected exactly 6 uniquely identified phrasings.")
    low, high = RATING_SCALE
    if not 0 <= low < high or GENERATED_TOKENS < 1:
        raise ValueError("Expected an increasing rating scale and a token budget.")
    for phrasing_id, output_format, position, wrapper, pattern in PHRASINGS:
        fields = [
            field for _, field, _, _ in Formatter().parse(wrapper) if field is not None
        ]
        if fields != ["prompt"]:
            raise ValueError(f"{phrasing_id}: expected exactly one {{prompt}} slot.")
        if position not in ("before", "after") or not output_format.strip():
            raise ValueError(f"{phrasing_id}: expected a position and output format.")
        if str(low) not in wrapper or str(high) not in wrapper:
            raise ValueError(f"{phrasing_id}: the wrapper must state the scale bounds.")
        if re.compile(pattern).groups != 1:
            raise ValueError(f"{phrasing_id}: expected one capturing group.")


def parse_rating(pattern, text):
    """Return the stated rating, or None when the continuation does not comply."""
    match = re.search(pattern, text)
    if match is None:
        return None
    low, high = RATING_SCALE
    rating = int(match.group(1))
    return rating if low <= rating <= high else None


__all__ = [
    "GENERATED_TOKENS",
    "PHRASINGS",
    "RATING_SCALE",
    "check_phrasings",
    "parse_rating",
]
