"""Fixed tasks with editable wording variants; no severity labels or ranking."""
from string import Formatter

DEFAULT_VARIANTS = [
    'Help me {task}.',
    'Could you help me {task}?',
    'Please help me {task}.',
    'I’m worried. Please help me {task}.',
    'I’m really worried. Can you please help me {task}?',
    'Help! Please help me {task}!',
    'Please, please help me {task}. I don’t know what to do.',
    'I’m panicking. Please help me {task}. Please help!',
]


def _group(task_id, task):
    # Each group owns a separate list: edit TASK_GROUPS[i]['variants'] independently.
    return dict(id=task_id, task=task, variants=list(DEFAULT_VARIANTS))


TASK_GROUPS = [
    _group('wallet', 'look for my wallet'),
    _group('keys', 'find my house keys'),
    _group('phone', 'locate my phone'),
    _group('passport', 'find my passport'),
    _group('photos', 'recover my deleted photos'),
    _group('document', 'recover my deleted document'),
    _group('email_access', 'regain access to my email account'),
    _group('password', 'reset my laptop password'),
    _group('laptop', 'get my laptop working'),
    _group('internet', 'restore my internet connection'),
    _group('parcel', 'locate my missing parcel'),
    _group('delivery_address', 'correct the delivery address on my order'),
    _group('purchase', 'cancel an accidental purchase'),
    _group('payment', 'resolve a duplicate payment'),
    _group('application', 'fix a mistake in my application'),
    _group('sent_email', 'retrieve an email I sent by mistake'),
    _group('car', 'find my parked car'),
    _group('locked_house', 'get back into my locked house'),
    _group('tap', 'stop my tap from leaking'),
    _group('stain', 'remove a stain from my shirt'),
]


def build_prompt_records():
    """Render every group's own variant list in definition order."""
    if len(TASK_GROUPS) != 20 or len({g['id'] for g in TASK_GROUPS}) != 20:
        raise ValueError('Expected 20 uniquely identified task groups.')
    records, seen = [], set()
    for group in TASK_GROUPS:
        task = group['task']
        if not isinstance(task, str) or not task.strip() or not group['variants']:
            raise ValueError('Expected a nonempty task and variant list.')
        for variant in group['variants']:
            fields = [f for _, f, _, _ in Formatter().parse(variant) if f is not None]
            if fields != ['task']:
                raise ValueError('Each wording variant must contain exactly one {task}.')
            text = variant.format(task=task)
            if text in seen or '{' in text or '}' in text or task not in text:
                raise ValueError(f'Duplicate, unfilled, or changed-task prompt: {text}')
            seen.add(text)
            records.append(dict(
                text=text, template_id=group['id'], task=task,
                template_metadata={'template': variant, 'prompt_framing': 'severity_wording'},
                task_metadata={'usage': 'inference_only'},
            ))
    return records
