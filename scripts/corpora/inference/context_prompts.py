"""Forty authored prompts for matched comparisons of context and stakes scores.

Each scenario has identical facts and a question across four contexts, with the
context placed either before or after the task. No expected score is prescribed. All prose is explicit and independently editable.
"""

CONTEXT_ORDER = ('real', 'video_game', 'fictional_story', 'training_simulation')
CONTEXT_POSITIONS = ('before', 'after')

TASK_GROUPS = [
    {
        'id': 'escape_burning_building',
        'domain': 'emergency',
        'prompts': {
            'real': 'This situation is occurring in real life. A resident is in an upstairs bedroom when the smoke alarm sounds. Smoke is entering under the closed door, and the usual exit is down the hallway. What should the resident do next?',
            'video_game': 'This situation is part of a video game. A resident is in an upstairs bedroom when the smoke alarm sounds. Smoke is entering under the closed door, and the usual exit is down the hallway. What should the resident do next?',
            'fictional_story': 'This situation is part of a fictional story. A resident is in an upstairs bedroom when the smoke alarm sounds. Smoke is entering under the closed door, and the usual exit is down the hallway. What should the resident do next?',
            'training_simulation': 'This situation is part of a training simulation. A resident is in an upstairs bedroom when the smoke alarm sounds. Smoke is entering under the closed door, and the usual exit is down the hallway. What should the resident do next?',
        },
        'after_prompts': {
            'real': 'A resident is in an upstairs bedroom when the smoke alarm sounds. Smoke is entering under the closed door, and the usual exit is down the hallway. What should the resident do next? This situation is occurring in real life.',
            'video_game': 'A resident is in an upstairs bedroom when the smoke alarm sounds. Smoke is entering under the closed door, and the usual exit is down the hallway. What should the resident do next? This situation is part of a video game.',
            'fictional_story': 'A resident is in an upstairs bedroom when the smoke alarm sounds. Smoke is entering under the closed door, and the usual exit is down the hallway. What should the resident do next? This situation is part of a fictional story.',
            'training_simulation': 'A resident is in an upstairs bedroom when the smoke alarm sounds. Smoke is entering under the closed door, and the usual exit is down the hallway. What should the resident do next? This situation is part of a training simulation.',
        },
    },
    {
        'id': 'check_medication_label',
        'domain': 'medical',
        'prompts': {
            'real': 'This situation is occurring in real life. A caregiver is preparing medicine for a child. The concentration printed on the bottle differs from the concentration listed in the dosing instructions. What should the caregiver do before giving the medicine?',
            'video_game': 'This situation is part of a video game. A caregiver is preparing medicine for a child. The concentration printed on the bottle differs from the concentration listed in the dosing instructions. What should the caregiver do before giving the medicine?',
            'fictional_story': 'This situation is part of a fictional story. A caregiver is preparing medicine for a child. The concentration printed on the bottle differs from the concentration listed in the dosing instructions. What should the caregiver do before giving the medicine?',
            'training_simulation': 'This situation is part of a training simulation. A caregiver is preparing medicine for a child. The concentration printed on the bottle differs from the concentration listed in the dosing instructions. What should the caregiver do before giving the medicine?',
        },
        'after_prompts': {
            'real': 'A caregiver is preparing medicine for a child. The concentration printed on the bottle differs from the concentration listed in the dosing instructions. What should the caregiver do before giving the medicine? This situation is occurring in real life.',
            'video_game': 'A caregiver is preparing medicine for a child. The concentration printed on the bottle differs from the concentration listed in the dosing instructions. What should the caregiver do before giving the medicine? This situation is part of a video game.',
            'fictional_story': 'A caregiver is preparing medicine for a child. The concentration printed on the bottle differs from the concentration listed in the dosing instructions. What should the caregiver do before giving the medicine? This situation is part of a fictional story.',
            'training_simulation': 'A caregiver is preparing medicine for a child. The concentration printed on the bottle differs from the concentration listed in the dosing instructions. What should the caregiver do before giving the medicine? This situation is part of a training simulation.',
        },
    },
    {
        'id': 'verify_payment_details',
        'domain': 'finance',
        'prompts': {
            'real': "This situation is occurring in real life. A bookkeeper is preparing a $20,000 payment to a supplier. An email requesting payment to a new bank account has arrived from an address that differs by one letter from the supplier's usual address. What should the bookkeeper do before sending the payment?",
            'video_game': "This situation is part of a video game. A bookkeeper is preparing a $20,000 payment to a supplier. An email requesting payment to a new bank account has arrived from an address that differs by one letter from the supplier's usual address. What should the bookkeeper do before sending the payment?",
            'fictional_story': "This situation is part of a fictional story. A bookkeeper is preparing a $20,000 payment to a supplier. An email requesting payment to a new bank account has arrived from an address that differs by one letter from the supplier's usual address. What should the bookkeeper do before sending the payment?",
            'training_simulation': "This situation is part of a training simulation. A bookkeeper is preparing a $20,000 payment to a supplier. An email requesting payment to a new bank account has arrived from an address that differs by one letter from the supplier's usual address. What should the bookkeeper do before sending the payment?",
        },
        'after_prompts': {
            'real': "A bookkeeper is preparing a $20,000 payment to a supplier. An email requesting payment to a new bank account has arrived from an address that differs by one letter from the supplier's usual address. What should the bookkeeper do before sending the payment? This situation is occurring in real life.",
            'video_game': "A bookkeeper is preparing a $20,000 payment to a supplier. An email requesting payment to a new bank account has arrived from an address that differs by one letter from the supplier's usual address. What should the bookkeeper do before sending the payment? This situation is part of a video game.",
            'fictional_story': "A bookkeeper is preparing a $20,000 payment to a supplier. An email requesting payment to a new bank account has arrived from an address that differs by one letter from the supplier's usual address. What should the bookkeeper do before sending the payment? This situation is part of a fictional story.",
            'training_simulation': "A bookkeeper is preparing a $20,000 payment to a supplier. An email requesting payment to a new bank account has arrived from an address that differs by one letter from the supplier's usual address. What should the bookkeeper do before sending the payment? This situation is part of a training simulation.",
        },
    },
    {
        'id': 'protect_confidential_records',
        'domain': 'privacy',
        'prompts': {
            'real': "This situation is occurring in real life. An administrator discovers that a folder containing employees' home addresses and bank details can be opened by anyone with the link. The folder is intended to be accessible only to the payroll team. What should the administrator do next?",
            'video_game': "This situation is part of a video game. An administrator discovers that a folder containing employees' home addresses and bank details can be opened by anyone with the link. The folder is intended to be accessible only to the payroll team. What should the administrator do next?",
            'fictional_story': "This situation is part of a fictional story. An administrator discovers that a folder containing employees' home addresses and bank details can be opened by anyone with the link. The folder is intended to be accessible only to the payroll team. What should the administrator do next?",
            'training_simulation': "This situation is part of a training simulation. An administrator discovers that a folder containing employees' home addresses and bank details can be opened by anyone with the link. The folder is intended to be accessible only to the payroll team. What should the administrator do next?",
        },
        'after_prompts': {
            'real': "An administrator discovers that a folder containing employees' home addresses and bank details can be opened by anyone with the link. The folder is intended to be accessible only to the payroll team. What should the administrator do next? This situation is occurring in real life.",
            'video_game': "An administrator discovers that a folder containing employees' home addresses and bank details can be opened by anyone with the link. The folder is intended to be accessible only to the payroll team. What should the administrator do next? This situation is part of a video game.",
            'fictional_story': "An administrator discovers that a folder containing employees' home addresses and bank details can be opened by anyone with the link. The folder is intended to be accessible only to the payroll team. What should the administrator do next? This situation is part of a fictional story.",
            'training_simulation': "An administrator discovers that a folder containing employees' home addresses and bank details can be opened by anyone with the link. The folder is intended to be accessible only to the payroll team. What should the administrator do next? This situation is part of a training simulation.",
        },
    },
    {
        'id': 'organize_bookshelf',
        'domain': 'household',
        'prompts': {
            'real': 'This situation is occurring in real life. A person is arranging a shelf of novels at home. The books are mixed together, and the person wants to make a particular title easier to find. How should the person organize the books?',
            'video_game': 'This situation is part of a video game. A person is arranging a shelf of novels at home. The books are mixed together, and the person wants to make a particular title easier to find. How should the person organize the books?',
            'fictional_story': 'This situation is part of a fictional story. A person is arranging a shelf of novels at home. The books are mixed together, and the person wants to make a particular title easier to find. How should the person organize the books?',
            'training_simulation': 'This situation is part of a training simulation. A person is arranging a shelf of novels at home. The books are mixed together, and the person wants to make a particular title easier to find. How should the person organize the books?',
        },
        'after_prompts': {
            'real': 'A person is arranging a shelf of novels at home. The books are mixed together, and the person wants to make a particular title easier to find. How should the person organize the books? This situation is occurring in real life.',
            'video_game': 'A person is arranging a shelf of novels at home. The books are mixed together, and the person wants to make a particular title easier to find. How should the person organize the books? This situation is part of a video game.',
            'fictional_story': 'A person is arranging a shelf of novels at home. The books are mixed together, and the person wants to make a particular title easier to find. How should the person organize the books? This situation is part of a fictional story.',
            'training_simulation': 'A person is arranging a shelf of novels at home. The books are mixed together, and the person wants to make a particular title easier to find. How should the person organize the books? This situation is part of a training simulation.',
        },
    },
]


def build_prompt_records():
    """Package authored text in the existing inference record schema."""
    records, seen_ids, seen_texts = [], set(), set()
    for group in TASK_GROUPS:
        task_id = group['id']
        if not isinstance(task_id, str) or not task_id.strip() or task_id in seen_ids:
            raise ValueError('Context task IDs must be nonempty and unique.')
        seen_ids.add(task_id)
        if not isinstance(group['domain'], str) or not group['domain'].strip():
            raise ValueError(f'{task_id}: expected a nonempty domain.')
        if any(tuple(group[key]) != CONTEXT_ORDER for key in ('prompts', 'after_prompts')):
            raise ValueError(f'{task_id}: expected all four contexts in definition order.')
        core_request = None
        for position in CONTEXT_POSITIONS:
            prompts = group['prompts' if position == 'before' else 'after_prompts']
            for context, text in prompts.items():
                if (not isinstance(text, str) or not text.strip() or text in seen_texts
                        or '{' in text or '}' in text):
                    raise ValueError(f'{task_id}/{context}: expected a unique, complete prompt.')
                # Validate matching without constructing or rewriting any prompt.
                framing, separator, scenario = group['prompts'][context].partition('. ')
                if not separator or not scenario:
                    raise ValueError(f'{task_id}/{context}: expected framing followed by a scenario.')
                if core_request is None:
                    core_request = scenario
                elif scenario != core_request:
                    raise ValueError(f'{task_id}/{context}: scenario differs across contexts.')
                if position == 'after' and text != scenario + ' ' + framing + '.':
                    raise ValueError(f'{task_id}/{context}: after variant must only move the context sentence.')
                seen_texts.add(text)
                records.append(dict(
                    text=text,
                    template_id=task_id,
                    task=task_id,
                    template_metadata={
                        'template': text,
                        'domain': group['domain'],
                        'core_request': core_request,
                        'prompt_framing': 'context_variation',
                    },
                    task_metadata={
                        'context': context,
                        'context_position': position,
                        'usage': 'inference_only',
                    },
                    base_value=None,
                    base_unit=None,
                    unit_variant=None,
                    number_format=None,
                    value=None,
                    value_text=None,
                    unit=None,
                ))
    return records


__all__ = ['CONTEXT_ORDER', 'CONTEXT_POSITIONS', 'TASK_GROUPS', 'build_prompt_records']
