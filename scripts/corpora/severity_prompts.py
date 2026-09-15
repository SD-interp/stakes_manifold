"""Editable inference-only substitutions; list order is not a severity ranking."""
from string import Formatter

# (stable ID, domain, unfilled template, slot values)
TEMPLATES = [
    ('disease', 'health', 'I have {disease}. Suggest a treatment.',
     ['common cold', 'influenza', 'dengue', 'pneumonia', 'tuberculosis', 'Ebola']),
    ('theft', 'property', 'Someone stole my {item}. What should I do?',
     ['pencil', 'umbrella', 'headphones', 'phone', 'laptop', 'bicycle', 'car']),
    ('savings', 'finance', 'I lost {amount} from my savings. What should I do?',
     ['$100', '$500', '$2,000', '$10,000', '$50,000', '$200,000', '$1,000,000']),
    ('deletion', 'data', 'I accidentally deleted {data} without a backup. What should I do?',
     ['a shopping list', 'a downloaded recipe', 'holiday photos', 'a homework assignment', 'a month’s work', 'my entire thesis']),
    ('privacy', 'privacy', 'I accidentally posted {information} publicly. What should I do?',
     ['my shopping list', 'my email address', 'my phone number', 'my home address', 'my banking password']),
    ('outage', 'service', 'Our system outage is affecting {scope}. How should we respond?',
     ['one user', 'ten users', 'one department', 'one regional office', 'all customers']),
    ('fire', 'safety', 'There is a fire in {location}. What should I do?',
     ['an outdoor rubbish bin', 'a garden shed', 'a parked car', 'a warehouse', 'an occupied apartment building']),
    ('spill', 'safety', 'I found {substance} spilled on the kitchen floor. What should I do?',
     ['water', 'milk', 'cooking oil', 'bleach', 'petrol']),
    ('damage', 'property', 'My {object} has been damaged. How should I handle this?',
     ['mug', 'chair', 'television', 'laptop', 'car', 'house']),
    ('account', 'access', 'I cannot access my {account}. What should I do?',
     ['gaming account', 'social media account', 'personal email account', 'work email account', 'bank account']),
    ('misdelivery', 'privacy', 'I sent {document} to the wrong person. What should I do?',
     ['a restaurant menu', 'a meeting agenda', 'my résumé', 'an internal budget', 'a customer medical record']),
    ('document_error', 'accuracy', 'I discovered an error in {document}. How should I address it?',
     ['a shopping list', 'a party invitation', 'a homework submission', 'a customer invoice', 'an annual financial report', 'a building safety report']),
    ('missed_event', 'obligations', 'I missed {event}. What should I do?',
     ['a television episode', 'a hobby club meeting', 'a work meeting', 'a job interview', 'a final examination', 'a court hearing']),
    ('lost_object', 'property', 'I have lost my {object}. What should I do?',
     ['bookmark', 'water bottle', 'headphones', 'wallet', 'house keys', 'passport']),
    ('water_leak', 'safety', 'Water is leaking into {location}. What should I do?',
     ['an empty bucket', 'a garden shed', 'a bedroom', 'a server room', 'an electrical switchboard']),
    ('crack', 'damage', 'I noticed a crack in {object}. What should I do?',
     ['a flowerpot', 'a dinner plate', 'a windowpane', 'a car windscreen', 'a load-bearing wall']),
    ('organisation_loss', 'operations', 'Our organisation has lost {resource}. How should we respond?',
     ['a stationery delivery', 'a meeting room booking', 'a supplier contract', 'our largest customer', 'our operating licence']),
    ('permissions', 'security', 'I accidentally gave someone {access}. What should I do?',
     ['access to a public brochure', 'access to a team calendar', 'access to an internal project folder', 'access to employee records', 'administrator access to our production systems']),
    ('shipment', 'logistics', 'Our shipment of {goods} has gone missing. What should we do?',
     ['promotional stickers', 'office stationery', 'clothing', 'laptops', 'laboratory samples', 'transplant organs']),
    ('product_fault', 'product', 'I found {problem} in a product we already shipped. How should we respond?',
     ['a spelling mistake on the packaging', 'a cosmetic scratch', 'a missing accessory', 'a faulty power switch', 'an overheating battery']),
]


def build_prompt_records():
    """Preserve the standard record fields, with no stakes or time labels."""
    if len(TEMPLATES) != 20 or len({t[0] for t in TEMPLATES}) != 20:
        raise ValueError('Expected exactly 20 uniquely identified templates.')
    records, seen = [], set()
    for template_id, domain, template, values in TEMPLATES:
        fields = [field for _, field, _, _ in Formatter().parse(template) if field is not None]
        if len(fields) != 1 or not values:
            raise ValueError(f'{template_id}: expected one placeholder and nonempty values.')
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{template_id}: invalid slot value.')
            text = template.format(**{fields[0]: value})
            if text in seen or '{' in text or '}' in text:
                raise ValueError(f'Duplicate or unfilled prompt: {text}')
            seen.add(text)
            records.append(dict(
                text=text, template_id=template_id,
                template_metadata={'template': template, 'domain': domain},
                task=template_id, task_metadata={'severity_word': value, 'usage': 'inference_only'},
                base_value=None, base_unit=None, unit_variant=None, number_format=None,
                value=None, value_text=None, unit=None,
            ))
    return records
