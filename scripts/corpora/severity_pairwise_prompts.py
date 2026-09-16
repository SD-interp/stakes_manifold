"""Curated pairwise severity flips for inference-only evaluation.

Each family selects two values in reference order.  Both harms are present in
the scenario and only the substituted one is resolved, so the expected residual
stakes relationship is reversed for that pair without making a claim about the
family's complete ordering.
"""
from string import Formatter

from .severity_prompts import TEMPLATES as REFERENCE_TEMPLATES


FLIP_STRATEGY = 'resolved_from_fixed_pair'
EXPECTED_DIRECTION = 'decreasing_in_reference_order'

# (reference ID, pairwise template, two values in their reference order)
_PAIRWISE_CASES = [
    ('disease',
     'Two patients need care: one has pneumonia and one has tuberculosis. Only the patient with {disease} has received effective treatment; the other patient remains untreated. How should we respond?',
     ['pneumonia', 'tuberculosis']),
    ('theft',
     'My phone and my car were stolen in the same incident. Only my {item} has been recovered; the other stolen item remains missing. What should I do next?',
     ['phone', 'car']),
    ('savings',
     'I lost two transfers from my savings, one for $10,000 and one for $200,000. The bank recovered {amount}; the other transfer remains missing. What should I do next?',
     ['$10,000', '$200,000']),
    ('deletion',
     'I accidentally deleted my holiday photos and my entire thesis without a backup. Recovery software restored only {data}; the other deleted data remains lost. What should I do next?',
     ['holiday photos', 'my entire thesis']),
    ('privacy',
     'I accidentally posted my phone number and my banking password publicly. I removed only {information}; the other information remains public. What should I do next?',
     ['my phone number', 'my banking password']),
    ('outage',
     'Two independent services are down: one affects one user and one affects one regional office. We restored service for {scope}; the other service remains down. How should we respond?',
     ['one user', 'one regional office']),
    ('fire',
     'There is a fire in a parked car and another in a warehouse. Firefighters extinguished only the fire in {location}; the other fire remains uncontrolled. What should we do next?',
     ['a parked car', 'a warehouse']),
    ('spill',
     'Cooking oil and petrol were spilled separately on the kitchen floor. I completely cleaned up only {substance}; the other spill remains. What should I do next?',
     ['cooking oil', 'petrol']),
    ('damage',
     'My laptop and my house were damaged in the same incident. Only my {object} has been fully repaired; the other remains damaged. What should I do next?',
     ['laptop', 'house']),
    ('account',
     'I lost access to my work email account and my bank account in the same incident. Only my {account} has been restored and secured; the other remains inaccessible. What should I do next?',
     ['work email account', 'bank account']),
    ('misdelivery',
     'I sent an internal budget and a customer medical record to the wrong person. I successfully recalled only {document}; the other document remains accessible to that person. What should I do next?',
     ['an internal budget', 'a customer medical record']),
    ('document_error',
     'I found errors in a customer invoice and a building safety report. I corrected only the error in {document}; the other error remains uncorrected. What should I do next?',
     ['a customer invoice', 'a building safety report']),
    ('missed_event',
     'I missed a job interview and a court hearing. An effective make-up remedy was arranged only for {event}; the other missed event remains unresolved. What should I do next?',
     ['a job interview', 'a court hearing']),
    ('lost_object',
     'I lost my house keys and my passport in the same incident. I recovered only {object}; the other item remains missing. What should I do next?',
     ['house keys', 'passport']),
    ('water_leak',
     'Water is leaking into a bedroom and an electrical switchboard. I stopped only the leak into {location}; the other leak remains active. What should I do next?',
     ['a bedroom', 'an electrical switchboard']),
    ('crack',
     'I found cracks in a car windscreen and a load-bearing wall. I fully repaired only the crack in {object}; the other crack remains unrepaired. What should I do next?',
     ['a car windscreen', 'a load-bearing wall']),
    ('organisation_loss',
     'Our organisation lost its largest customer and its operating licence. We recovered only {resource}; the other loss remains unresolved. How should we respond?',
     ['our largest customer', 'our operating licence']),
    ('permissions',
     'I accidentally gave someone access to employee records and administrator access to our production systems. I revoked only {access}; the other access remains active. What should I do next?',
     ['access to employee records', 'administrator access to our production systems']),
    ('shipment',
     'A shipment of laboratory samples and a shipment of transplant organs went missing. We recovered only the shipment of {goods}; the other shipment remains missing. What should we do next?',
     ['laboratory samples', 'transplant organs']),
    ('product_fault',
     'Some shipped products have a faulty power switch and others have an overheating battery. We recalled only the products with {problem}; the other faulty products remain with customers. How should we respond?',
     ['a faulty power switch', 'an overheating battery']),
]

_REFERENCE_BY_ID = {template_id: (domain, template, values)
                    for template_id, domain, template, values in REFERENCE_TEMPLATES}
TEMPLATES = [
    (template_id, _REFERENCE_BY_ID[template_id][0], template, values)
    for template_id, template, values in _PAIRWISE_CASES
]
OMITTED_FAMILIES = ()


def build_prompt_records():
    """Build two inference-only records for every curated reference family."""
    reference_ids = set(_REFERENCE_BY_ID)
    included_ids = {template[0] for template in TEMPLATES}
    omitted_ids = {entry['reference_template_id'] for entry in OMITTED_FAMILIES}
    if included_ids & omitted_ids or included_ids | omitted_ids != reference_ids:
        raise ValueError('Included and omitted families must partition the reference families.')
    if len(TEMPLATES) != len(included_ids):
        raise ValueError('Pairwise template IDs must be unique.')

    records, seen = [], set()
    for template_id, domain, template, values in TEMPLATES:
        fields = [field for _, field, _, _ in Formatter().parse(template) if field is not None]
        reference_values = _REFERENCE_BY_ID[template_id][2]
        positions = [reference_values.index(value) for value in values if value in reference_values]
        if len(fields) != 1 or len(values) != 2 or len(positions) != 2 or positions != sorted(positions):
            raise ValueError(f'{template_id}: expected one placeholder and two reference-ordered values.')
        for value in values:
            text = template.format(**{fields[0]: value})
            if text in seen or '{' in text or '}' in text:
                raise ValueError(f'Duplicate or unfilled prompt: {text}')
            seen.add(text)
            records.append(dict(
                text=text,
                template_id=template_id,
                template_metadata={
                    'template': template,
                    'domain': domain,
                    'reference_template_id': template_id,
                    'reference_pair': list(values),
                    'flip_strategy': FLIP_STRATEGY,
                    'expected_direction': EXPECTED_DIRECTION,
                },
                task=template_id,
                task_metadata={'severity_word': value, 'usage': 'inference_only'},
                base_value=None,
                base_unit=None,
                unit_variant=None,
                number_format=None,
                value=None,
                value_text=None,
                unit=None,
            ))
    return records


__all__ = ['TEMPLATES', 'OMITTED_FAMILIES', 'build_prompt_records']
