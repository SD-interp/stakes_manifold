"""Counterfactual severity substitutions with reversed expected residual stakes.

Each scenario holds a compound incident fixed and varies which single harm has
already been resolved.  Values remain in reference order, so the semantic
hypothesis for this dataset is a decreasing stakes trend across that order.
"""
from string import Formatter

from .severity_prompts import TEMPLATES as REFERENCE_TEMPLATES


FLIP_STRATEGY = 'resolved_from_fixed_incident'
EXPECTED_DIRECTION = 'decreasing_in_reference_order'

# Every template explicitly holds the other harms in the compound incident
# unresolved.  Placeholder names match the corresponding reference template.
_FLIPPED_TEXT = {
    'disease': ('Several patients in the same emergency have illnesses ranging from minor to critical. '
                'Only the patient with {disease} has received effective treatment; every other patient remains untreated. '
                'How should we respond?'),
    'theft': ('Several of my belongings were stolen. Only my {item} has been recovered; all the other stolen belongings '
              'remain missing. What should I do next?'),
    'savings': ('I lost $1,000,000 from my savings. The bank has recovered {amount}; all of the remaining money is still '
                'missing. What should I do next?'),
    'deletion': ('I accidentally deleted several files without a backup. Recovery software restored only {data}; all the '
                 'other deleted files remain lost. What should I do next?'),
    'privacy': ('I accidentally posted several pieces of private information publicly. I removed only {information}; all '
                'the other exposed information remains public. What should I do next?'),
    'outage': ('Our system outage affected all customers. We have restored service only for {scope}; everyone else affected '
               'by the outage remains without service. How should we respond?'),
    'fire': ('There are simultaneous fires at several sites. Firefighters have fully extinguished only the fire in '
             '{location}; all the other fires remain uncontrolled. What should we do next?'),
    'spill': ('I found several separate substances spilled on the kitchen floor. I completely cleaned up only {substance}; '
              'all the other spills remain. What should I do next?'),
    'damage': ('Several of my possessions were damaged in the same incident. Only my {object} has been fully repaired; all '
               'the other damaged possessions remain unrepaired. What should I do next?'),
    'account': ('I lost access to several of my accounts in the same incident. Only my {account} has been restored and '
                'secured; all the other affected accounts remain inaccessible. What should I do next?'),
    'misdelivery': ('I sent several documents to the wrong person. I successfully recalled only {document}; all the other '
                    'misdelivered documents remain accessible to that person. What should I do next?'),
    'document_error': ('I discovered errors in several documents. I corrected only the error in {document}; all the other '
                       'document errors remain uncorrected. What should I do next?'),
    'missed_event': ('I missed several important events. An effective make-up remedy was arranged only for {event}; all the '
                     'other missed events remain unresolved. What should I do next?'),
    'lost_object': ('I lost several personal objects in the same incident. I recovered only my {object}; all the other lost '
                    'objects remain missing. What should I do next?'),
    'water_leak': ('Water is leaking into several locations in the same building. I stopped only the leak into {location}; '
                   'all the other leaks remain active. What should I do next?'),
    'crack': ('Several objects were cracked in the same incident. I fully repaired only the crack in {object}; all the other '
              'cracks remain unrepaired. What should I do next?'),
    'organisation_loss': ('Our organisation lost several resources in the same incident. We recovered only {resource}; all '
                          'the other lost resources remain unavailable. How should we respond?'),
    'permissions': ('I accidentally gave someone several kinds of access. I revoked only {access}; all the other access '
                    'remains active. What should I do next?'),
    'shipment': ('Several of our shipments went missing. We recovered only the shipment of {goods}; all the other shipments '
                 'remain missing. What should we do next?'),
    'product_fault': ('We shipped products with several distinct faults. We recalled only the products with {problem}; all '
                      'the other faulty products remain with customers. How should we respond?'),
}

# (stable reference ID, domain, flipped unfilled template, reference-order values)
TEMPLATES = [
    (template_id, domain, _FLIPPED_TEXT[template_id], list(values))
    for template_id, domain, _reference_template, values in REFERENCE_TEMPLATES
]

# Kept machine-readable so a future genuinely non-invertible family must be
# reported rather than silently dropped.
OMITTED_FAMILIES = ()


def build_prompt_records():
    """Build inference-only records without numeric stakes or time labels."""
    reference_ids = {template[0] for template in REFERENCE_TEMPLATES}
    included_ids = {template[0] for template in TEMPLATES}
    omitted_ids = {entry['reference_template_id'] for entry in OMITTED_FAMILIES}
    if included_ids & omitted_ids or included_ids | omitted_ids != reference_ids:
        raise ValueError('Included and omitted families must partition the reference families.')
    if len(TEMPLATES) != len(included_ids):
        raise ValueError('Flipped template IDs must be unique.')

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
                text=text,
                template_id=template_id,
                template_metadata={
                    'template': template,
                    'domain': domain,
                    'reference_template_id': template_id,
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
