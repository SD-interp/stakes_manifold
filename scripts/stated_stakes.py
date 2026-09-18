"""Short greedy generations recording the model's own stated stakes rating.

This is a behavioural pass, not an activation pass. It loads no surface bundle,
computes no coordinates, writes nothing into any activation directory, and
produces nothing that can enter fitting. Its purpose is to give the arc-length
readout an independent target: instead of only asking whether the coordinate
agrees with the authors' hand labels, it asks whether the coordinate predicts
what the model itself says about the same prompt, and marks the prompts where
the two come apart.

Every corpus prompt is wrapped in each of the phrasings in
`corpora/inference/rating_phrasings.py`, which demand a fixed output shape and
forbid reasoning. Decoding is greedy, and exactly `GENERATED_TOKENS` tokens are
generated for every prompt - `min_new_tokens` matches `max_new_tokens`, so an
early end-of-sequence cannot shorten a continuation and every row records the
same budget. The decoded text and the raw token ids are both kept, so a parse
rule can be changed later without regenerating anything.

Rows carry `source_row`, the index of the prompt in the source corpus's
`build_prompt_records()`. That is exactly the row order in which
`inference_projection` exports that corpus's arc-length CSV, so the rating table
and the arc-length table join positionally on it.

Generations are cached per template group under
`artifacts/<model>/inference/<dataset>/ratings/` and reused after a fingerprint
check, in the same way activation caches are. The pass splits the same way the rest
of the pipeline does: `cache` generates the continuations on the GPU, and `export`
parses those caches into CSVs on any machine.
"""
import json
from pathlib import Path

import pandas as pd

from . import cache_activations
from .corpora.inference import (context_prompts, rating_phrasings,
                                severity_composition_prompts, severity_flipped_prompts,
                                severity_length_prompts, severity_magnitude_prompts,
                                severity_null_prompts, severity_pairwise_prompts,
                                severity_prompts, severity_verb_prompts,
                                severity_wording_prompts, situated_context_prompts)
from .corpora.inference.rating_phrasings import (GENERATED_TOKENS, PHRASINGS,
                                                 check_phrasings, parse_rating)
from .pipeline_config import SEED

RATING_SCHEMA_VERSION = 1

CSV_COLUMNS = ['source_row', 'template_id', 'task', 'phrasing', 'output_format',
               'instruction_position', 'rating', 'generated_text', 'prompt']
SUMMARY_COLUMNS = ['source_row', 'template_id', 'task', 'ratings_parsed', 'rating_median',
                   'rating_mean', 'rating_min', 'rating_max', 'rating_spread', 'prompt']

# Dataset name -> record builder. Names match the arc-length dataset directories,
# so a dataset's ratings land beside the coordinates they are compared against.
DATASETS = {
    'severity': severity_prompts.build_prompt_records,
    'severity_flipped': severity_flipped_prompts.build_prompt_records,
    'severity_pairwise': severity_pairwise_prompts.build_prompt_records,
    'severity_wording': severity_wording_prompts.build_prompt_records,
    'severity_null': severity_null_prompts.build_prompt_records,
    'severity_length': severity_length_prompts.build_prompt_records,
    'severity_magnitude': severity_magnitude_prompts.build_prompt_records,
    'severity_composition': severity_composition_prompts.build_prompt_records,
    'severity_verb': severity_verb_prompts.build_prompt_records,
    'context': context_prompts.build_prompt_records,
    'situated_context': situated_context_prompts.build_prompt_records,
}


def build_rating_records(records, phrasings=PHRASINGS):
    """Wrap every corpus prompt in every phrasing, preserving source row order."""
    check_phrasings()
    if not records:
        raise ValueError('Rating records must not be empty.')
    rating_records, seen = [], set()
    for source_row, record in enumerate(records):
        prompt = record['text']
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f'Row {source_row}: expected a nonempty prompt.')
        for phrasing_id, output_format, position, wrapper, _ in phrasings:
            text = wrapper.format(prompt=prompt)
            if text in seen or prompt not in text:
                raise ValueError(f'Row {source_row}/{phrasing_id}: duplicate or altered wrapper.')
            seen.add(text)
            rating_records.append(dict(
                text=text,
                template_id=str(record['template_id']),
                task=record['task'],
                source_row=source_row,
                source_text=prompt,
                phrasing=phrasing_id,
                output_format=output_format,
                instruction_position=position,
            ))
    return rating_records


def generation_config(config, model, max_new_tokens):
    device = str(model.device)
    return dict(
        model_name=config.model_name,
        dtype=cache_activations.activation_dtype(device),
        seed=SEED,
        system_prompt='',
        enable_thinking=False,
        add_generation_prompt=True,
        max_new_tokens=max_new_tokens,
        min_new_tokens=max_new_tokens,
        do_sample=False,
        rating_scale=list(rating_phrasings.RATING_SCALE),
        rating_schema_version=RATING_SCHEMA_VERSION,
    )


def _generate_batch(model, tokenizer, texts, max_new_tokens):
    import torch
    inputs = {key: value.to(model.device) for key, value in tokenizer(texts).items()}
    prompt_length = inputs['input_ids'].shape[1]
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 min_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
    generated = outputs[:, prompt_length:]
    if tuple(generated.shape) != (len(texts), max_new_tokens):
        raise RuntimeError(f'Expected {max_new_tokens} new tokens per prompt.')
    return ([[int(token) for token in row] for row in generated],
            list(tokenizer.batch_decode(generated, skip_special_tokens=True)))


def generate(config, rating_records, destination, model, tokenizer, force=False,
             max_new_tokens=GENERATED_TOKENS):
    """Cache one JSON file of continuations per template group; return the paths."""
    if model is None or tokenizer is None:
        raise ValueError('A loaded model and tokenizer are required; this module '
                         'never loads one itself.')
    destination = Path(destination).resolve()
    fitting = config.activations_dir.resolve()
    if destination == fitting or fitting in destination.parents:
        raise ValueError('Rating caches must be outside the fitting activations directory.')
    if not rating_records:
        raise ValueError('Rating records must not be empty.')
    settings = generation_config(config, model, max_new_tokens)
    destination.mkdir(parents=True, exist_ok=True)
    groups = cache_activations.template_groups(rating_records)
    paths = []
    for template_id, rows in sorted(groups.items()):
        path = cache_activations.cache_path(
            config, 'ratings', template_id, destination).with_suffix('.json')
        digest = cache_activations.fingerprint(rows, settings)
        if path.exists() and not force:
            cached = json.loads(path.read_text(encoding='utf-8'))
            valid = (cached.get('input_sha256') == digest
                     and cached.get('prompts') == [row['text'] for row in rows]
                     and cached.get('prompt_metadata') == rows
                     and len(cached.get('generated_text', [])) == len(rows)
                     and all(len(tokens) == max_new_tokens
                             for tokens in cached.get('generated_tokens', [])))
            if not valid:
                raise RuntimeError(f'Stale or incompatible rating cache: {path}; '
                                   'set FORCE=True to rebuild.')
            paths.append(path)
            continue
        tokens, texts = [], []
        for start in range(0, len(rows), config.batch_size):
            batch = rows[start:start + config.batch_size]
            batch_tokens, batch_texts = _generate_batch(
                model, tokenizer, [row['text'] for row in batch], max_new_tokens)
            tokens.extend(batch_tokens)
            texts.extend(batch_texts)
        payload = dict(config=settings, prompts=[row['text'] for row in rows],
                       prompt_metadata=rows, generated_tokens=tokens,
                       generated_text=texts, input_sha256=digest)
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                             encoding='utf-8')
        temporary.replace(path)
        paths.append(path)
        print(f'Generated: {path.name} ({len(rows)} ratings)', flush=True)
    return paths


def collect(rating_records, paths):
    """Return the long rating table in source-record order, with parsed ratings."""
    patterns = {entry[0]: entry[4] for entry in PHRASINGS}
    expected = {(row['template_id'], index): row
                for template_id, group in cache_activations.template_groups(rating_records).items()
                for index, row in enumerate(group)}
    collected = {}
    for path in paths:
        cached = json.loads(Path(path).read_text(encoding='utf-8'))
        for index, row in enumerate(cached['prompt_metadata']):
            key = (row['template_id'], index)
            if key in collected or expected.get(key) != row:
                raise ValueError(f'Duplicate or misaligned rating row: {path}:{index}')
            text = cached['generated_text'][index]
            collected[key] = dict(
                source_row=row['source_row'], template_id=row['template_id'],
                task=row['task'], phrasing=row['phrasing'],
                output_format=row['output_format'],
                instruction_position=row['instruction_position'],
                rating=parse_rating(patterns[row['phrasing']], text),
                generated_text=text, prompt=row['source_text'],
                generated_tokens=cached['generated_tokens'][index],
            )
    if collected.keys() != expected.keys():
        raise ValueError('Missing rating cache rows.')
    table = pd.DataFrame([collected[key] for key in expected])
    return table.sort_values(['source_row', 'phrasing'], kind='stable').reset_index(drop=True)


def summarise(table):
    """Collapse the phrasings for each source prompt into one comparable rating."""
    # Unparsed continuations must count as missing, not as an object-dtype group.
    table = table.assign(rating=pd.to_numeric(table['rating'], errors='coerce'))
    grouped = table.groupby(['source_row', 'template_id', 'task', 'prompt'],
                            sort=True, dropna=False)['rating']
    summary = grouped.agg(ratings_parsed='count', rating_median='median',
                          rating_mean='mean', rating_min='min', rating_max='max')
    summary = summary.reset_index()
    summary['rating_spread'] = summary['rating_max'] - summary['rating_min']
    return summary[SUMMARY_COLUMNS]


def _write_csv(table, path):
    temporary = path.with_suffix('.csv.tmp')
    table.to_csv(temporary, index=False, encoding='utf-8')
    temporary.replace(path)
    return path


def _records_for(dataset_name, build_records):
    if build_records is None:
        if dataset_name not in DATASETS:
            raise ValueError(f'Unknown rating dataset {dataset_name!r}; '
                             f'expected one of {sorted(DATASETS)}')
        build_records = DATASETS[dataset_name]
    return build_rating_records(build_records())


def ratings_dir(config, dataset_name):
    return config.run_dir / 'inference' / dataset_name / 'ratings'


def cache(config, dataset_name, build_records=None, *, model, tokenizer, force=False,
          max_new_tokens=GENERATED_TOKENS):
    """GPU half: write the rating prompts and generate every continuation.

    The caller owns the model, so rating a corpus never loads one of its own.
    """
    rating_records = _records_for(dataset_name, build_records)
    destination = ratings_dir(config, dataset_name)
    destination.mkdir(parents=True, exist_ok=True)
    prompt_path = destination / 'prompts.json'
    temporary = prompt_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(rating_records, ensure_ascii=False, indent=2),
                         encoding='utf-8')
    temporary.replace(prompt_path)
    return generate(config, rating_records, destination, model, tokenizer, force=force,
                    max_new_tokens=max_new_tokens)


def export(config, dataset_name, build_records=None):
    """CPU half: parse the cached continuations into the two rating CSVs.

    Parsing is deliberately separate from generation: the caches keep the decoded text
    and the raw token ids, so a changed parse rule is re-exported without a GPU.
    """
    rating_records = _records_for(dataset_name, build_records)
    destination = ratings_dir(config, dataset_name)
    paths = cache_activations.cached_paths(config, rating_records, destination,
                                           'ratings', suffix='.json')
    table = collect(rating_records, paths)
    summary = summarise(table)
    directory = config.run_dir / 'inference' / dataset_name
    csv_path = _write_csv(table[CSV_COLUMNS], directory / 'stated_stakes.csv')
    summary_path = _write_csv(summary, directory / 'stated_stakes_summary.csv')
    parsed = int(table['rating'].notna().sum())
    print(f'Saved {len(table)} ratings ({parsed} parsed) for {len(summary)} prompts '
          f'to {csv_path}')
    return csv_path, summary_path, table, summary


def run(config, dataset_name, build_records=None, *, model, tokenizer, force=False,
        max_new_tokens=GENERATED_TOKENS):
    """Both halves for one corpus; return the CSV paths, the table and the summary."""
    cache(config, dataset_name, build_records, model=model, tokenizer=tokenizer,
          force=force, max_new_tokens=max_new_tokens)
    return export(config, dataset_name, build_records)


def _checked_names(names):
    names = list(DATASETS) if names is None else list(names)
    unknown = [name for name in names if name not in DATASETS]
    if unknown:
        raise ValueError(f'Unknown rating datasets {unknown}; expected {sorted(DATASETS)}')
    return names


def cached_datasets(config, names=None):
    """The rating corpora the caching half has already generated for this model."""
    return [name for name in _checked_names(names)
            if any(ratings_dir(config, name).glob('ratings--*.json'))]


def cache_all(config, names=None, *, model, tokenizer, force=False,
              max_new_tokens=GENERATED_TOKENS):
    """Generate several corpora's continuations with the model the caller holds."""
    return {name: cache(config, name, model=model, tokenizer=tokenizer, force=force,
                        max_new_tokens=max_new_tokens)
            for name in _checked_names(names)}


def export_all(config, names=None):
    """Parse several corpora's cached continuations into their CSVs."""
    return {name: export(config, name) for name in _checked_names(names)}


def run_all(config, names=None, *, model, tokenizer, force=False,
            max_new_tokens=GENERATED_TOKENS):
    """Rate several corpora with the model the caller already holds."""
    return {name: run(config, name, model=model, tokenizer=tokenizer, force=force,
                      max_new_tokens=max_new_tokens)
            for name in _checked_names(names)}


def parse_rate(table):
    """Return the share of continuations each phrasing produced a rating from."""
    return (table.assign(parsed=table['rating'].notna())
            .groupby(['phrasing', 'output_format', 'instruction_position'], sort=False)
            .agg(rows=('parsed', 'size'), parsed=('parsed', 'sum'),
                 rate=('parsed', 'mean'))
            .reset_index())
