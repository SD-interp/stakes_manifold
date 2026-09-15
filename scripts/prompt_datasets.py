"""Generate, save, and preflight-check the horizon-free prompt corpora."""
from collections import Counter
import json
import random

from .corpora import DATASETS, generate_task_dataset
from .corpora import horizon_free_task_set
from .pipeline_config import NO_TIME_CORPORA, REGISTERS, SEED

EXPECTED_MIN_TASKS = 350
PROMPT_BUDGET = 100_000


def generate_datasets(config, corpora=NO_TIME_CORPORA):
    """Generate each corpus, write it to the run's datasets folder, return {corpus: records}."""
    corpora = list(dict.fromkeys(corpora))
    if not corpora or not set(corpora) <= set(NO_TIME_CORPORA):
        raise ValueError(f'Choose horizon-free corpora from {list(NO_TIME_CORPORA)}.')
    config.datasets_dir.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for corpus in corpora:
        random.seed(SEED)
        records = generate_task_dataset(dataset=corpus)
        path = config.datasets_dir / f'{corpus}.json'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(records, ensure_ascii=False), encoding='utf-8')
        temporary.replace(path)
        datasets[corpus] = records
    return datasets


def preflight(datasets):
    """Print corpus counts and the stakes x duration crossing; raise on a stale corpus set.

    A stale task set generates without error and silently caches the old 50-task corpora,
    so this fails loudly instead. The crossing matters because a stakes effect is only
    separable from a duration effect if both off-diagonal corners are populated.
    """
    stated = set(horizon_free_task_set.STATED_TASKS)
    if len(horizon_free_task_set.tasks) < EXPECTED_MIN_TASKS:
        raise RuntimeError(
            f'Only {len(horizon_free_task_set.tasks)} horizon-free tasks; '
            f'expected at least {EXPECTED_MIN_TASKS}.')
    if not stated <= set(horizon_free_task_set.tasks):
        raise RuntimeError('The stated task set is not a subset of the horizon-free one.')

    rows, total = [], 0
    for corpus, records in datasets.items():
        total += len(records)
        rows.append((corpus, REGISTERS[corpus], len(DATASETS[corpus].templates),
                     len({r['task'] for r in records}), len(records)))
    print(f"{'corpus':<24}{'register':<16}{'tpl':>5}{'tasks':>7}{'prompts':>10}")
    for corpus, register, templates, tasks, prompts in rows:
        print(f'{corpus:<24}{register:<16}{templates:>5}{tasks:>7}{prompts:>10,}')
    print(f"{'TOTAL':<24}{'':<16}{sum(r[2] for r in rows):>5}{'':>7}{total:>10,}")
    if total > PROMPT_BUDGET:
        raise RuntimeError(f'{total:,} horizon-free prompts exceeds the {PROMPT_BUDGET:,} budget.')

    # Every prompt must be unique: two registers rendering identical text would
    # double-count a task rather than add an independent reading of it.
    seen = Counter(r['text'] for records in datasets.values() for r in records)
    duplicates = [text for text, count in seen.items() if count > 1]
    if duplicates:
        raise RuntimeError(f'{len(duplicates)} duplicate horizon-free prompts, e.g. '
                           f'{duplicates[0][:70]!r}')
    print(f'\n{len(seen):,} unique prompts, no duplicates across registers.')

    expansion = {task: config for task, config in horizon_free_task_set.tasks.items()
                 if task not in stated}
    crossing = Counter((config['stakes'], config['temporal_horizon'])
                       for config in expansion.values())
    # Report coverage across the bands and stakes levels present in the current
    # source definitions; empty combinations are diagnostic, not a preflight failure.
    occupied_bands = {band for _, band in crossing}
    occupied_levels = {level for level, _ in crossing}
    bands = [band for band in horizon_free_task_set.BANDS if band in occupied_bands]
    levels = [level for level in horizon_free_task_set.STAKES_LEVELS if level in occupied_levels]
    print(f'\nstakes x duration band, expansion tasks only ({len(expansion)}):')
    print(f"{'':<18}" + ''.join(f"{b.replace('_', '/'):>16}" for b in bands))
    for level in levels:
        counts = [crossing[(level, band)] for band in bands]
        print(f'{level:<18}' + ''.join(f'{c:>16}' for c in counts))
    empty = sum(1 for level in levels for band in bands if crossing[(level, band)] == 0)
    print(f'empty cells in that {len(levels)}x{len(bands)} grid: {empty} '
          f'(0 means stakes and duration are fully crossed across the claimed cells)')
    unused = [band for band in horizon_free_task_set.BANDS if band not in occupied_bands]
    print(f"bands the expansion does not use: {', '.join(unused) or 'none'}")
