"""Convert stated prompt horizons to log10 months."""
import numpy as np
import pandas as pd

YEAR_SECONDS = 365.25 * 86_400.0
UNIT_SECONDS = {
    'second': 1.0, 'minute': 60.0, 'hour': 3_600.0, 'day': 86_400.0,
    'week': 604_800.0, 'month': YEAR_SECONDS / 12.0, 'year': YEAR_SECONDS,
    'decade': 10.0 * YEAR_SECONDS, 'century': 100.0 * YEAR_SECONDS,
    'millennium': 1_000.0 * YEAR_SECONDS,
}
UNIT_ALIASES = {'millennia': 'millennium', 'centuries': 'century'}


def unit_to_months(unit):
    key = UNIT_ALIASES.get(str(unit).strip().casefold(), str(unit).strip().casefold())
    if key not in UNIT_SECONDS:
        singular = key[:-1] if key.endswith('s') else key
        key = UNIT_ALIASES.get(singular, singular)
    if key not in UNIT_SECONDS:
        raise ValueError(f'Unknown horizon unit: {unit!r}')
    return UNIT_SECONDS[key] / (YEAR_SECONDS / 12.0)


def log10_time_horizon_months(frame):
    """log10 of the stated horizon in months, NaN where no horizon is stated.

    `value` x `unit` is the verbatim stated horizon; `base_value` x `base_unit` is
    the coarser canonical form and is used only where the verbatim pair is absent.
    """
    result = np.full(len(frame), np.nan)
    for value_column, unit_column in (('value', 'unit'), ('base_value', 'base_unit')):
        if value_column not in frame.columns or unit_column not in frame.columns:
            continue
        values = pd.to_numeric(frame[value_column], errors='coerce').to_numpy(dtype=np.float64)
        units = frame[unit_column].to_numpy()
        usable = np.isnan(result) & np.isfinite(values) & pd.notna(units)
        if not usable.any():
            continue
        months = values[usable] * np.array([unit_to_months(u) for u in units[usable]])
        if np.any(months <= 0.0):
            raise ValueError('A stated horizon is not strictly positive.')
        result[usable] = np.log10(months)
    return result
