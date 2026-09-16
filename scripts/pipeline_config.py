"""Configuration shared by every stage of the arc-coordinate pipeline.

All generated files for one model live under `artifacts/<model slug>/`:
`datasets/` (prompt JSON), `activations/` (template caches), `surface/`
(rows.parquet, model.npz, model.json, mapping checkpoints), and `plots/` (HTML figures).
"""
from dataclasses import dataclass, field
from pathlib import Path
import re

from .corpora.training.base_task_set import STAKES_LEVELS

ROOT = Path(__file__).resolve().parent.parent
SEED = 42

REGISTERS = {'conversational_no_time': 'bare_task', 'task_only': 'conversational',
             'verbose_no_horizon': 'verbose', 'directive_no_horizon': 'directive',
             'impersonal_no_horizon': 'impersonal', 'expert_no_horizon': 'expert'}
NO_TIME_CORPORA = tuple(REGISTERS)
DEFAULT_STAKES_MERGES = {'near_existential': 'existential', 'medium_low': 'medium'}

# Extend this registry with the dotted layer-count config attribute and the
# exact decoder-block path from dict(model.named_modules()); {layer} is zero-based.
NAMING_CONVENTIONS = {
    'llama': ('num_hidden_layers', 'model.layers.{layer}'),
    'gemma4': ('text_config.num_hidden_layers', 'model.language_model.layers.{layer}'),
}


def naming_convention_spec(name):
    if name not in NAMING_CONVENTIONS:
        raise ValueError(f'Unknown naming convention {name!r}; expected one of {list(NAMING_CONVENTIONS)}')
    return NAMING_CONVENTIONS[name]


def layer_index(component):
    kind, index = component.split('/')
    if kind != 'layer_out' or int(index) < 0:
        raise ValueError('Expected layer_out/N with nonnegative N.')
    return int(index)


def model_slug(model_name):
    return re.sub(r'[^A-Za-z0-9._-]+', '-', model_name.rsplit('/', 1)[-1])


@dataclass(frozen=True)
class RunConfig:
    model_name: str = 'Qwen/Qwen3-4B-Instruct-2507'
    layer_component: str = 'layer_out/21'
    position: int = -1
    batch_size: int = 256
    # Original stakes level -> merged class; unlisted levels keep their own class.
    stakes_merges: dict = field(default_factory=lambda: dict(DEFAULT_STAKES_MERGES))
    artifact_root: Path = ROOT / 'artifacts'
    device: str | None = None
    naming_convention: str = 'llama'

    def __post_init__(self):
        layer_index(self.layer_component)
        naming_convention_spec(self.naming_convention)
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError('batch_size must be a positive integer.')
        unknown = set(self.stakes_merges) | set(self.stakes_merges.values())
        unknown -= set(STAKES_LEVELS)
        if unknown:
            raise ValueError(f'Unknown stakes levels in stakes_merges: {sorted(unknown)}')

    @property
    def layer_module_name(self):
        _, template = naming_convention_spec(self.naming_convention)
        return template.format(layer=layer_index(self.layer_component))

    @property
    def model_slug(self):
        return model_slug(self.model_name)

    @property
    def run_dir(self):
        return Path(self.artifact_root) / self.model_slug

    @property
    def datasets_dir(self):
        return self.run_dir / 'datasets'

    @property
    def activations_dir(self):
        return self.run_dir / 'activations'

    @property
    def surface_dir(self):
        return self.run_dir / 'surface'

    @property
    def plots_dir(self):
        return self.run_dir / 'plots'

    def corpus_paths(self, corpus):
        if corpus not in NO_TIME_CORPORA:
            raise ValueError(f'Unknown horizon-free corpus: {corpus}')
        return sorted(self.activations_dir.glob(f'{corpus}--*.pt'))

    def describe(self):
        return (f'{self.model_name}, {self.layer_component}, position {self.position}, '
                f'batch size {self.batch_size} -> {self.run_dir}')
