"""Configuration shared by every stage of the arc-coordinate pipeline.

All generated files for one model live under `artifacts/<model slug>/`:
`datasets/` (prompt JSON), `activations/` (template caches), `bcpc/` (the full-fit
BCPC and arc-length spline: rows.parquet, model.npz, model.json, and `plot_data/` from the
out-of-fold check), `surface/` (rows.parquet, model.npz, model.json, mapping checkpoints),
and `plots/` (HTML figures).
"""
from dataclasses import dataclass, field
from pathlib import Path
import json
import re

from .corpora.training.base_task_set import STAKES_LEVELS

ROOT = Path(__file__).resolve().parent.parent
SEED = 42

REGISTERS = {'conversational_no_time': 'bare_task', 'task_only': 'conversational',
             'verbose_no_horizon': 'verbose', 'directive_no_horizon': 'directive',
             'impersonal_no_horizon': 'impersonal', 'expert_no_horizon': 'expert'}
NO_TIME_CORPORA = tuple(REGISTERS)
DEFAULT_STAKES_MERGES = {'near_existential': 'existential', 'medium_low': 'medium'}

# The caching half writes this beside the artifacts it produces and the analysis half
# reads it back, so the two halves agree on the settings the caches were made with
# without the analysis host contacting Hugging Face or retyping a configuration.
RUN_CONFIG_FILE = 'run_config.json'
# Device, artifact root and weights cache describe the machine, not the caches, so
# they are deliberately absent: each half sets its own.
SHARED_SETTINGS = ('model_name', 'naming_convention', 'layer_component', 'position',
                   'batch_size', 'stakes_merges')
# The subset that determines what the caches contain. Batch size only changes how rows
# are grouped for the forward pass, and stakes merges are applied when caches are read,
# so neither enters a cache fingerprint and changing them never invalidates a cache.
CACHE_SETTINGS = ('model_name', 'naming_convention', 'layer_component', 'position')

# Extend this registry with the dotted layer-count config attribute and the
# exact decoder-block path from dict(model.named_modules()); {layer} is zero-based.
NAMING_CONVENTIONS = {
    'llama': ('num_hidden_layers', 'model.layers.{layer}'),
    'gemma4': ('text_config.num_hidden_layers', 'model.language_model.layers.{layer}'),
    'mistral3': ('text_config.num_hidden_layers', 'model.language_model.layers.{layer}'),
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
    # Where Hugging Face stores downloaded weights. None keeps the Hugging Face
    # default (HF_HOME/hub, or HF_HUB_CACHE when set). A path is expanded and made
    # absolute so every download, load and cache scan agrees on one directory.
    hf_cache_dir: str | Path | None = None

    def __post_init__(self):
        layer_index(self.layer_component)
        naming_convention_spec(self.naming_convention)
        if self.hf_cache_dir is not None:
            if not isinstance(self.hf_cache_dir, (str, Path)) or not str(self.hf_cache_dir).strip():
                raise ValueError('hf_cache_dir must be None or a nonempty path.')
            object.__setattr__(self, 'hf_cache_dir',
                               Path(self.hf_cache_dir).expanduser().resolve())
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
    def bcpc_dir(self):
        return self.run_dir / 'bcpc'

    @property
    def plots_dir(self):
        return self.run_dir / 'plots'

    def corpus_paths(self, corpus):
        if corpus not in NO_TIME_CORPORA:
            raise ValueError(f'Unknown horizon-free corpus: {corpus}')
        return sorted(self.activations_dir.glob(f'{corpus}--*.pt'))

    def shared_settings(self):
        """The settings both pipeline halves must agree on."""
        return {name: getattr(self, name) for name in SHARED_SETTINGS}

    def describe(self):
        cache = self.hf_cache_dir or 'Hugging Face default'
        return (f'{self.model_name}, {self.layer_component}, position {self.position}, '
                f'batch size {self.batch_size} -> {self.run_dir} '
                f'(weights cache: {cache})')


def save_run_config(config, force=False):
    """Record, beside the artifacts, the settings the analysis half must reuse.

    A manifest that disagrees with `config` on a cache-defining setting means the
    directory already holds caches made with other settings, which the analysis half
    would silently mix. That raises unless `force`, the same flag that rebuilds those
    caches. The other shared settings are simply updated to the current values.
    """
    settings = config.shared_settings()
    path = config.run_dir / RUN_CONFIG_FILE
    if path.is_file() and not force:
        recorded = json.loads(path.read_text(encoding='utf-8'))
        differing = sorted(name for name in CACHE_SETTINGS
                           if recorded.get(name) != settings.get(name))
        if differing:
            raise ValueError(f'{path} records different settings for {differing}; '
                             'use another artifact_root, or rebuild with FORCE.')
    config.run_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(settings, indent=2, sort_keys=True), encoding='utf-8')
    temporary.replace(path)
    return path


def load_run_config(run_dir, **overrides):
    """Rebuild the RunConfig whose artifacts are in `run_dir`."""
    run_dir = Path(run_dir).resolve()
    path = run_dir / RUN_CONFIG_FILE
    if not path.is_file():
        raise FileNotFoundError(f'No {RUN_CONFIG_FILE} in {run_dir}; run the caching '
                                'notebook for this model first.')
    recorded = json.loads(path.read_text(encoding='utf-8'))
    missing = [name for name in SHARED_SETTINGS if name not in recorded]
    if missing:
        raise ValueError(f'{path} is missing {missing}.')
    config = RunConfig(artifact_root=run_dir.parent,
                       **{name: recorded[name] for name in SHARED_SETTINGS}, **overrides)
    if config.run_dir.resolve() != run_dir:
        raise ValueError(f'{path} names {config.model_name}, whose artifacts belong in '
                         f'{config.run_dir}, not {run_dir}.')
    return config


def discover_run_dirs(artifact_root):
    """Every model directory under `artifact_root` the caching half has written."""
    root = Path(artifact_root)
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if (path / RUN_CONFIG_FILE).is_file())
