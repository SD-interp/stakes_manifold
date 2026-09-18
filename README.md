# Arc-coordinate pipeline

Compute `arc_length_parallel` and `arc_length_orthogonal` for horizon-free task prompts.
The orthogonal column is the existing name for PLS3 height relative to zero; parallel
length is signed slice length.

## Run

The pipeline runs in two passes, so the expensive half is paid for once.

```powershell
python -m pip install -r requirements.txt
python -m jupyterlab notebooks/arc_length_cache.ipynb    # 1. GPU
python -m jupyterlab notebooks/arc_length_surface.ipynb  # 2. CPU
```

**1. `notebooks/arc_length_cache.ipynb` needs a GPU.** For every model in its `MODELS` list it
generates the prompt corpora, caches the layer outputs the surface is fitted from, caches the
activations of every inference corpus, and generates the stated-stakes rating continuations.
Each model is loaded exactly once and that instance serves every stage; no module in `scripts/`
loads weights on its own. Its config cell selects the models, batch size, stakes class-merging
dict, corpora, weights cache directory, rating corpora, and `FORCE`; the layer is
`floor(0.6 * num_hidden_layers)` per model. It fits nothing and computes no coordinates.

**2. `notebooks/arc_length_surface.ipynb` needs no GPU, no weights and no network.** For every
model it finds under `ARTIFACT_ROOT` it fits the BCPC target, PLS, centroid-plane rotation and
height-slice surface from the cached activations, exports and plots the surface, projects every
cached inference corpus through it, and parses the cached rating continuations into CSVs. Each
model describes itself through the `run_config.json` the caching pass wrote, so the layer,
position, batch size and stakes merges are never retyped.

Everything runs and is stored locally; nothing is uploaded. The two passes can run on one
machine or on two: to split them, copy `artifacts/<model name>/` off the GPU host. Changing a
merge rule, a slice setting or a rating parse rule means re-running the second pass only.

## Artifacts

Everything generated for a model lives under `artifacts/<model name>/`, e.g.
`artifacts/Qwen3-4B-Instruct-2507/`:

- `run_config.json` - the settings the caching pass used, which the analysis pass reads back
- `datasets/` - prompt JSON per corpus
- `activations/` - one `.pt` cache per corpus/template, preserving equal-file weights
- `inference/<dataset>/` - `prompts.json`, `activations/`, `ratings/`, and the exported CSVs
- `surface/` - `rows.parquet`, `model.npz`, `model.json`, and `slice_mapping_checkpoints/`
- `plots/` - self-contained HTML figures: `bcpc_centroid_spline.html`,
  `pls_centroid_splines.html`, `pls_surface.html`

Completed caches are fingerprint-checked before reuse; set `FORCE = True` to rebuild stale
ones (for example after changing the layer). Compatible stated caches placed in
`activations/` are projected with frozen transforms but never enter any fit.

## Code layout

All logic is in `scripts/`; the notebooks only orchestrate.

| module | role |
|---|---|
| `pipeline_config.py` | `RunConfig`: model, layer, batch size, stakes merges, artifact paths |
| `prompt_datasets.py` | corpus generation and preflight checks |
| `cache_activations.py` | model loading and per-template activation caching |
| `inference_datasets.py` | ordered registry of the inference corpora both passes walk |
| `inference_projection.py` | the caching and projection halves of every inference corpus |
| `stated_stakes.py` | rating continuations: generation on the GPU, parsing anywhere |
| `cache_inventory.py` | cache inventory and checked batch reads |
| `bcpc_arc_length.py` | between-class PCA and `bcpc_arc_length` |
| `pls_fit.py` | weighted single-target PLS from streamed cross-products |
| `stakes_surface_pipeline.py` | stage-by-stage fitting, mapping, export, and plots |
| `stakes_height_slices.py` | height-slice surface geometry and validation |
| `stakes_surface_bundle.py` | centroid splines and loading saved bundles |
| `corpora/training/` | prompt templates and task sets the manifold is fitted on |
| `corpora/inference/` | self-contained inference-only evaluation datasets |

Caching uses left padding, an empty system prompt, thinking disabled, and a generation
prefix, via the loader, chat tokenizer, and temporary hooks in `utils/mech_interp_toolkit`
(license in `utils/mech_interp_toolkit/LICENSE`).

## Severity inference

The reference severity dataset has editable substitution lists in
`scripts/corpora/inference/severity_prompts.py`. These prompts have no severity labels and never
enter fitting or slice-grid construction. The caching pass caches their activations; after
surface export, the analysis pass applies the frozen PLS transform and saved slice mapper.

Outputs live under `artifacts/<model name>/inference/severity/`: `prompts.json`,
`activations/`, and `severity_arc_lengths.csv`. Keep these caches outside the main
`activations/` directory, which the fitting inventory scans recursively.

The CSV has exactly `template`, `severity_word`, `arc_length_parallel`, and
`arc_length_orthogonal`; `template` is the unfilled sentence. Compatible activation
caches are reused, while coordinates are recomputed against the current surface.
Projection never loads a model. The analysis notebook reports projection
diagnostics, including prompts outside saved height coverage, which snap to the nearest
saved slice. `arc_length_orthogonal` retains its legacy meaning of snapped PLS3 height.

The counterfactual dataset in
`scripts/corpora/inference/severity_flipped_prompts.py` preserves all 20 reference families,
domains, slot values, and their original row order. In each fixed compound incident, the
substituted harm is the only one successfully mitigated and all other harms remain
unresolved. The expected residual-stakes direction is therefore decreasing across the
reference order. This is an evaluation hypothesis: the exported coordinates are never
forced or asserted to follow it.

Its independent outputs live under
`artifacts/<model name>/inference/severity_flipped/`: `prompts.json`, `activations/`, and
`severity_flipped_arc_lengths.csv`. The CSV schema matches the reference severity CSV.
The cache namespace is `severity_flipped_inference`, so neither reference severity caches
nor fitting caches can be reused accidentally. Projection uses the frozen saved bundle;
these prompts never contribute to fitting or slice coverage.

The pairwise counterfactual dataset in
`scripts/corpora/inference/severity_pairwise_prompts.py` selects one curated pair from each of
10 reference families (20 prompts total). A pair reuses its family's least and most severe
slot values verbatim, but places them in a context that reverses their real stakes: the
reference-high value is simply what belongs there, while the reference-low value is what
signals that something has gone badly wrong. The `severity_pole` field names each value's
rank in the reference corpus, so a representation tracking surface wording should order a
pair the way the poles are labelled, while one tracking actual stakes should order it the
other way. This is a prediction about the contrast, not a claim that the model must
exhibit it.

Pairwise outputs are isolated under
`artifacts/<model name>/inference/severity_pairwise/`, including `prompts.json`, an
independent `severity_pairwise_inference` activation namespace, and
`severity_pairwise_arc_lengths.csv`. The CSV schema matches the other substitution-based
severity datasets, and projection uses the frozen saved bundle without entering fitting.

A wording-based dataset provides 20 fixed tasks
with eight initial variants each (160 prompts), defined in
`scripts/corpora/inference/severity_wording_prompts.py`. Each group's `variants` list is independent
and can be edited or extended without changing other groups. These are expressions of
distress and requests for help, without severity labels or calibrated rankings.

Its outputs are isolated under `artifacts/<model name>/inference/severity_wording/`,
including `prompts.json`, `activations/`, and `severity_wording_arc_lengths.csv`.
The CSV columns are `task`, `prompt`, `arc_length_parallel`, and `arc_length_orthogonal`.
The context-variation dataset in `scripts/corpora/inference/context_prompts.py` presents 20 new
tasks in four matched settings, ordered as real life, video game, fictional story, and
training simulation (80 prompts total). Within each task the core request is copied
verbatim; only the explicitly identified setting changes. The prompts carry no severity
rank or expected coordinate ordering.

Its outputs are isolated under `artifacts/<model name>/inference/context/`, including
`prompts.json`, `activations/`, and `context_arc_lengths.csv`. The CSV columns are
`task`, `context`, `prompt`, `arc_length_parallel`, and `arc_length_orthogonal`, and the
activation namespace is `context_inference`.

The situated-context dataset in `scripts/corpora/inference/situated_context_prompts.py`
takes the complementary approach: nothing is attached to the prompt, and the stakes are
carried by the task itself. Each of its 16 tasks is one stem sentence with a single slot
for a situating phrase, plus a fixed closing question, so asking what a person who has
run out of water should do `in the foyer of a cinema between films` and asking the same
about `on a desert trail in the middle of summer` are two genuinely different
predicaments described in the same words. Every task declares the `context_axis` its
four settings move along -- distance from help, how many people are exposed, what a
lapse would cost -- so a ladder is never a list of synonyms for "worse", and
`stakes_rank` runs 1 to 4 within a task rather than on a scale shared between tasks.
Two controls sit outside each ladder: `bare` drops the slot entirely, and `neutral`
fills it with a length-matched phrase that situates the task in time without bearing on
what is at stake. All situating phrases are held to one 6-10 word band, so prompt length
cannot stand in for the setting. The prompts carry no severity label and no expected
coordinate ordering (16 tasks x 6 settings = 96 prompts).

Its outputs are isolated under `artifacts/<model name>/inference/situated_context/`,
including `prompts.json`, `activations/`, and `situated_context_arc_lengths.csv`. The
CSV columns are `task`, `setting`, `stakes_rank`, `context_axis`, `is_control`, `prompt`,
`arc_length_parallel`, and `arc_length_orthogonal`; the two controls carry no rank, so
`stakes_rank` is blank on those rows. The activation namespace is
`situated_context_inference`.

Every inference dataset shares the frozen projection and export logic in
`scripts/inference_projection.py` and is listed once in `scripts/inference_datasets.py`, which
both passes walk; none contributes to fitting or slice coverage, and each reuses only its own
caches.
