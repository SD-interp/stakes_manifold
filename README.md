# Arc-coordinate pipeline

Compute `arc_length_parallel` and `arc_length_orthogonal` for horizon-free task prompts.
The orthogonal column is the existing name for PLS3 height relative to zero; parallel
length is signed slice length.

## Run

```powershell
python -m pip install -r requirements.txt
python -m jupyterlab notebooks/arc_length_pipeline.ipynb
```

Run `notebooks/arc_length_pipeline.ipynb` top to bottom on a local GPU machine. Everything
runs and is stored locally; nothing is uploaded. Its config cell selects the model, batch size, and
stakes class-merging dict; layer, position, corpora, and `FORCE` are set there too.
Stages: load model, generate prompts, cache activations, BCPC target, PLS, centroid-plane
rotation, surface fit, stated-row projection, slice cache, arc lengths, export, plots.

## Artifacts

Everything generated for a model lives under `artifacts/<model name>/`, e.g.
`artifacts/Qwen3-4B-Instruct-2507/`:

- `datasets/` - prompt JSON per corpus
- `activations/` - one `.pt` cache per corpus/template, preserving equal-file weights
- `surface/` - `rows.parquet`, `model.npz`, `model.json`, and `slice_mapping_checkpoints/`
- `plots/` - self-contained HTML figures: `bcpc_centroid_spline.html`,
  `pls_centroid_splines.html`, `pls_surface.html`

Completed caches are fingerprint-checked before reuse; set `FORCE = True` to rebuild stale
ones (for example after changing the layer). Compatible stated caches placed in
`activations/` are projected with frozen transforms but never enter any fit.

## Code layout

All logic is in `scripts/`; the notebook only orchestrates.

| module | role |
|---|---|
| `pipeline_config.py` | `RunConfig`: model, layer, batch size, stakes merges, artifact paths |
| `prompt_datasets.py` | corpus generation and preflight checks |
| `cache_activations.py` | model loading and per-template activation caching |
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

The final notebook section generates 20 templates with editable substitution lists in
`scripts/corpora/inference/severity_prompts.py`. These prompts have no severity labels and never
enter fitting or slice-grid construction. After surface export, it caches their
activations and applies the frozen PLS transform and saved slice mapper.

Outputs live under `artifacts/<model name>/inference/severity/`: `prompts.json`,
`activations/`, and `severity_arc_lengths.csv`. Keep these caches outside the main
`activations/` directory, which the fitting inventory scans recursively.

The CSV has exactly `template`, `severity_word`, `arc_length_parallel`, and
`arc_length_orthogonal`; `template` is the unfilled sentence. Compatible activation
caches are reused, while coordinates are recomputed against the current surface.
The model loads only when caching requires it. The notebook reports projection
diagnostics, including prompts outside saved height coverage, which snap to the nearest
saved slice. `arc_length_orthogonal` retains its legacy meaning of snapped PLS3 height.

The next notebook section runs the counterfactual dataset in
`scripts/corpora/inference/severity_flipped_prompts.py`. It preserves all 20 reference families,
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

The following notebook section runs a wording-based dataset: 20 fixed tasks
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

All five datasets share frozen projection and export logic in `scripts/inference_projection.py`;
none contributes to fitting or slice coverage, and each reuses only its own caches.
