"""Load the model and cache final-prompt-token layer outputs, one file per template."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from .pipeline_config import NO_TIME_CORPORA, SEED, layer_index

CACHE_SCHEMA_VERSION = 1


def template_groups(records):
    groups = defaultdict(list)
    for record in records:
        groups[str(record["template_id"])].append(record)
    return groups


def fingerprint(records, config):
    return hashlib.sha256(
        json.dumps(
            dict(records=records, config=config),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def resolve_device(config):
    import torch

    return config.device or ("cuda" if torch.cuda.is_available() else "cpu")


def activation_dtype(device):
    import torch

    if device.startswith("cuda"):
        return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    return "float32"


def load_tokenizer(config):
    """The chat tokenizer `load_model` pairs with the model, without loading weights.

    Its formatting (empty system prompt, generation prompt, thinking off) is what every
    cache was built from, so it also serves token counts of cached prompts.
    """
    from transformers import AutoTokenizer
    from utils.mech_interp_toolkit.utils import ChatTemplateTokenizer

    cache_dir = None if config.hf_cache_dir is None else str(config.hf_cache_dir)
    mistral = config.naming_convention == "mistral3"
    # Mistral 3 checkpoints ship a pre-tokenizer regex that mis-splits text
    # unless corrected at load time.
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, use_fast=True, padding_side="left", cache_dir=cache_dir,
        **(dict(fix_mistral_regex=True) if mistral else {}),
    )
    if mistral:
        # The chat template ships only in the processor's chat_template.json,
        # which AutoTokenizer does not read. Without an explicit empty system
        # message it injects a default system prompt stating today's date.
        from huggingface_hub import hf_hub_download

        template_path = hf_hub_download(
            config.model_name, "chat_template.json", cache_dir=cache_dir
        )
        tokenizer.chat_template = json.loads(
            Path(template_path).read_text(encoding="utf-8")
        )["chat_template"]
    return ChatTemplateTokenizer(tokenizer, system_prompt="", send_empty_system_prompt=mistral)


def load_model(config):
    """Load the model and chat tokenizer with the settings every cache records.

    This is the only place in the package that loads weights, and nothing calls it
    implicitly: the notebook driving a run calls it once and hands the model and
    tokenizer to every caching, inference and rating stage.

    Weights are downloaded to and read from `config.hf_cache_dir`; None means the
    Hugging Face default location.
    """
    cache_dir = None if config.hf_cache_dir is None else str(config.hf_cache_dir)
    if config.naming_convention in ("gemma4", "mistral3"):
        import torch
        from transformers import AutoModelForMultimodalLM

        device = resolve_device(config)
        tokenizer = load_tokenizer(config)
        model = AutoModelForMultimodalLM.from_pretrained(
            config.model_name,
            dtype=getattr(torch, activation_dtype(device)),
            device_map="auto",
            attn_implementation="sdpa",
            cache_dir=cache_dir,
        ).eval()
        return model, tokenizer

    from utils.mech_interp_toolkit.utils import load_model_tokenizer_config

    device = resolve_device(config)
    model, tokenizer, _ = load_model_tokenizer_config(
        config.model_name,
        device=device,
        dtype=activation_dtype(device),
        padding_side="left",
        attn_type="sdpa",
        system_prompt="",
        cache_dir=cache_dir,
    )
    return model, tokenizer


def cache_path(config, dataset, template_id, destination=None):
    # Hash prevents collisions after sanitizing template IDs for Windows.
    safe_id = (
        re.sub(r"[^A-Za-z0-9._-]+", "-", template_id).strip(".")[:80] or "template"
    )
    suffix = hashlib.sha256(template_id.encode()).hexdigest()[:12]
    directory = config.activations_dir if destination is None else Path(destination)
    return directory / f"{dataset}--{safe_id}-{suffix}.pt"


def cached_paths(config, records, destination, dataset, suffix=".pt"):
    """Return the cache file of every template group in `records`, without a model.

    `cache_path` is a pure function of the template id and destination, so the
    analysis half resolves what the caching half wrote from the records alone. A
    missing file means that model's caching pass has not been run, or not been
    copied across, and says so rather than silently projecting a partial corpus.
    """
    destination = Path(destination)
    paths = [
        cache_path(config, dataset, template_id, destination).with_suffix(suffix)
        for template_id in sorted(template_groups(records))
    ]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(paths)} caches are missing from {destination} "
            f"(first: {missing[0]}); run the caching notebook for this model first."
        )
    return paths


def run(config, datasets, model, tokenizer, force=False):
    """Cache every template group of `datasets` ({corpus: records}); return the cache paths.

    Completed caches are reused after fingerprint and row checks; a mismatch raises
    unless `force`. The caller owns the model: this module never loads one itself,
    so a single load serves every stage of a run.
    """
    if not datasets or not set(datasets) <= set(NO_TIME_CORPORA):
        raise ValueError("This caching entry point accepts horizon-free corpora only.")
    return _run(config, datasets, config.activations_dir, model, tokenizer, force)


def run_inference(
    config,
    records,
    destination,
    model,
    tokenizer,
    force=False,
    *,
    namespace="severity_inference",
):
    """Cache inference rows outside the recursively scanned fitting directory."""
    destination = Path(destination).resolve()
    fitting = config.activations_dir.resolve()
    if destination == fitting or fitting in destination.parents:
        raise ValueError(
            "Inference caches must be outside the fitting activations directory."
        )
    if not records:
        raise ValueError("Inference records must not be empty.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", namespace) or namespace in NO_TIME_CORPORA:
        raise ValueError(
            "Expected a distinct inference cache namespace with no path separators."
        )
    return _run(config, {namespace: records}, destination, model, tokenizer, force)


def _run(config, datasets, destination, model, tokenizer, force):
    import torch

    from utils.mech_interp_toolkit.hook_utils import HookSpecPost, temporary_hooks

    if model is None or tokenizer is None:
        raise ValueError(
            "A loaded model and tokenizer are required; the caller loads them once "
            "with load_model and passes them to every stage."
        )
    if config.position != -1:
        raise ValueError("Expected final token (-1).")
    device = str(model.device)
    layer = layer_index(config.layer_component)
    layer_component = f"layer_out/{layer}"
    cache_config = dict(
        model_name=config.model_name,
        layer_component=layer_component,
        position=-1,
        dtype=activation_dtype(device),
        seed=SEED,
        system_prompt="",
        enable_thinking=False,
        add_generation_prompt=True,
        cache_schema_version=CACHE_SCHEMA_VERSION,
    )
    # Preserve existing Llama cache fingerprints; other conventions must not
    # reuse activations captured with a different module path.
    if config.naming_convention != "llama":
        cache_config.update(naming_convention=config.naming_convention,
                            layer_module_name=config.layer_module_name)
    Path(destination).mkdir(parents=True, exist_ok=True)
    paths = []
    for dataset, records in datasets.items():
        groups = template_groups(records)
        print(
            f"{dataset}: {len(records):,} prompts, {len(groups)} template caches",
            flush=True,
        )
        for template_id, rows in sorted(groups.items()):
            path = cache_path(config, dataset, template_id, destination)
            digest = fingerprint(rows, cache_config)
            if path.exists() and not force:
                cached = torch.load(
                    path, map_location="cpu", weights_only=False, mmap=True
                )
                tensor = cached.get("activations", {}).get(layer_component)
                valid = (
                    cached.get("input_sha256") == digest
                    and tensor is not None
                    and tensor.ndim == 3
                    and tensor.shape[:2] == (len(rows), 1)
                    and cached.get("positions") == [-1]
                    and cached.get("prompts") == [row["text"] for row in rows]
                    and cached.get("prompt_metadata") == rows
                )
                del cached, tensor
                if not valid:
                    raise RuntimeError(
                        f"Stale or incompatible cache: {path}; set FORCE=True to rebuild."
                    )
                paths.append(path)
                continue
            acts = None
            for start in range(0, len(rows), config.batch_size):
                batch = rows[start : start + config.batch_size]
                inputs = {
                    key: value.to(model.device)
                    for key, value in tokenizer([row["text"] for row in batch]).items()
                }
                captured = []

                def capture(layer_component_, inputs_, output):
                    hidden = output[0] if isinstance(output, (tuple, list)) else output
                    captured.append(hidden[:, -1:, :].detach().to("cpu").clone())

                spec = HookSpecPost((layer, "layer_out"), capture)
                with (
                    torch.inference_mode(),
                    temporary_hooks(
                        dict(model.named_modules()), {"fwd": [spec]}, early_exit=True,
                        hookloc_resolver=lambda _: config.layer_module_name,
                    ),
                ):
                    model(**inputs, use_cache=False)
                if len(captured) != 1 or not torch.isfinite(captured[0]).all():
                    raise RuntimeError("Expected exactly one finite layer capture.")
                if acts is None:
                    acts = torch.empty(
                        (len(rows), 1, captured[0].shape[-1]), dtype=captured[0].dtype
                    )
                acts[start : start + len(batch)] = captured[0]
            payload = dict(
                config=cache_config,
                layer_component=layer_component,
                positions=[-1],
                activations={layer_component: acts},
                prompts=[row["text"] for row in rows],
                prompt_metadata=rows,
                input_sha256=digest,
            )
            temporary = path.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            temporary.replace(path)
            paths.append(path)
            print(
                f"Cached: {path.name} ({path.stat().st_size / 2**20:.1f} MiB)",
                flush=True,
            )
    return paths
