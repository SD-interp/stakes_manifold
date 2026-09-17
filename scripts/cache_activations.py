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


def load_model(config):
    """Load the model and chat tokenizer with the settings every cache records.

    Weights are downloaded to and read from `config.hf_cache_dir`; None means the
    Hugging Face default location.
    """
    cache_dir = None if config.hf_cache_dir is None else str(config.hf_cache_dir)
    if config.naming_convention == "gemma4":
        import torch
        from transformers import AutoModelForMultimodalLM, AutoTokenizer
        from utils.mech_interp_toolkit.utils import ChatTemplateTokenizer

        device = resolve_device(config)
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, use_fast=True, padding_side="left", cache_dir=cache_dir
        )
        tokenizer = ChatTemplateTokenizer(tokenizer, system_prompt="")
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


def run(config, datasets, model=None, tokenizer=None, force=False):
    """Cache every template group of `datasets` ({corpus: records}); return the cache paths.

    Completed caches are reused after fingerprint and row checks; a mismatch raises
    unless `force`. Without a model argument, the model loads only when needed.
    """
    if not datasets or not set(datasets) <= set(NO_TIME_CORPORA):
        raise ValueError("This caching entry point accepts horizon-free corpora only.")
    return _run(config, datasets, config.activations_dir, model, tokenizer, force)


def run_inference(
    config,
    records,
    destination,
    model=None,
    tokenizer=None,
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

    if (model is None) != (tokenizer is None):
        raise ValueError("Supply both model and tokenizer, or neither.")
    if config.position != -1:
        raise ValueError("Expected final token (-1).")
    device = resolve_device(config) if model is None else str(model.device)
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
            if model is None:
                model, tokenizer = load_model(config)
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
