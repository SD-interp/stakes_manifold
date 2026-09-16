"""Inference-only evaluation corpora.

Each module here is self-contained and exposes the same `build_prompt_records`
interface as :mod:`scripts.corpora.inference.severity_prompts`, so an inference
corpus can be projected onto a fitted manifold without importing anything from
:mod:`scripts.corpora.training`.
"""
