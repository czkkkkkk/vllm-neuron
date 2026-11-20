# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 hybrid model: Gated DeltaNet linear attention with full attention.

Target checkpoint: ``Qwen/Qwen3.8-27B``.

The Qwen3.8 checkpoint declares ``Qwen3_5ForConditionalGeneration`` as its
architecture and ``qwen3_5`` as its model type. The package and the class names
are the same as in vLLM (``vllm/model_executor/models/qwen3_5.py``).
"""

from .config import FULL_ATTENTION, LINEAR_ATTENTION, Qwen3_5TextConfig
from .factory import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration

__all__ = [
    "FULL_ATTENTION",
    "LINEAR_ATTENTION",
    "Qwen3_5TextConfig",
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
]
