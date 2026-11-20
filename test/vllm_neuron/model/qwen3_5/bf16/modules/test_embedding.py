# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 embedding parallelism tests."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_neuron.model.qwen3_5.model_bf16 import Qwen3_5Model

pytestmark = [pytest.mark.fast]


class _RecordingEmbedding(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.calls = []

    def forward(self, input_ids, scatter_tokens=True, rank=None):
        self.calls.append((scatter_tokens, rank))
        return torch.zeros(input_ids.shape[0], self.hidden_size)


class _ZeroRotary(nn.Module):
    def forward(self, position_ids, dtype):
        shape = (*position_ids.shape[-2:], 2)
        value = torch.zeros(shape, dtype=dtype)
        return value, value


def test_embedding_keeps_decode_tokens_tp_replicated():
    """A one-token TP decode must use all-reduce instead of reduce-scatter."""
    model = Qwen3_5Model.__new__(Qwen3_5Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(layer_types=[])
    model.embed_tokens = _RecordingEmbedding(hidden_size=8)
    model.rotary_emb = _ZeroRotary()
    model.layers = nn.ModuleList()
    model.norm = nn.Identity()
    model._first_attention_layer = None

    rank = torch.tensor(2, dtype=torch.int32)
    output = model(
        input_ids=torch.tensor([[7]]),
        position_ids=torch.tensor([[0]]),
        rank=rank,
    )

    assert output.shape == (1, 1, 8)
    assert len(model.embed_tokens.calls) == 1
    scatter_tokens, actual_rank = model.embed_tokens.calls[0]
    assert scatter_tokens is False
    assert actual_rank is rank
