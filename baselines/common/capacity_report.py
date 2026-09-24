"""Model capacity reporting for baseline fidelity tables.

Records trainable parameters, embedding parameters, architecture shape,
and runtime statistics for each model.
"""
from __future__ import annotations

from typing import Any

import torch.nn as nn


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count trainable and total parameters in a PyTorch model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    embedding = 0
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            embedding += sum(p.numel() for p in m.parameters())
    return {
        "trainable_params": trainable,
        "total_params": total,
        "embedding_params": embedding,
        "non_embedding_trainable": trainable - min(embedding, trainable),
    }


def model_capacity_report(model: nn.Module, model_card: dict[str, Any]) -> dict[str, Any]:
    """Full capacity report combining parameter counts with architecture card."""
    counts = count_parameters(model)
    return {
        **counts,
        "layers": model_card.get("layers", "N/A"),
        "heads": model_card.get("heads", "N/A"),
        "hidden_size": model_card.get("hidden_size", "N/A"),
        "ffn_size": model_card.get("ffn_size", "N/A"),
        "max_seq_len": model_card.get("max_seq_len", "N/A"),
    }


def format_capacity_row(name: str, report: dict[str, Any]) -> dict[str, Any]:
    """Format a capacity report as a table row for the paper."""
    return {
        "model": name,
        "params": report.get("trainable_params", 0),
        "emb_params": report.get("embedding_params", 0),
        "layers": report.get("layers", "—"),
        "heads": report.get("heads", "—"),
        "d_model": report.get("hidden_size", "—"),
        "d_ff": report.get("ffn_size", "—"),
        "max_L": report.get("max_seq_len", "—"),
    }
