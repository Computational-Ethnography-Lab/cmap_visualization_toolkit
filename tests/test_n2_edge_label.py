"""Semantic-network edge label regression (N2)."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest


class _FakePatch:
    def set_facecolor(self, color: str) -> None:
        return None


class FakeFigure:
    patch = _FakePatch()


class TextCaptureAxes:
    """Minimal axes stub capturing ax.text labels."""

    def __init__(self) -> None:
        self.text_calls: list[str] = []
        self.patches: list[Any] = []

    def text(self, x: float, y: float, text: str, **kwargs: Any) -> None:
        self.text_calls.append(text)

    def add_patch(self, patch: Any) -> None:
        self.patches.append(patch)

    def plot(self, *args: Any, **kwargs: Any) -> None:
        return None

    def set_facecolor(self, color: str) -> None:
        return None


def test_edge_labels_use_raw_similarity_not_rescaled_width(
    vis_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N2: cos= labels must reflect raw similarity in [0, 1], not rescaled edge width."""
    words = ["seed", "alpha", "beta"]
    embeddings = {word: np.array([float(i), float(i + 1)]) for i, word in enumerate(words)}
    similarity = np.array(
        [
            [1.0, 0.82, 0.61],
            [0.82, 1.0, 0.44],
            [0.61, 0.44, 1.0],
        ],
        dtype=float,
    )
    semantic_categories = {
        "Seed": {"color": "#FFD700", "words": ["seed"]},
        "GroupA": {"color": "#1f77b4", "words": ["alpha"]},
        "GroupB": {"color": "#ff7f0e", "words": ["beta"]},
    }

    capture_axes = TextCaptureAxes()

    def _fake_subplots(*args: Any, **kwargs: Any) -> tuple[FakeFigure, TextCaptureAxes]:
        return FakeFigure(), capture_axes

    monkeypatch.setattr(vis_module.plt, "subplots", _fake_subplots)
    monkeypatch.setattr(vis_module.plt, "scatter", lambda *args, **kwargs: None)
    monkeypatch.setattr(vis_module.plt, "text", lambda *args, **kwargs: None)
    monkeypatch.setattr(vis_module.plt, "legend", lambda *args, **kwargs: None)
    monkeypatch.setattr(vis_module.plt, "axis", lambda *args, **kwargs: None)
    monkeypatch.setattr(vis_module.plt, "tight_layout", lambda *args, **kwargs: None)

    vis_module.plot_semantic_network(
        embeddings,
        ["seed"],
        clustering_method=1,
        similarity_matrix=similarity,
        co_occurrence_matrix=None,
        semantic_categories=semantic_categories,
        link_threshold=0.1,
        link_color_threshold=0.3,
        distance_metric="cosine",
    )

    cos_labels = [
        float(match.group(1))
        for text in capture_axes.text_calls
        for match in [re.search(r"cos=([0-9.]+)", text)]
        if match
    ]
    assert cos_labels, f"Expected cos= edge labels, got {capture_axes.text_calls!r}"
    assert all(0.0 <= value <= 1.0 for value in cos_labels), cos_labels
