"""Single-occurrence seed retention regression (N3)."""

from __future__ import annotations

import numpy as np
import pytest


def fake_cosine_similarity(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    if b is None:
        b = a
    return np.ones((len(a), len(b)))


def test_single_occurrence_seed_survives_roberta_selection(
    vis_module,
    recording_tokenizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N3: a seed that appears once must remain in selected embeddings."""
    seed = "quasar"
    paragraph = f"The {seed} term appears only once in this short paragraph."

    monkeypatch.setattr(vis_module, "cosine_similarity", fake_cosine_similarity)

    embeddings, _, _ = vis_module.train_embedding(
        sentences=[paragraph],
        context_window=3,
        stop_list=set(),
        seed_words=[seed],
        clustering_method=1,
        num_words=4,
        lemmatize=True,
        min_word_frequency=1,
        reuse_clusterings=False,
        cross_pos_normalize=False,
        distance_metric="cosine",
        custom_word_filter=lambda word: len(word) > 2,
    )

    assert embeddings is not None
    assert seed in embeddings, f"Seed {seed!r} missing from {list(embeddings.keys())!r}"
