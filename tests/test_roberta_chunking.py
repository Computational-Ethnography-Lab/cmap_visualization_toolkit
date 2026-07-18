"""RoBERTa chunking regression (B4)."""

from __future__ import annotations

import numpy as np
import pytest


def fake_cosine_similarity(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """Return ones matrix matching sklearn cosine_similarity shapes."""
    if b is None:
        b = a
    return np.ones((len(a), len(b)))


def _long_paragraph(min_words: int = 40) -> str:
    base = (
        "Patients describe symptoms and treatment goals during interviews. "
        "Clinicians document observations and coordinate follow up care plans. "
    )
    words = base.split()
    while len(words) < min_words:
        words.extend(base.split())
    return " ".join(words[:min_words])


def test_tokenizer_called_per_chunk_not_whole_paragraph(
    vis_module,
    recording_tokenizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B4: split_into_chunks output must be tokenized, not the full paragraph."""
    vis_module.MAX_TOKENS = 10
    paragraph = _long_paragraph()

    chunks = vis_module.split_into_chunks(paragraph, max_tokens=vis_module.MAX_TOKENS)
    assert len(chunks) > 1, "Test paragraph must produce multiple chunks"

    monkeypatch.setattr(vis_module, "cosine_similarity", fake_cosine_similarity)

    vis_module.train_embedding(
        sentences=[paragraph],
        context_window=3,
        stop_list=set(),
        seed_words=["patient"],
        clustering_method=1,
        num_words=5,
        lemmatize=True,
        min_word_frequency=1,
        reuse_clusterings=False,
        cross_pos_normalize=False,
        distance_metric="cosine",
        custom_word_filter=lambda word: len(word) > 2,
    )

    assert len(recording_tokenizer.calls) > 1
    for call_text in recording_tokenizer.calls:
        assert call_text != paragraph
        assert len(call_text.split()) <= vis_module.MAX_TOKENS + 5
