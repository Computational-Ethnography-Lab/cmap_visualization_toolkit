"""t-SNE vocabulary ordering determinism regression (N5-DET)."""

from __future__ import annotations

import numpy as np
import pytest


def _deterministic_keep_order(
    words: list[str],
    similarity: np.ndarray,
    seed_words: list[str],
) -> list[str]:
    """Reference ordering: seeds in input order, then scored candidates."""
    present_seeds = [seed for seed in seed_words if seed in words]
    stop_words_std = {
        "the",
        "and",
        "to",
        "of",
        "a",
        "in",
        "is",
        "it",
        "that",
        "was",
        "for",
        "on",
        "with",
        "as",
        "be",
        "at",
        "by",
        "have",
        "are",
        "this",
    }
    stop_filter = stop_words_std

    candidates = []
    for word in words:
        if word in present_seeds or word in stop_filter:
            continue
        idx = words.index(word)
        max_sim = max(similarity[idx, words.index(seed)] for seed in present_seeds)
        candidates.append((word, max_sim, idx))
    candidates.sort(key=lambda item: (-item[1], item[2]))

    cap = min(50, len(words))
    max_added = max(0, cap - len(present_seeds))
    added = [word for word, _, _ in candidates[:max_added]]

    keep = list(present_seeds)
    for word in added:
        if word not in keep:
            keep.append(word)
    return keep


def test_tsne_keep_order_is_deterministic(
    vis_module,
    tsne_spy,
) -> None:
    """N5-DET: matrix row order must follow deterministic seed/neighbour ordering."""
    words = ["seedb", "seeda", "wordc", "worda", "wordb"]
    embeddings = {word: np.array([float(i), float(i + 1), float(i + 2)]) for i, word in enumerate(words)}
    similarity = np.array(
        [
            [1.0, 0.70, 0.40, 0.55, 0.35],
            [0.70, 1.0, 0.50, 0.60, 0.45],
            [0.40, 0.50, 1.0, 0.30, 0.65],
            [0.55, 0.60, 0.30, 1.0, 0.25],
            [0.35, 0.45, 0.65, 0.25, 1.0],
        ],
        dtype=float,
    )
    co_occurrence = similarity.copy()
    seed_words = ["seeda", "seedb"]

    vis_module.plot_tsne_dimensional_reduction(
        word_embeddings=embeddings,
        similarity_matrix=similarity,
        co_occurrence_matrix=co_occurrence,
        clustering_method=2,
        seed_words=seed_words,
        distance_metric="cosine",
    )

    assert tsne_spy.instances, "TSNE was not invoked"
    fit_input = tsne_spy.instances[-1].fit_input
    assert fit_input is not None

    expected_keep = _deterministic_keep_order(
        words,
        similarity,
        seed_words,
    )
    assert len(expected_keep) == fit_input.shape[0]

    idx = [words.index(item) for item in expected_keep]
    expected_matrix = similarity[np.ix_(idx, idx)]
    dist = (1.0 - expected_matrix).clip(0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    observed_rows = [tuple(np.round(row, 6)) for row in fit_input]
    expected_rows = [tuple(np.round(row, 6)) for row in dist]
    assert observed_rows == expected_rows, (
        "TSNE input row order must follow deterministic seed/neighbour ordering"
    )
