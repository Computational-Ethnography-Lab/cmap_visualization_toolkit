"""t-SNE metric and input-matrix regressions (B5)."""

from __future__ import annotations

import numpy as np
import pytest

METHOD1 = 1
METHOD2 = 2
METHOD3 = 3
METHOD4 = 4


def _words_and_matrices(n: int = 4) -> tuple[list[str], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    words = [f"word{i}" for i in range(n)]
    embeddings = {word: np.array([float(i), float(i + 1), float(i + 2)]) for i, word in enumerate(words)}
    similarity = np.array(
        [
            [1.0, 0.85, 0.55, 0.25],
            [0.85, 1.0, 0.65, 0.35],
            [0.55, 0.65, 1.0, 0.45],
            [0.25, 0.35, 0.45, 1.0],
        ],
        dtype=float,
    )
    co_occurrence = np.array(
        [
            [1.0, 0.6, 0.3, 0.1],
            [0.6, 1.0, 0.4, 0.2],
            [0.3, 0.4, 1.0, 0.15],
            [0.1, 0.2, 0.15, 1.0],
        ],
        dtype=float,
    )
    return words, embeddings, similarity, co_occurrence


def _expected_precomputed(similarity: np.ndarray, keep: list[str], words: list[str]) -> np.ndarray:
    idx = [words.index(word) for word in keep]
    sub = similarity[np.ix_(idx, idx)].copy()
    dist = (1.0 - sub).clip(0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    return dist


def _run_tsne(
    vis_module,
    tsne_spy,
    *,
    clustering_method: int,
    distance_metric: str,
) -> None:
    words, embeddings, similarity, co_occurrence = _words_and_matrices()
    vis_module.plot_tsne_dimensional_reduction(
        word_embeddings=embeddings,
        similarity_matrix=similarity,
        co_occurrence_matrix=co_occurrence,
        clustering_method=clustering_method,
        seed_words=[words[0], words[1]],
        distance_metric=distance_metric,
    )


@pytest.mark.parametrize(
    ("clustering_method", "distance_metric"),
    [
        (METHOD2, "cosine"),
        (METHOD3, "cosine"),
        (METHOD4, "cosine"),
        (METHOD4, "default"),
        (METHOD2, "default"),
    ],
)
def test_tsne_uses_precomputed_distance_for_similarity_methods(
    vis_module,
    tsne_spy,
    clustering_method: int,
    distance_metric: str,
) -> None:
    """Similarity-driven t-SNE must use metric='precomputed' on (1-sim).clip(0,2)."""
    _run_tsne(
        vis_module,
        tsne_spy,
        clustering_method=clustering_method,
        distance_metric=distance_metric,
    )
    assert tsne_spy.instances, "TSNE was not invoked"
    capture = tsne_spy.instances[-1]
    assert capture.ctor_kwargs.get("metric") == "precomputed"

    words, _, similarity, co_occurrence = _words_and_matrices()
    keep = capture.fit_input is not None and capture.fit_input.shape[0]
    assert keep and keep >= 3

    n = capture.fit_input.shape[0]
    candidate_keep_lists = []
    for i in range(len(words)):
        for j in range(i + 3, len(words) + 1):
            for combo in __import__("itertools").combinations(words, j):
                if len(combo) == n:
                    candidate_keep_lists.append(list(combo))
    matched = None
    source_matrices = [similarity]
    if clustering_method == METHOD2 and distance_metric == "default":
        source_matrices = [co_occurrence, similarity]
    for matrix_source in source_matrices:
        for keep_guess in candidate_keep_lists:
            expected = _expected_precomputed(matrix_source, keep_guess, words)
            if expected.shape == capture.fit_input.shape and np.allclose(
                expected, capture.fit_input, atol=1e-6
            ):
                matched = expected
                break
        if matched is not None:
            break
    assert matched is not None, "fit_transform input is not (1-sim).clip(0,2) with zero diagonal"


def test_method1_tsne_uses_embedding_vectors_not_similarity_rows(
    vis_module,
    tsne_spy,
) -> None:
    """Method 1 t-SNE must stack embedding vectors for kept words."""
    words, embeddings, similarity, co_occurrence = _words_and_matrices()
    vis_module.plot_tsne_dimensional_reduction(
        word_embeddings=embeddings,
        similarity_matrix=similarity,
        co_occurrence_matrix=co_occurrence,
        clustering_method=METHOD1,
        seed_words=[words[0], words[1]],
        distance_metric="cosine",
    )
    assert tsne_spy.instances, "TSNE was not invoked"
    capture = tsne_spy.instances[-1]
    fit_input = capture.fit_input
    assert fit_input is not None

    # Expected: each row is the embedding vector for a kept word (dim=3), not a similarity row.
    assert fit_input.shape[1] == 3
    for row in fit_input:
        assert not np.any((row >= 0.0) & (row <= 1.0) & (row != row[0])) or row.shape[0] == 3
        matches_embedding = any(np.allclose(row, embeddings[word]) for word in words)
        assert matches_embedding, "Method 1 must pass stacked embedding vectors to TSNE"
