"""Cross-POS normalization dedup regression (N1)."""

from __future__ import annotations

import pytest


def test_cross_pos_normalize_preserves_token_multiset(
    vis_module,
) -> None:
    """N1: cross_pos_normalize must relabel stems without dropping repeated tokens."""
    sentence = "learn learn learning learning"
    baseline = vis_module.tokenize_and_filter(
        [sentence],
        stop_list=set(),
        lemmatize=True,
        cross_pos_normalize=False,
    )
    tokens = vis_module.tokenize_and_filter(
        [sentence],
        stop_list=set(),
        lemmatize=True,
        cross_pos_normalize=True,
    )
    assert len(baseline) >= 4, f"Fixture sanity: baseline too short: {baseline!r}"
    assert len(tokens) == len(baseline), (
        f"N1 dedup collapsed tokens: got {tokens!r} (len={len(tokens)}) "
        f"vs baseline {baseline!r} (len={len(baseline)})"
    )
