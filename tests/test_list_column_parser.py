"""Bounded list-column parser (GR-EVAL): safe replacement for eval() on codes/data_group.

``parse_list_column_cell`` replaces an ``eval()`` call that previously ran on every
stringified ``codes`` / ``data_group`` cell.  These tests pin the contract: it parses a
bracketed list of strings, treats missing/scalar values sensibly, and fails loud on
tuples, malformed literals, non-string items, and oversized cells.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_parses_bracketed_string_list(vis_module) -> None:
    assert vis_module.parse_list_column_cell("['nebeker', 'background']") == [
        "nebeker",
        "background",
    ]


def test_parses_single_item_list(vis_module) -> None:
    assert vis_module.parse_list_column_cell("['interview']") == ["interview"]


def test_passes_through_existing_list(vis_module) -> None:
    assert vis_module.parse_list_column_cell(["a", "b"]) == ["a", "b"]


def test_bare_scalar_string_becomes_single_item(vis_module) -> None:
    assert vis_module.parse_list_column_cell("interview") == ["interview"]


@pytest.mark.parametrize("empty", [None, "", "   ", np.nan, float("nan")])
def test_missing_values_become_empty_list(vis_module, empty) -> None:
    assert vis_module.parse_list_column_cell(empty) == []


def test_tuple_literal_is_rejected(vis_module) -> None:
    with pytest.raises(ValueError):
        vis_module.parse_list_column_cell("('a', 'b')")


def test_dict_literal_is_rejected(vis_module) -> None:
    with pytest.raises(ValueError):
        vis_module.parse_list_column_cell("{'a': 1}")


def test_non_string_items_are_rejected(vis_module) -> None:
    with pytest.raises(ValueError):
        vis_module.parse_list_column_cell("[1, 2, 3]")


def test_malformed_literal_fails_loud(vis_module) -> None:
    with pytest.raises(ValueError):
        vis_module.parse_list_column_cell("['unclosed")


def test_oversized_cell_is_rejected(vis_module) -> None:
    oversized = str([f"code_{i}" for i in range(1001)])
    with pytest.raises(ValueError):
        vis_module.parse_list_column_cell(oversized)


def test_no_arbitrary_code_execution(vis_module) -> None:
    """A code-injection payload must not execute; it is not a list literal."""
    with pytest.raises(ValueError):
        vis_module.parse_list_column_cell("[__import__('os').getcwd()]")
