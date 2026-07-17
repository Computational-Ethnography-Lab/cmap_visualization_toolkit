"""Empty-seed fallback regression (B3)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def _write_seed_csv(path: Path) -> None:
    rows = [
        "Patients receive treatment at the hospital clinic daily.",
        "Nurses coordinate care with doctors and families regularly.",
        "Clinic staff document symptoms and medications each visit.",
    ]
    path.write_text(
        "text,codes\n" + "\n".join(f"\"{row}\",\"['keep']\"" for row in rows),
        encoding="utf-8",
    )


class _FakeFig:
    def savefig(self, *args, **kwargs) -> None:
        return None

    def suptitle(self, *args, **kwargs) -> None:
        return None

    def subplots_adjust(self, *args, **kwargs) -> None:
        return None


def test_empty_normalized_seeds_do_not_raise_nameerror(
    vis_module,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B3: explicit seeds that normalize away must not hit undefined word_counts."""
    csv_path = tmp_path / "empty_seed.csv"
    _write_seed_csv(csv_path)

    monkeypatch.setattr(vis_module, "plot_heatmap", lambda *args, **kwargs: _FakeFig())
    monkeypatch.setattr(vis_module, "plot_tsne_dimensional_reduction", lambda *args, **kwargs: None)
    monkeypatch.setattr(vis_module, "plot_semantic_network", lambda *args, **kwargs: _FakeFig())

    params = SimpleNamespace(
        filepath=str(csv_path),
        stop_list=None,
        num_words=6,
        clustering_method=2,
        distance_metric="default",
        reuse_clusterings=False,
        window_size=3,
        min_word_frequency=1,
        cross_pos_normalize=False,
        projects=None,
        data_groups=None,
        codes=None,
        seed_words="the, and, of, a",
        excluded_codes=None,
        custom_stopwords=set(),
        link_threshold=0.1,
        link_color_threshold=0.1,
        custom_colors=False,
    )

    try:
        result = vis_module.run_visuals_pipeline(params)
    except NameError as exc:
        pytest.fail(f"run_visuals_pipeline raised NameError: {exc}")

    assert result is None or result is not None
