"""Pipeline ordering regressions: excluded codes and group replacement (B1, B2)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

EXCLUDE_MARKER = "EXCLUDE_MARKER_ZZZ"
GROUP_LABEL = "careteam"
GROUP_MEMBER_A = "doctor"
GROUP_MEMBER_B = "nurse"


def _write_pipeline_csv(path: Path) -> None:
    rows = [
        f"Keep sentence about {GROUP_MEMBER_A} and {GROUP_MEMBER_B} in clinic.",
        f"This row must drop {EXCLUDE_MARKER} from embedding inputs.",
        "Another keep sentence with hospital and clinic vocabulary.",
        "Support staff discuss patients and treatment plans daily.",
    ]
    codes = ["['keep']", "['drop_code']", "['keep']", "['keep']"]
    content = "text,codes\n" + "\n".join(
        f"\"{text}\",\"{code}\"" for text, code in zip(rows, codes)
    )
    path.write_text(content, encoding="utf-8")


def _pipeline_input(
    csv_path: Path,
    *,
    clustering_method: int,
    seed_words: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        filepath=str(csv_path),
        stop_list=None,
        num_words=8,
        clustering_method=clustering_method,
        distance_metric="cosine" if clustering_method != 2 else "default",
        reuse_clusterings=False,
        window_size=3,
        min_word_frequency=1,
        cross_pos_normalize=False,
        projects=None,
        data_groups=None,
        codes=None,
        seed_words=seed_words,
        excluded_codes=["drop_code"],
        custom_stopwords=set(),
        link_threshold=0.1,
        link_color_threshold=0.1,
        custom_colors=False,
    )


def _patch_post_embedding_noops(vis_module, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vis_module, "plot_heatmap", lambda *args, **kwargs: None)
    monkeypatch.setattr(vis_module, "plot_tsne_dimensional_reduction", lambda *args, **kwargs: None)
    monkeypatch.setattr(vis_module, "plot_semantic_network", lambda *args, **kwargs: None)


def test_excluded_codes_removed_before_train_embedding(
    vis_module,
    train_embedding_spy,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B1: rows with excluded codes must not reach train_embedding sentences."""
    csv_path = tmp_path / "pipeline_b1.csv"
    _write_pipeline_csv(csv_path)
    _patch_post_embedding_noops(vis_module, monkeypatch)

    params = _pipeline_input(
        csv_path,
        clustering_method=2,
        seed_words=f"{GROUP_LABEL}:{GROUP_MEMBER_A},{GROUP_MEMBER_B}",
    )
    vis_module.run_visuals_pipeline(params)

    captured_text = " ".join(train_embedding_spy.all_sentences)
    assert EXCLUDE_MARKER not in captured_text


@pytest.mark.parametrize("clustering_method", [1, 2])
def test_group_replacement_visible_in_train_embedding_sentences(
    vis_module,
    train_embedding_spy,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clustering_method: int,
) -> None:
    """B2: group-label replacement must appear in sentences passed to train_embedding."""
    csv_path = tmp_path / f"pipeline_b2_m{clustering_method}.csv"
    _write_pipeline_csv(csv_path)
    _patch_post_embedding_noops(vis_module, monkeypatch)

    params = _pipeline_input(
        csv_path,
        clustering_method=clustering_method,
        seed_words=f"{GROUP_LABEL}:{GROUP_MEMBER_A},{GROUP_MEMBER_B}",
    )
    vis_module.run_visuals_pipeline(params)

    captured_text = " ".join(train_embedding_spy.all_sentences).lower()
    assert GROUP_LABEL in captured_text
    assert GROUP_MEMBER_A not in captured_text
    assert GROUP_MEMBER_B not in captured_text
