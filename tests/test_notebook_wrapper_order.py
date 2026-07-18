"""Notebook wrapper ordering regressions (B1, B2) for the four run_*_pipeline wrappers.

The teaching notebook (`visualization_toolkit_final.ipynb`) carries four convenience
wrappers that duplicate the core `run_visuals_pipeline` preprocessing.  These tests load
the wrapper definitions straight from the notebook and assert, at the level of the
sentences handed to `train_embedding`, that:

- B1: rows carrying an excluded code are dropped *before* sentence extraction, so their
  text never reaches the embedding stage.
- B2: after group-label replacement the rebuilt `filtered_sentences` (used by the
  co-occurrence / PMI / TF-IDF methods) reflects the replacement, not the pre-replacement
  text.

These are discriminating assertions, not a "the notebook produced N figures" smoke test:
they fail against the pre-fix wrapper ordering and pass once the wrappers match the core
contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = TOOLKIT_ROOT / "visualization_toolkit_final.ipynb"

EXCLUDE_MARKER = "EXCLUDE_MARKER_ZZZ"
GROUP_LABEL = "careteam"
GROUP_MEMBER_A = "doctor"
GROUP_MEMBER_B = "nurse"

WRAPPERS = [
    "run_heatmap_pipeline",
    "run_tsne_pipeline",
    "run_heatmap_network_plain_pipeline",
    "run_heatmap_network_pipeline",
]


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _import_preamble(nb: dict) -> str:
    """Collect the notebook's top-level import statements.

    The wrapper definitions use names the setup cell imports (``lru_cache``, typing
    aliases, ``dataclass``, ...) that are not part of ``vis_tool_core``'s globals.  We
    replay only module-level ``import`` / ``from ... import`` lines — never the setup
    cell's executable model-loading code.
    """
    seen: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for line in "".join(cell["source"]).split("\n"):
            stripped = line.strip()
            if (
                (stripped.startswith("import ") or stripped.startswith("from "))
                and line == stripped  # module-level only (no indentation)
                and "*" not in stripped
            ):
                if stripped not in seen:
                    seen.append(stripped)
    return "\n".join(seen)


def _helpers_source(nb: dict) -> str:
    """Return the wrapper-defining notebook cell truncated before its top-level data I/O.

    The helpers cell ends with top-level data-cleaning statements (``DATA_PATH = ...`` on)
    that read and write CSV files as an import side effect.  We exec only the definitions
    above that point.
    """
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell["source"])
        if all(f"def {name}(" in src for name in WRAPPERS):
            lines = src.split("\n")
            cut = next(
                (i for i, line in enumerate(lines) if line.startswith("DATA_PATH")),
                len(lines),
            )
            return "\n".join(lines[:cut])
    raise AssertionError("Notebook cell defining the run_*_pipeline wrappers not found")


class _FakeFig:
    """Stand-in figure whose every method (including savefig) is a no-op."""

    def savefig(self, *_args, **_kwargs) -> None:
        return None

    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


def _write_pipeline_csv(path: Path) -> None:
    rows = [
        f"Keep sentence about {GROUP_MEMBER_A} and {GROUP_MEMBER_B} in clinic.",
        f"This row must drop {EXCLUDE_MARKER} from embedding inputs.",
        "Another keep sentence with hospital and clinic vocabulary.",
        "Support staff discuss patients and treatment plans daily.",
    ]
    codes = ["['keep']", "['drop_code']", "['keep']", "['keep']"]
    content = "text,codes\n" + "\n".join(
        f'"{text}","{code}"' for text, code in zip(rows, codes)
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
        num_codes=8,
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
        seed_codes=None,
        excluded_codes=["drop_code"],
        custom_stopwords={
            "the",
            "a",
            "and",
            "of",
            "to",
            "in",
            "is",
            "it",
            "that",
            "for",
            "on",
            "with",
            "must",
            "from",
            "this",
        },
        link_threshold=0.1,
        link_color_threshold=0.1,
        custom_colors=False,
        clustered=False,
        semantic_categories=None,
    )


@pytest.fixture
def wrapper_ns(vis_module):
    """Exec the notebook wrapper definitions with train_embedding / plots instrumented.

    Wrappers resolve ``train_embedding`` and the ``plot_*`` helpers from their global
    namespace at call time, so overriding them in the exec namespace after definition is
    sufficient to capture the embedding inputs without loading RoBERTa or drawing figures.
    Helper functions imported from ``vis_tool_core`` (e.g. ``tokenize_and_filter``) keep
    resolving their own globals from the real module, which the autouse
    ``inject_vis_globals`` fixture has already stubbed.
    """
    import warnings

    nb = _load_notebook()
    namespace = dict(vars(vis_module))
    namespace["warnings"] = warnings
    exec(compile(_import_preamble(nb), "<notebook-imports>", "exec"), namespace)
    exec(compile(_helpers_source(nb), "<notebook-helpers>", "exec"), namespace)

    captured: list[list[str]] = []

    def _spy_train_embedding(sentences, *_args, **_kwargs):
        captured.append(list(sentences))
        words = ["alpha", "beta", "gamma", "delta"]
        embeddings = {
            word: np.array([float(i), float(i + 1)]) for i, word in enumerate(words)
        }
        return embeddings, np.eye(len(words)), None

    namespace["train_embedding"] = _spy_train_embedding
    for plot_fn in (
        "plot_heatmap",
        "plot_tsne_dimensional_reduction",
        "plot_semantic_network",
        "create_code_cooccurrence_heatmap",
        "generate_wordcloud",
    ):
        namespace[plot_fn] = lambda *_args, **_kwargs: _FakeFig()
    namespace["WORD_FAMILIES"] = {}
    namespace["_captured"] = captured
    return namespace


@pytest.mark.parametrize("wrapper", WRAPPERS)
def test_wrapper_excluded_codes_removed_before_embedding(
    wrapper_ns,
    tmp_path: Path,
    wrapper: str,
) -> None:
    """B1: excluded-code rows must not reach the sentences passed to train_embedding."""
    csv_path = tmp_path / f"{wrapper}_b1.csv"
    _write_pipeline_csv(csv_path)
    params = _pipeline_input(
        csv_path,
        clustering_method=2,
        seed_words=f"{GROUP_LABEL}:{GROUP_MEMBER_A},{GROUP_MEMBER_B}",
    )

    wrapper_ns[wrapper](params)

    captured = wrapper_ns["_captured"]
    assert captured, f"{wrapper} never called train_embedding"
    captured_text = " ".join(s for batch in captured for s in batch)
    assert EXCLUDE_MARKER not in captured_text


@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize("clustering_method", [1, 2])
def test_wrapper_group_replacement_visible_in_embedding(
    wrapper_ns,
    tmp_path: Path,
    wrapper: str,
    clustering_method: int,
) -> None:
    """B2: group-label replacement must appear in the sentences passed to train_embedding.

    Method 1 (RoBERTa) embeds ``sentences``; methods 2-4 embed the rebuilt
    ``filtered_sentences``.  Both must show the replaced group label, which fails if
    ``filtered_sentences`` is left stale after group replacement.
    """
    csv_path = tmp_path / f"{wrapper}_b2_m{clustering_method}.csv"
    _write_pipeline_csv(csv_path)
    params = _pipeline_input(
        csv_path,
        clustering_method=clustering_method,
        seed_words=f"{GROUP_LABEL}:{GROUP_MEMBER_A},{GROUP_MEMBER_B}",
    )

    wrapper_ns[wrapper](params)

    captured = wrapper_ns["_captured"]
    assert captured, f"{wrapper} never called train_embedding"
    captured_text = " ".join(s for batch in captured for s in batch).lower()
    assert GROUP_LABEL in captured_text
    assert GROUP_MEMBER_A not in captured_text
    assert GROUP_MEMBER_B not in captured_text
