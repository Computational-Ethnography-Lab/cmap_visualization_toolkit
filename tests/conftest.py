"""Pytest fixtures for CMAP visualization toolkit regression tests."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
FUNCTION_DIR = TOOLKIT_ROOT / "function"
if str(FUNCTION_DIR) not in sys.path:
    sys.path.insert(0, str(FUNCTION_DIR))

vis_tool_core = importlib.import_module("vis_tool_core")


def _ensure_nltk_resources() -> None:
    import nltk

    resources = (
        "tokenizers/punkt",
        "tokenizers/punkt_tab",
        "corpora/stopwords",
        "corpora/wordnet",
        "taggers/averaged_perceptron_tagger",
        "taggers/averaged_perceptron_tagger_eng",
    )
    for resource in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            package = resource.split("/")[-1]
            nltk.download(package, quiet=True)


_ensure_nltk_resources()


def _default_stop_words() -> set[str]:
    try:
        return set(stopwords.words("english"))
    except LookupError:
        return {
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
        }


class TokenizerBatch(dict):
    """Mimic HuggingFace batch objects that support `.to(device)`."""

    def to(self, device: Any) -> TokenizerBatch:
        return self


@dataclass
class RecordingTokenizer:
    """Stub tokenizer that records each call's first positional argument."""

    calls: list[Any] = field(default_factory=list)

    def __call__(
        self,
        text: str,
        return_tensors: str = "pt",
        truncation: bool = True,
        padding: bool = True,
        return_offsets_mapping: bool = True,
    ) -> dict[str, torch.Tensor]:
        self.calls.append(text)
        words = self._alpha_words(text)
        seq_len = len(words) + 2
        input_ids = torch.tensor([[101] + [1000] * len(words) + [102]])
        offset_mapping = torch.zeros((1, seq_len, 2), dtype=torch.long)
        return TokenizerBatch(
            input_ids=input_ids,
            offset_mapping=offset_mapping,
        )

    @staticmethod
    def _alpha_words(text: str) -> list[str]:
        import re

        return re.findall(r"[A-Za-z]+", text) or ["word"]

    def convert_ids_to_tokens(self, input_ids: torch.Tensor) -> list[str]:
        if not self.calls:
            return ["[CLS]", "Ġword", "[SEP]"]
        words = self._alpha_words(str(self.calls[-1]))
        return ["[CLS]"] + [f"Ġ{word}" for word in words] + ["[SEP]"]


class StubModel:
    """Minimal transformer body for RoBERTa-path tests."""

    embedding_dim: int = 32

    def to(self, device: Any) -> StubModel:
        return self

    def eval(self) -> StubModel:
        return self

    def __call__(self, **inputs: Any) -> SimpleNamespace:
        seq_len = inputs["input_ids"].shape[1]
        hidden = torch.randn(1, seq_len, self.embedding_dim)
        return SimpleNamespace(last_hidden_state=hidden)


@dataclass
class TSNECapture:
    """Records TSNE constructor kwargs and fit_transform input."""

    ctor_kwargs: dict[str, Any] = field(default_factory=dict)
    fit_input: np.ndarray | None = None

    def fit_transform(self, matrix: np.ndarray) -> np.ndarray:
        self.fit_input = np.asarray(matrix, dtype=float)
        return np.zeros((self.fit_input.shape[0], 2), dtype=float)


class TSNEFactory:
    """Callable stand-in for sklearn.manifold.TSNE."""

    instances: list[TSNECapture] = field(default_factory=list)

    def __init__(self) -> None:
        self.instances = []

    def __call__(self, *args: Any, **kwargs: Any) -> TSNECapture:
        capture = TSNECapture(ctor_kwargs=dict(kwargs))
        self.instances.append(capture)
        return capture


@pytest.fixture(scope="session")
def vis_module():
    return vis_tool_core


@pytest.fixture(autouse=True)
def inject_vis_globals(
    vis_module,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inject caller-supplied globals expected by vis_tool_core."""
    output_dir = tmp_path_factory.mktemp("output")
    cluster_dir = tmp_path_factory.mktemp("clustering")
    tokenizer = RecordingTokenizer()
    model = StubModel()

    monkeypatch.setattr(vis_module, "WORD_FAMILIES", {}, raising=False)
    monkeypatch.setattr(vis_module, "default_stop_words", _default_stop_words(), raising=False)
    monkeypatch.setattr(vis_module, "TOKENIZER", tokenizer, raising=False)
    monkeypatch.setattr(vis_module, "MODEL", model, raising=False)
    monkeypatch.setattr(vis_module, "lemmatizer", WordNetLemmatizer(), raising=False)
    monkeypatch.setattr(vis_module, "MAX_TOKENS", 10, raising=False)
    monkeypatch.setattr(vis_module, "OUTPUT_DIR", str(output_dir), raising=False)
    monkeypatch.setattr(vis_module, "CLUSTERING_DIR", str(cluster_dir), raising=False)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)


@pytest.fixture
def recording_tokenizer(vis_module) -> RecordingTokenizer:
    tokenizer = RecordingTokenizer()
    vis_module.TOKENIZER = tokenizer
    return tokenizer


@pytest.fixture
def tsne_spy(vis_module, monkeypatch: pytest.MonkeyPatch) -> TSNEFactory:
    factory = TSNEFactory()
    monkeypatch.setattr(vis_module, "TSNE", factory)
    return factory


@pytest.fixture
def train_embedding_spy(vis_module, monkeypatch: pytest.MonkeyPatch):
    """Capture sentences passed to train_embedding; optionally delegate."""

    captured: list[list[str]] = []
    original = vis_module.train_embedding

    def _wrapper(sentences: list[str], *args: Any, **kwargs: Any):
        captured.append(list(sentences))
        if kwargs.pop("_delegate", False):
            return original(sentences, *args, **kwargs)
        words = ["alpha", "beta", "gamma", "delta"]
        embeddings = {word: np.array([float(i), float(i + 1)]) for i, word in enumerate(words)}
        matrix = np.eye(len(words))
        return embeddings, matrix, None

    monkeypatch.setattr(vis_module, "train_embedding", _wrapper)

    @dataclass
    class SpyHandle:
        captured: list[list[str]]

        @property
        def sentences(self) -> list[str]:
            if not self.captured:
                return []
            return self.captured[-1]

        @property
        def all_sentences(self) -> list[str]:
            return [item for batch in self.captured for item in batch]

    handle = SpyHandle(captured=captured)

    def delegate(enable: bool = True) -> None:
        def _delegating(sentences: list[str], *args: Any, **kwargs: Any):
            captured.append(list(sentences))
            kwargs["_delegate"] = enable
            return _wrapper(sentences, *args, **kwargs)

        monkeypatch.setattr(vis_module, "train_embedding", _delegating)

    handle.enable_delegate = delegate  # type: ignore[attr-defined]
    return handle
