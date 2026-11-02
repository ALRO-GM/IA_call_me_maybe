"""Shared fixtures: a tiny tokenizer and a fake language model."""

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pytest

from src.decoder import ConstrainedDecoder, TokenIndex
from src.models import FunctionDefinition
from src.tokenizer import Tokenizer, build_byte_alphabet

EXTRA_TOKENS = [
    '{"', '":', ' "', '",', '}}', '"}', '"}}', "fn", "_add", "_numbers",
    "_greet", "hello", " world", "12", "345", " -", ", ", "[]", "true",
    "false", "\\\\", '\\"', "name", "parameters", " {", "é", "你", "ab",
    "abc", " a", "\n", "\t", '"a', "e+", ".5", " [", "],", '":"', "null",
]

SAMPLE_FUNCTIONS: List[Dict[str, Any]] = [
    {
        "name": "fn_add_numbers",
        "description": "Add two numbers.",
        "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
        "returns": {"type": "number"},
    },
    {
        "name": "fn_greet",
        "description": "Greet somebody.",
        "parameters": {"name": {"type": "string"}},
        "returns": {"type": "string"},
    },
    {
        "name": "fn_add",
        "description": "A name that is a prefix of another one.",
        "parameters": {"flag": {"type": "boolean"}},
    },
    {
        "name": "fn_complex",
        "description": "Nested arguments.",
        "parameters": {
            "count": {"type": "integer"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "point": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "label": {"type": "string", "enum": ["low", "high"]},
                },
            },
            "nothing": {"type": "null"},
        },
    },
    {"name": "fn_no_args", "description": "No arguments.", "parameters": {}},
]


def to_symbols(raw: bytes) -> str:
    """Spell bytes in the byte-level alphabet used by vocabulary files."""
    symbols = {byte: char for char, byte in build_byte_alphabet().items()}
    return "".join(symbols[byte] for byte in raw)


def make_vocab() -> Dict[str, int]:
    """Build a small vocabulary: every byte plus a few longer tokens."""
    vocab = {to_symbols(bytes([byte])): byte for byte in range(256)}
    for text in EXTRA_TOKENS:
        vocab.setdefault(to_symbols(text.encode("utf-8")), len(vocab))
    return vocab


@pytest.fixture(scope="session")
def tokenizer() -> Tokenizer:
    """A tokenizer over the tiny vocabulary, without merge rules."""
    return Tokenizer(vocab=make_vocab())


@pytest.fixture(scope="session")
def index(tokenizer: Tokenizer) -> TokenIndex:
    """Token index of the tiny tokenizer."""
    return TokenIndex(tokenizer=tokenizer)


@pytest.fixture(scope="session")
def functions() -> List[FunctionDefinition]:
    """A set of functions covering every supported type."""
    return [FunctionDefinition.model_validate(f) for f in SAMPLE_FUNCTIONS]


@pytest.fixture()
def random_model(tokenizer: Tokenizer) -> Callable[[int], Any]:
    """Factory of fake models that answer with random logits."""

    def factory(seed: int) -> Callable[[List[int]], List[float]]:
        generator = np.random.default_rng(seed)
        size = len(tokenizer.vocab)

        def logits(ids: List[int]) -> List[float]:
            scores: List[float] = generator.normal(size=size).tolist()
            return scores

        return logits

    return factory


@pytest.fixture()
def make_decoder(
    tokenizer: Tokenizer, index: TokenIndex
) -> Callable[..., ConstrainedDecoder]:
    """Factory of decoders bound to the tiny tokenizer."""

    def factory(
        logits: Callable[[List[int]], List[float]], **options: Any
    ) -> ConstrainedDecoder:
        return ConstrainedDecoder(
            tokenizer=tokenizer,
            index=index,
            logits_function=logits,
            **options,
        )

    return factory


@pytest.fixture()
def write_json(tmp_path: Path) -> Callable[[str, Any], str]:
    """Write a JSON file in a temporary directory and return its path."""

    def writer(name: str, content: Any) -> str:
        path = tmp_path / name
        text = content if isinstance(content, str) else json.dumps(content)
        path.write_text(text, encoding="utf-8")
        return str(path)

    return writer


def pytest_collection_modifyitems(
    config: pytest.Config, items: List[pytest.Item]
) -> None:
    """Skip tests needing the real model unless RUN_SLOW=1 is set."""
    if os.environ.get("RUN_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="set RUN_SLOW=1 to run with the model")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def isolated_thread_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo, after each test, the thread count a run may have set."""
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
