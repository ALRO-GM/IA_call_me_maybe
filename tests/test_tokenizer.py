"""Tests of the home-made BPE tokenizer."""

import json
from pathlib import Path
from typing import Any, List

import pytest

from src.tokenizer import (
    TOKEN_ID_KEY,
    Tokenizer,
    build_byte_alphabet,
    build_trie,
    split_words,
)
from tests.conftest import make_vocab, to_symbols

TEXTS = [
    "", " ", "hello world", '{"name": "fn_add", "parameters": {"a": 12}}',
    "café ñandú 你好 🙂", "tabs\tand\nnewlines\r\n", "It's 3.5 * -2!",
    "  leading and trailing  ", "é", "\\d+ [a-z] \"quoted\"",
]


def test_alphabet_covers_every_byte() -> None:
    alphabet = build_byte_alphabet()
    assert sorted(alphabet.values()) == list(range(256))
    assert alphabet["Ġ"] == ord(" ") and alphabet["Ċ"] == ord("\n")
    assert alphabet["a"] == ord("a")


@pytest.mark.parametrize("text", TEXTS)
def test_split_words_loses_nothing(text: str) -> None:
    assert "".join(split_words(text)) == text


@pytest.mark.parametrize(
    "text, expected",
    [
        ("hello world", ["hello", " world"]),
        ("a1b22", ["a", "1", "b", "2", "2"]),
        ("don't", ["don", "'t"]),
        ("x  y", ["x", " ", " y"]),
        ("a\n\n b", ["a", "\n\n", " b"]),
        ("wait...  ok", ["wait", "...", " ", " ok"]),
        ("a ½", ["a", " ", "½"]),
        ("end  ", ["end", "  "]),
        ("fn_add", ["fn", "_add"]),
    ],
)
def test_split_words_rules(text: str, expected: List[str]) -> None:
    assert split_words(text) == expected


@pytest.mark.parametrize("text", TEXTS)
def test_round_trip(tokenizer: Tokenizer, text: str) -> None:
    import unicodedata

    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == unicodedata.normalize("NFC", text)


def test_longest_match_prefers_long_tokens(tokenizer: Tokenizer) -> None:
    ids = tokenizer.encode_bytes(b'{"name": "fn_add"}}')
    pieces = [tokenizer.decode([token]) for token in ids]
    assert pieces == ['{"', "name", '":', ' "', "fn", "_add", '"}}']


def test_encode_bytes_accepts_cut_utf8(tokenizer: Tokenizer) -> None:
    half = "é".encode("utf-8")[:1]
    assert tokenizer.decode_bytes(tokenizer.encode_bytes(half)) == half


def test_words_are_cut_independently(tokenizer: Tokenizer) -> None:
    """A token never spans two words, even if the vocabulary has one."""
    pieces = [tokenizer.decode([i]) for i in tokenizer.encode("hello world")]
    assert pieces == ["hello", " world"]
    pieces = [tokenizer.decode([i]) for i in tokenizer.encode('a", "b')]
    assert '",' in pieces and ' "' in pieces


def test_tokens_outside_the_byte_alphabet_are_ignored() -> None:
    vocab = make_vocab()
    vocab["\u2581word"] = len(vocab)  # SentencePiece style, not byte-level
    tokenizer = Tokenizer(vocab=vocab)
    assert vocab["\u2581word"] not in tokenizer.token_bytes
    assert tokenizer.decode([vocab["\u2581word"]]) == ""


def test_unknown_byte_is_reported() -> None:
    vocab = {to_symbols(b"a"): 0}
    with pytest.raises(ValueError, match="not in the vocabulary"):
        Tokenizer(vocab=vocab).encode("ab")


def test_trie_holds_every_token(tokenizer: Tokenizer) -> None:
    assert build_trie({7: b"ab", 8: b"a"}) == {
        ord("a"): {TOKEN_ID_KEY: 8, ord("b"): {TOKEN_ID_KEY: 7}}
    }
    for token_id, raw in tokenizer.token_bytes.items():
        node = tokenizer.trie
        for byte in raw:
            node = node[byte]
        assert node[TOKEN_ID_KEY] == token_id


def test_from_file(tmp_path: Path) -> None:
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(make_vocab()), encoding="utf-8")
    tokenizer = Tokenizer.from_file(str(path))
    assert tokenizer.decode(tokenizer.encode("hello")) == "hello"


@pytest.mark.parametrize("content", ["[1, 2]", "{}", "{oops", '{"a": "b"}'])
def test_from_file_rejects_bad_vocab(tmp_path: Path, content: str) -> None:
    path = tmp_path / "vocab.json"
    path.write_text(content)
    with pytest.raises(ValueError, match="unusable vocabulary file"):
        Tokenizer.from_file(str(path))


def test_from_file_missing_vocab(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        Tokenizer.from_file(str(tmp_path / "missing.json"))


@pytest.mark.slow
def test_real_vocabulary_round_trip() -> None:
    """With the model's own vocabulary, decode(encode(x)) gives x back."""
    import random
    import unicodedata

    from llm_sdk import Small_LLM_Model

    model: Any = Small_LLM_Model()
    tokenizer = Tokenizer.from_file(model.get_path_to_vocab_file())
    assert len(tokenizer.token_bytes) > 100000
    generator = random.Random(42)
    alphabet = "abcXYZ  \n\r\t'_-.,!?\"\\/(){}[]:;0123456789éñ你🙂½² dsmtrevl"
    samples = list(TEXTS)
    for _ in range(1500):
        length = generator.randint(1, 20)
        samples.append(
            "".join(generator.choice(alphabet) for _ in range(length))
        )
    for text in samples:
        expected = unicodedata.normalize("NFC", text)
        assert tokenizer.decode(tokenizer.encode(text)) == expected, text
