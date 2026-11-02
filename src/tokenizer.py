"""A byte-level tokenizer written from scratch, using only ``vocab.json``.

The project never calls the SDK ``encode``/``decode`` helpers. Everything is
rebuilt here from the vocabulary file, which maps each token (written in the
GPT-2 "byte-level" alphabet) to its integer id:

* every one of the 256 byte values is shown as one printable character, for
  instance the space byte is drawn as ``Ġ`` and the newline byte as ``Ċ``;
* a token is therefore just a sequence of bytes, and a character that needs
  several UTF-8 bytes may be split across tokens.

Encoding splits the text into words, then cuts each word by greedy longest
match over the vocabulary. Decoding concatenates the bytes of the tokens.
"""

import json
import unicodedata
from typing import Any, Dict, List

from pydantic import BaseModel, PrivateAttr

Trie = Dict[int, Any]
"""Byte trie: ``node[byte]`` is the child node, ``node[-1]`` a token id."""

TOKEN_ID_KEY = -1

MAX_CACHED_WORDS = 50000

_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")

_LETTER, _NUMBER, _NEWLINE, _SPACE, _OTHER = range(5)


def _char_class(char: str) -> int:
    """Classify a character the way the tokenizer's split rule does.

    Args:
        char: A single character.

    Returns:
        One of the ``_LETTER`` ... ``_OTHER`` constants.
    """
    if char in "\r\n":
        return _NEWLINE
    if char.isspace():
        return _SPACE
    category = unicodedata.category(char)
    if category[0] == "L":
        return _LETTER
    if category[0] == "N":
        return _NUMBER
    return _OTHER


def split_words(text: str) -> List[str]:
    r"""Split text into the words that are tokenized one by one.

    This reproduces, without a regex engine that knows Unicode classes, the
    split rule declared in the model's ``tokenizer.json``::

        (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}
        | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+

    In words: contractions, a run of letters with at most one leading
    symbol or space, single digits, runs of punctuation, and whitespace.
    Joining the returned chunks always gives back the original text.

    Args:
        text: Text to split.

    Returns:
        The chunks, in order.
    """
    classes = [_char_class(char) for char in text]
    size = len(text)
    words: List[str] = []
    start = 0
    while start < size:
        end = _word_end(text, classes, start)
        words.append(text[start:end])
        start = end
    return words


def _word_end(text: str, classes: List[int], start: int) -> int:
    """Find where the chunk beginning at ``start`` stops.

    Args:
        text: Text being split.
        classes: Character class of every character of ``text``.
        start: Index of the first character of the chunk.

    Returns:
        Index just past the last character of the chunk.
    """
    size = len(text)
    kind = classes[start]
    if text[start] == "'":
        lowered = text[start:start + 3].lower()
        for contraction in _CONTRACTIONS:
            if lowered.startswith(contraction):
                return start + len(contraction)
    # An optional non-letter character followed by a run of letters.
    letters = start
    if (
        kind in (_SPACE, _OTHER)
        and start + 1 < size
        and classes[start + 1] == _LETTER
    ):
        letters = start + 1
    if classes[letters] == _LETTER:
        end = letters
        while end < size and classes[end] == _LETTER:
            end += 1
        return end
    if kind == _NUMBER:
        return start + 1
    # An optional space, a run of symbols, then any trailing newlines.
    symbols = start
    if text[start] == " " and start + 1 < size:
        symbols = start + 1
    if classes[symbols] == _OTHER:
        end = symbols
        while end < size and classes[end] == _OTHER:
            end += 1
        while end < size and classes[end] == _NEWLINE:
            end += 1
        return end
    # Only whitespace is left: find the whole run.
    end = start
    last_newline = -1
    while end < size and classes[end] in (_SPACE, _NEWLINE):
        if classes[end] == _NEWLINE:
            last_newline = end
        end += 1
    if last_newline >= 0:
        return last_newline + 1
    if end < size and end - start > 1:
        return end - 1  # leave one space to be glued to the next word
    return end


def build_byte_alphabet() -> Dict[str, int]:
    """Build the character to byte table of byte-level vocabularies.

    Printable bytes keep their own character; the others (control bytes,
    space, ...) are shifted to code points starting at 256 so that every
    byte has a visible, unambiguous symbol.

    Returns:
        Mapping from the vocabulary character to the byte it represents.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(0xA1, 0xAD))
        + list(range(0xAE, 0x100))
    )
    table: Dict[str, int] = {chr(byte): byte for byte in printable}
    shift = 0
    for byte in range(256):
        if byte not in printable:
            table[chr(256 + shift)] = byte
            shift += 1
    return table


def build_trie(tokens: Dict[int, bytes]) -> Trie:
    """Store tokens in a byte trie.

    Args:
        tokens: Mapping from token id to the bytes of the token.

    Returns:
        The trie: following the bytes of a token from the root leads to a
        node holding its id under ``TOKEN_ID_KEY``.
    """
    root: Trie = {}
    for token_id, raw in tokens.items():
        node = root
        for byte in raw:
            node = node.setdefault(byte, {})
        node[TOKEN_ID_KEY] = token_id
    return root


class Tokenizer(BaseModel):
    """Byte-level tokenizer built from the model's vocabulary file.

    Attributes:
        vocab: Mapping from the token text (byte-level alphabet) to its id.
    """

    vocab: Dict[str, int]

    _token_bytes: Dict[int, bytes] = PrivateAttr(default_factory=dict)
    _trie: Trie = PrivateAttr(default_factory=dict)
    _word_cache: Dict[str, List[int]] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context: Any) -> None:
        """Precompute the lookup tables used by encode and decode.

        Args:
            context: Unused pydantic validation context.
        """
        alphabet = build_byte_alphabet()
        for token, token_id in self.vocab.items():
            try:
                self._token_bytes[token_id] = bytes(
                    alphabet[char] for char in token
                )
            except KeyError:
                continue  # not a byte-level token: cannot be generated
        self._trie = build_trie(self._token_bytes)

    @classmethod
    def from_file(cls, vocab_path: str) -> "Tokenizer":
        """Create a tokenizer from a ``vocab.json`` file.

        Args:
            vocab_path: Path to the vocabulary file.

        Returns:
            A ready to use tokenizer.

        Raises:
            OSError: If the vocabulary file cannot be read.
            ValueError: If the vocabulary file has an unexpected content.
        """
        try:
            with open(vocab_path, "r", encoding="utf-8") as handle:
                vocab = json.load(handle)
            if not isinstance(vocab, dict) or not vocab:
                raise ValueError("a non-empty JSON object is expected")
            return cls(vocab=vocab)
        except ValueError as error:  # includes JSON and pydantic errors
            reason = str(error).splitlines()[0]
            raise ValueError(
                f"unusable vocabulary file {vocab_path}: {reason}"
            ) from None

    @property
    def token_bytes(self) -> Dict[int, bytes]:
        """Mapping from token id to the raw bytes the token stands for."""
        return self._token_bytes

    @property
    def trie(self) -> Trie:
        """Byte trie holding every token of the vocabulary."""
        return self._trie

    def encode(self, text: str) -> List[int]:
        """Convert text into token ids.

        Args:
            text: Text to tokenize.

        Returns:
            The token ids.

        Raises:
            ValueError: If a byte is missing from the vocabulary.
        """
        ids: List[int] = []
        for word in split_words(unicodedata.normalize("NFC", text)):
            cached = self._word_cache.get(word)
            if cached is None:
                cached = self.encode_bytes(word.encode("utf-8"))
                if len(self._word_cache) < MAX_CACHED_WORDS:
                    self._word_cache[word] = cached
            ids.extend(cached)
        return ids

    def encode_bytes(self, raw: bytes) -> List[int]:
        """Tokenize raw bytes by always taking the longest known token.

        Args:
            raw: Bytes to tokenize.

        Returns:
            The token ids.

        Raises:
            ValueError: If a byte is missing from the vocabulary.
        """
        ids: List[int] = []
        start = 0
        while start < len(raw):
            node = self._trie
            best_id = -1
            best_end = start
            for position in range(start, len(raw)):
                child = node.get(raw[position])
                if child is None:
                    break
                node = child
                if TOKEN_ID_KEY in node:
                    best_id = node[TOKEN_ID_KEY]
                    best_end = position + 1
            if best_id < 0:
                raise ValueError(
                    f"byte {raw[start]:#04x} is not in the vocabulary"
                )
            ids.append(best_id)
            start = best_end
        return ids

    def decode_bytes(self, ids: List[int]) -> bytes:
        """Concatenate the raw bytes of a token sequence.

        Args:
            ids: Token ids; unknown ids are skipped.

        Returns:
            The bytes spelled by the tokens.
        """
        return b"".join(self._token_bytes.get(token, b"") for token in ids)

    def decode(self, ids: List[int]) -> str:
        """Convert token ids back into text.

        Args:
            ids: Token ids; unknown ids are skipped.

        Returns:
            The decoded text. Invalid UTF-8 is shown as ``U+FFFD``.
        """
        return self.decode_bytes(ids).decode("utf-8", errors="replace")
