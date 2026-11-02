"""Constrained decoding: only tokens the grammar accepts can be chosen.

At every step the model gives a score (logit) to each token of its
vocabulary. Before choosing, every token that the grammar would reject gets
the score ``-inf``, so the best remaining token is always a valid one. The
output is therefore correct by construction, whatever the model "wants".
"""

import re
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from src.grammar import (
    NOT_TEXT,
    TEXT_INSIDE,
    TEXT_START,
    Grammar,
    State,
)
from src.tokenizer import TOKEN_ID_KEY, Tokenizer, Trie, build_trie

IdArray = npt.NDArray[np.int64]
LogitArray = npt.NDArray[np.float32]
LogitsFunction = Callable[[List[int]], npt.ArrayLike]

_NOT_PLAIN = re.compile(rb'["\\\x00-\x1f]')
_SPACE = 0x20
_ALTERNATIVES_SHOWN = 4


class LogitsCache(BaseModel):
    """Remembers the logits of the last contexts sent to the model.

    Decoding is greedy, so the same context always leads to the same
    logits. A retry with stricter limits replays the first attempt up to
    the point where the limits differ, and a prompt repeated in the input
    file replays a whole generation: those forward passes are served from
    memory. Logits are kept as ``float32``, the precision the model
    computes them in, so the cache changes no result.

    Attributes:
        logits_function: The real model call.
        max_entries: Contexts kept, least recently used dropped first.
        hits: Requests answered from memory.
        misses: Requests that ran the model.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    logits_function: LogitsFunction
    max_entries: int = Field(default=128, ge=0)
    hits: int = 0
    misses: int = 0

    _entries: "OrderedDict[Tuple[int, ...], LogitArray]" = PrivateAttr(
        default_factory=OrderedDict
    )

    def __call__(self, context: List[int]) -> LogitArray:
        """Return the next-token logits of a context.

        Args:
            context: Token ids seen so far.

        Returns:
            The logits of the whole vocabulary.
        """
        key = tuple(context)
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
            self.hits += 1
            return cached
        logits = np.asarray(self.logits_function(context), dtype=np.float32)
        self.misses += 1
        if self.max_entries:
            self._entries[key] = logits
            if len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return logits


class TokenIndex(BaseModel):
    """Answers "which tokens may come next?" for any grammar state.

    A token is allowed when the grammar can consume all of its bytes. The
    vocabulary is stored as a byte trie that is walked together with the
    grammar: as soon as a byte is refused, the whole branch below it (every
    token sharing that prefix) is discarded at once.

    The only states where almost every token is allowed are the inside of
    a string. Those sets are computed once at start-up; only the few tokens
    holding a quote, a backslash or a cut UTF-8 character need the walk.

    Attributes:
        tokenizer: Source of the vocabulary.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokenizer: Tokenizer

    _plain: Dict[int, IdArray] = PrivateAttr(default_factory=dict)
    _special_trie: Trie = PrivateAttr(default_factory=dict)
    _cache: Dict[int, Tuple[Grammar, Dict[State, IdArray]]] = PrivateAttr(
        default_factory=dict
    )

    def model_post_init(self, context: Any) -> None:
        """Split the vocabulary into "plain text" tokens and the others.

        A plain token is complete UTF-8 text without quote, backslash or
        control character: inside a string it is always valid and leaves
        the string in the same state.

        Args:
            context: Unused pydantic validation context.
        """
        plain: List[int] = []
        special: Dict[int, bytes] = {}
        for token_id, raw in self.tokenizer.token_bytes.items():
            if not raw:
                continue
            try:
                raw.decode("utf-8")
                is_plain = _NOT_PLAIN.search(raw) is None
            except UnicodeDecodeError:
                is_plain = False
            if is_plain:
                plain.append(token_id)
            else:
                special[token_id] = raw
        token_bytes = self.tokenizer.token_bytes
        self._plain = {
            TEXT_INSIDE: np.array(sorted(plain), dtype=np.int64),
            TEXT_START: np.array(
                sorted(t for t in plain if token_bytes[t][0] != _SPACE),
                dtype=np.int64,
            ),
        }
        self._special_trie = build_trie(special)

    def _walk(self, grammar: Grammar, state: State, trie: Trie) -> List[int]:
        """Collect the tokens of a trie that the grammar accepts.

        Args:
            grammar: Grammar being followed.
            state: Current grammar state.
            trie: Byte trie of candidate tokens.

        Returns:
            Ids of the accepted tokens.
        """
        found: List[int] = []
        pending: List[Tuple[Trie, State]] = [(trie, state)]
        while pending:
            node, current = pending.pop()
            for byte, child in node.items():
                if byte == TOKEN_ID_KEY:
                    continue
                following = grammar.advance(current, byte)
                if following is None:
                    continue
                if TOKEN_ID_KEY in child:
                    found.append(child[TOKEN_ID_KEY])
                if following:
                    pending.append((child, following))
        return found

    def valid_ids(self, grammar: Grammar, state: State) -> IdArray:
        """Return the ids of every token allowed in a state.

        Results are cached per grammar and state, so the same position
        reached for another prompt costs nothing.

        Args:
            grammar: Grammar being followed.
            state: Current grammar state.

        Returns:
            Sorted array of allowed token ids (possibly empty).
        """
        # Keeping the grammar next to its cache pins its id().
        _, cache = self._cache.setdefault(id(grammar), (grammar, {}))
        cached = cache.get(state)
        if cached is not None:
            return cached
        plain = self._plain.get(grammar.text_mode(state))
        if plain is None:
            walked = self._walk(grammar, state, self.tokenizer.trie)
            ids = np.array(walked, dtype=np.int64)
        else:
            walked = self._walk(grammar, state, self._special_trie)
            ids = np.concatenate((plain, np.array(walked, dtype=np.int64)))
        ids.sort()
        cache[state] = ids
        return ids


class DecodingStep(BaseModel):
    """What happened at one step of a generation (used for display).

    Attributes:
        token_ids: Tokens added at this step.
        forced: True when the grammar left no choice (model not asked).
        text: Text added to the output at this step.
        allowed: Number of tokens the grammar accepted.
        alternatives: Best allowed tokens with their probability.
        unconstrained: Token the model would have picked without mask.
        unconstrained_valid: Whether that free choice was allowed.
    """

    token_ids: List[int]
    forced: bool
    text: str = ""
    allowed: int = 1
    alternatives: List[Tuple[str, float]] = []
    unconstrained: str = ""
    unconstrained_valid: bool = True


class Generation(BaseModel):
    """Result of one constrained generation.

    Attributes:
        data: Bytes produced, guaranteed to match the grammar.
        token_ids: Tokens produced, to extend the model context.
        steps: Step by step trace (empty unless tracing is enabled).
        model_calls: Number of forward passes that were needed.
    """

    data: bytes
    token_ids: List[int]
    steps: List[DecodingStep] = []
    model_calls: int = 0


class ConstrainedDecoder(BaseModel):
    """Generates text token by token under the control of a grammar.

    Attributes:
        tokenizer: Tokenizer giving the bytes of every token.
        index: Finder of the tokens allowed in a grammar state.
        logits_function: Returns next-token logits for a list of ids.
        max_new_tokens: Hard limit of generated tokens per call.
        max_string_tokens: Tokens after which a string is closed.
        trace: Record a ``DecodingStep`` for every step.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokenizer: Tokenizer
    index: TokenIndex
    logits_function: LogitsFunction
    max_new_tokens: int = Field(default=400, ge=1)
    max_string_tokens: int = Field(default=96, ge=1)
    trace: bool = False

    def _choose(self, context: List[int], allowed: IdArray) -> DecodingStep:
        """Pick the next token among the ones the grammar allows.

        Args:
            context: Every token id seen so far (prompt and output).
            allowed: Ids of the tokens accepted by the grammar.

        Returns:
            The step, holding the chosen token.

        Raises:
            RuntimeError: If no allowed token can be chosen.
        """
        if allowed.size == 0:
            raise RuntimeError("no token can continue the JSON")
        if allowed.size == 1:  # no real choice: the model is not asked
            return DecodingStep(token_ids=[int(allowed[0])], forced=True)
        logits = np.asarray(self.logits_function(context), dtype=np.float64)
        allowed = allowed[allowed < logits.size]
        if allowed.size == 0:
            raise RuntimeError("the model knows none of the allowed tokens")
        # The heart of constrained decoding: forbidden tokens get -inf.
        masked = np.full(logits.shape, -np.inf)
        masked[allowed] = logits[allowed]
        step = DecodingStep(
            token_ids=[int(np.argmax(masked))],
            forced=False,
            allowed=int(allowed.size),
        )
        if self.trace:
            self._describe(step, logits, masked[allowed], allowed)
        return step

    def _describe(
        self,
        step: DecodingStep,
        logits: npt.NDArray[np.float64],
        scores: npt.NDArray[np.float64],
        allowed: IdArray,
    ) -> None:
        """Add to a step what the visualizer shows about the choice.

        Args:
            step: Step to complete.
            logits: Raw logits of the whole vocabulary.
            scores: Logits of the allowed tokens only.
            allowed: Ids of the allowed tokens.
        """
        finite = scores[np.isfinite(scores)]
        peak = float(np.max(finite)) if finite.size else 0.0
        weights = np.exp(np.where(np.isfinite(scores), scores - peak, -745.0))
        weights = weights / weights.sum()
        best = np.argsort(-weights)[:_ALTERNATIVES_SHOWN]
        step.alternatives = [
            (self.tokenizer.decode([int(allowed[i])]), float(weights[i]))
            for i in best
        ]
        free = int(np.argmax(logits))
        step.unconstrained = self.tokenizer.decode([free])
        step.unconstrained_valid = bool(np.isin(free, allowed))

    def generate(self, context: List[int], grammar: Grammar) -> Generation:
        """Generate an output matching a grammar.

        The grammar must end with fixed text (as the grammars of this
        project do) so that the end of the output is unambiguous.

        Args:
            context: Token ids of the prompt (and of earlier output).
            grammar: Grammar the output has to match.

        Returns:
            The generated bytes, their token ids and the trace.

        Raises:
            RuntimeError: If generation cannot be completed.
        """
        state = grammar.initial_state()
        result = Generation(data=b"", token_ids=[])
        string_tokens = 0
        while state:
            if len(result.token_ids) >= self.max_new_tokens:
                raise RuntimeError(
                    f"generation stopped after {self.max_new_tokens} "
                    "tokens without reaching the end of the JSON"
                )
            forced = grammar.forced_bytes(state)
            closable = grammar.text_mode(state) in (TEXT_START, TEXT_INSIDE)
            if closable and string_tokens >= self.max_string_tokens:
                forced = b'"'  # too long: close the string
            if forced:
                step = DecodingStep(
                    token_ids=self.tokenizer.encode(forced.decode("ascii")),
                    forced=True,
                )
            else:
                step = self._choose(
                    context + result.token_ids,
                    self.index.valid_ids(grammar, state),
                )
                result.model_calls += not step.forced
            chunk = self.tokenizer.decode_bytes(step.token_ids)
            following = grammar.run(state, chunk)
            if following is None:
                raise RuntimeError("internal error: invalid token accepted")
            state = following
            result.data += chunk
            result.token_ids.extend(step.token_ids)
            counted = grammar.text_mode(state) != NOT_TEXT and not forced
            string_tokens = string_tokens + 1 if counted else 0
            if self.trace:
                step.text = chunk.decode("utf-8", errors="replace")
                result.steps.append(step)
        return result
