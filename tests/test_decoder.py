"""Tests of the token index and of constrained decoding."""

import json
from typing import Any, Callable, List

import numpy as np
import pytest

from src.decoder import ConstrainedDecoder, LogitsCache, TokenIndex
from src.grammar import arguments_grammar, function_name_grammar
from src.models import FunctionDefinition
from src.tokenizer import Tokenizer

DecoderFactory = Callable[..., ConstrainedDecoder]


def texts(tokenizer: Tokenizer, ids: Any) -> List[str]:
    """Decode every id of an array on its own."""
    return sorted(tokenizer.decode([int(token)]) for token in ids)


def test_valid_tokens_for_a_number(
    tokenizer: Tokenizer,
    index: TokenIndex,
    functions: List[FunctionDefinition],
) -> None:
    grammar = arguments_grammar(functions[0])
    state = grammar.run(grammar.initial_state(), b', "parameters": {"a": ')
    assert state is not None
    allowed = texts(tokenizer, index.valid_ids(grammar, state))
    assert allowed == sorted(list("-0123456789") + ["12", "345"])
    after_zero = grammar.advance(state, ord("0"))
    assert after_zero is not None
    allowed = texts(tokenizer, index.valid_ids(grammar, after_zero))
    assert allowed == sorted([".", "e", "E", ",", ", ", ".5", "e+"])


def test_valid_tokens_for_a_name(
    tokenizer: Tokenizer,
    index: TokenIndex,
    functions: List[FunctionDefinition],
) -> None:
    grammar = function_name_grammar([f.name for f in functions])
    state = grammar.run(grammar.initial_state(), b'{"name": "fn')
    assert state is not None
    allowed = texts(tokenizer, index.valid_ids(grammar, state))
    assert allowed == sorted(["_", "_add", "_greet"])


def test_valid_tokens_inside_a_string(
    tokenizer: Tokenizer,
    index: TokenIndex,
    functions: List[FunctionDefinition],
) -> None:
    grammar = arguments_grammar(functions[1])
    start = grammar.run(
        grammar.initial_state(), b', "parameters": {"name": "'
    )
    assert start is not None
    first = set(texts(tokenizer, index.valid_ids(grammar, start)))
    assert "hello" in first and '"}}' in first and "\\\\" in first
    assert " world" not in first and " " not in first  # no leading space
    assert "\n" not in first and "\t" not in first      # no control char
    inside = grammar.advance(start, ord("x"))
    assert inside is not None
    later = set(texts(tokenizer, index.valid_ids(grammar, inside)))
    assert " world" in later and '"}}' in later and '"}' in later
    assert '",' not in later   # nothing follows this last argument
    assert '"a' not in later   # would close the string then break the JSON
    assert index.valid_ids(grammar, inside) is index.valid_ids(
        grammar, inside
    )  # cached


@pytest.mark.parametrize("seed", range(40))
def test_random_logits_always_give_valid_json(
    seed: int,
    functions: List[FunctionDefinition],
    random_model: Callable[[int], Any],
    make_decoder: DecoderFactory,
) -> None:
    """Validity must come from the mask, not from the model."""
    decoder = make_decoder(random_model(seed), max_string_tokens=6)
    names = function_name_grammar([f.name for f in functions])
    chosen = decoder.generate([1, 2, 3], names)
    assert names.matches(chosen.data)
    name = json.loads(chosen.data + b"}")["name"]
    function = next(f for f in functions if f.name == name)
    grammar = arguments_grammar(function)
    arguments = decoder.generate([1, 2, 3] + chosen.token_ids, grammar)
    assert grammar.matches(arguments.data)
    parsed = json.loads(chosen.data + arguments.data)
    assert list(parsed) == ["name", "parameters"]
    assert list(parsed["parameters"]) == list(function.parameters)
    assert (
        decoder.tokenizer.decode_bytes(arguments.token_ids) == arguments.data
    )


def test_mask_overrides_the_model(
    tokenizer: Tokenizer,
    functions: List[FunctionDefinition],
    make_decoder: DecoderFactory,
) -> None:
    """A model that always wants a newline still produces valid JSON."""
    newline = tokenizer.encode_bytes(b"\n")[0]
    size = len(tokenizer.vocab)

    def stubborn(ids: List[int]) -> List[float]:
        scores = np.zeros(size)
        scores[newline] = 100.0
        result: List[float] = scores.tolist()
        return result

    decoder = make_decoder(stubborn, trace=True, max_string_tokens=3)
    grammar = arguments_grammar(functions[0])
    generation = decoder.generate([], grammar)
    assert grammar.matches(generation.data)
    chosen = [step for step in generation.steps if not step.forced]
    assert chosen and all(not step.unconstrained_valid for step in chosen)
    assert all(step.unconstrained == "\n" for step in chosen)


def test_forced_text_needs_no_model_call(
    functions: List[FunctionDefinition], make_decoder: DecoderFactory
) -> None:
    def forbidden(ids: List[int]) -> List[float]:
        raise AssertionError("the model must not be called")

    decoder = make_decoder(forbidden)
    grammar = arguments_grammar(functions[4])  # no arguments at all
    generation = decoder.generate([], grammar)
    assert generation.data == b', "parameters": {}}'
    assert generation.model_calls == 0
    only = function_name_grammar(["solo"])
    assert decoder.generate([], only).data == b'{"name": "solo"'


def test_long_strings_are_closed(
    tokenizer: Tokenizer,
    functions: List[FunctionDefinition],
    make_decoder: DecoderFactory,
) -> None:
    letter = tokenizer.encode_bytes(b"z")[0]
    size = len(tokenizer.vocab)

    def endless(ids: List[int]) -> List[float]:
        scores = np.zeros(size)
        scores[letter] = 50.0
        result: List[float] = scores.tolist()
        return result

    decoder = make_decoder(endless, max_string_tokens=5)
    generation = decoder.generate([], arguments_grammar(functions[1]))
    assert json.loads(b'{"name": "x"' + generation.data)["parameters"] == {
        "name": "zzzzz"
    }


def test_token_budget_is_enforced(
    functions: List[FunctionDefinition],
    random_model: Callable[[int], Any],
    make_decoder: DecoderFactory,
) -> None:
    decoder = make_decoder(random_model(0), max_new_tokens=3)
    with pytest.raises(RuntimeError, match="stopped after 3 tokens"):
        decoder.generate([], arguments_grammar(functions[3]))


def test_model_with_too_few_logits(
    functions: List[FunctionDefinition], make_decoder: DecoderFactory
) -> None:
    decoder = make_decoder(lambda ids: [0.0])
    with pytest.raises(RuntimeError, match="none of the allowed tokens"):
        decoder.generate([], arguments_grammar(functions[0]))


def test_trace_records_every_step(
    functions: List[FunctionDefinition],
    random_model: Callable[[int], Any],
    make_decoder: DecoderFactory,
) -> None:
    decoder = make_decoder(random_model(3), trace=True, max_string_tokens=4)
    generation = decoder.generate([], arguments_grammar(functions[1]))
    assert "".join(step.text for step in generation.steps).encode() == (
        generation.data
    )
    assert generation.steps[0].forced
    free = [step for step in generation.steps if not step.forced]
    assert len(free) == generation.model_calls
    for step in free:
        assert step.allowed > 1 and step.alternatives
        total = sum(share for _, share in step.alternatives)
        assert 0.0 < total <= 1.0 + 1e-9


def test_logits_cache_answers_repeated_contexts_from_memory() -> None:
    seen: List[List[int]] = []

    def model(context: List[int]) -> List[float]:
        seen.append(list(context))
        return [float(len(context)), 0.5]

    cache = LogitsCache(logits_function=model, max_entries=2)
    first = cache([1, 2])
    assert np.array_equal(cache([1, 2]), first)
    assert (cache.misses, cache.hits) == (1, 1)
    cache([3])
    cache([4])  # evicts [1, 2], the least recently used context
    cache([1, 2])
    assert cache.misses == 4 and len(seen) == 4


def test_logits_cache_can_be_disabled() -> None:
    cache = LogitsCache(logits_function=lambda ids: [1.0], max_entries=0)
    cache([1])
    cache([1])
    assert (cache.misses, cache.hits) == (2, 0)
