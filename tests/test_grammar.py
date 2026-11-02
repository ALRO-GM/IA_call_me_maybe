"""Tests of the byte-level automata."""

import json
from typing import Any, Dict, List

import pytest

from src.grammar import (
    DONE,
    MAX_ARRAY_ITEMS,
    MAX_NUMBER_CHARS,
    NOT_TEXT,
    TEXT_INSIDE,
    TEXT_PARTIAL,
    TEXT_START,
    ArrayNode,
    EnumNode,
    Grammar,
    LiteralNode,
    Node,
    NumberNode,
    SequenceNode,
    StringNode,
    arguments_grammar,
    function_name_grammar,
    value_grammar,
)
from src.models import FunctionDefinition, ParameterSpec


def accepts(spec: Dict[str, Any], text: str) -> bool:
    """Tell whether a text is a complete value of the given type."""
    grammar = value_grammar(ParameterSpec.model_validate(spec))
    return grammar.matches(text.encode("utf-8"))


@pytest.mark.parametrize(
    "text",
    ["0", "-0", "7", "12", "-3.5", "0.25", "1e10", "1E-5", "2.5e+3",
     "123456789012345678", "9" * MAX_NUMBER_CHARS],
)
def test_valid_numbers(text: str) -> None:
    assert accepts({"type": "number"}, text)
    json.loads(text)


@pytest.mark.parametrize(
    "text",
    ["", "-", "+1", "007", "01", "1.", ".5", "1e", "1e+", "1e999", "1.5.2",
     "1,5", "0x10", "NaN", "Infinity", "1 ", " 1", "--1",
     "9" * (MAX_NUMBER_CHARS + 1)],
)
def test_invalid_numbers(text: str) -> None:
    assert not accepts({"type": "number"}, text)


@pytest.mark.parametrize("text", ["0", "12", "-7"])
def test_valid_integers(text: str) -> None:
    assert accepts({"type": "integer"}, text)


@pytest.mark.parametrize("text", ["1.5", "1e3", "01", "1.0", ""])
def test_invalid_integers(text: str) -> None:
    assert not accepts({"type": "integer"}, text)


@pytest.mark.parametrize(
    "text",
    ['""', '"hi"', '"a b"', '"a\\"b"', '"a\\\\d+"', '"line\\nbreak"',
     '"café 你好 🙂"', '"\\u00e9"', '"\\/"', '"it\'s"', '"{[,:]}"',
     '"trailing "'],
)
def test_valid_strings(text: str) -> None:
    assert accepts({"type": "string"}, text)
    json.loads(text)


@pytest.mark.parametrize(
    "text",
    ['"', '"abc', 'abc"', '"a"b"', '"a\nb"', '"tab\there"', '"\\d"', '"\\x41"',
     '"\\u12"', '"\\ud83d"', '" leading space"', "'single'", '"\\b"'],
)
def test_invalid_strings(text: str) -> None:
    assert not accepts({"type": "string"}, text)


@pytest.mark.parametrize(
    "raw", [b'"\xff"', b'"\xc3"', b'"\xc3\x28"', b'"\xe0\x80\x80"',
            b'"\xed\xa0\x80"', b'"\xf5\x80\x80\x80"']
)
def test_invalid_utf8_is_refused(raw: bytes) -> None:
    grammar = value_grammar(ParameterSpec(type="string"))
    assert not grammar.matches(raw)


def test_booleans_and_null() -> None:
    assert accepts({"type": "boolean"}, "true")
    assert accepts({"type": "boolean"}, "false")
    assert not accepts({"type": "boolean"}, "True")
    assert not accepts({"type": "boolean"}, "tru")
    assert accepts({"type": "null"}, "null")
    assert not accepts({"type": "null"}, "nul")


def test_enums() -> None:
    spec = {"type": "string", "enum": ["a", "ab", 'q"uote']}
    assert accepts(spec, '"a"') and accepts(spec, '"ab"')
    assert accepts(spec, '"q\\"uote"')
    assert not accepts(spec, '"abc"') and not accepts(spec, '"b"')
    numbers = {"type": "integer", "enum": [1, 12]}
    assert accepts(numbers, "1") and accepts(numbers, "12")
    assert not accepts(numbers, "2") and not accepts(numbers, "123")


@pytest.mark.parametrize(
    "text", ["[]", "[1]", "[1, 2.5, -3]", "[0, 0, 0, 0]"]
)
def test_valid_arrays(text: str) -> None:
    assert accepts({"type": "array", "items": {"type": "number"}}, text)
    json.loads(text)


@pytest.mark.parametrize(
    "text", ["", "[", "[1", "[1,2]", "[1, ]", "[,]", "[1 , 2]", "1, 2", "[[]]"]
)
def test_invalid_arrays(text: str) -> None:
    assert not accepts({"type": "array", "items": {"type": "number"}}, text)


def test_array_size_is_bounded() -> None:
    spec = {"type": "array", "items": {"type": "integer"}}
    full = "[" + ", ".join(["1"] * MAX_ARRAY_ITEMS) + "]"
    assert accepts(spec, full)
    assert not accepts(spec, full[:-1] + ", 1]")


def test_nested_structures() -> None:
    matrix = {
        "type": "array",
        "items": {"type": "array", "items": {"type": "integer"}},
    }
    assert accepts(matrix, "[[1, 2], [], [3]]")
    assert accepts(
        {"type": "array", "items": {"type": "string"}}, '["a", "b, c", ""]'
    )
    record = {
        "type": "object",
        "properties": {
            "x": {"type": "number"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "inner": {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
            },
        },
    }
    good = '{"x": 1, "tags": ["q"], "inner": {"ok": true}}'
    assert accepts(record, good)
    assert not accepts(record, '{"x": 1, "tags": ["q"]}')
    swapped = '{"tags": ["q"], "x": 1, "inner": {"ok": true}}'
    assert not accepts(record, swapped)
    assert not accepts(record, good[:-1] + ', "more": 1}')
    assert accepts({"type": "object"}, "{}")


def test_function_name_grammar(functions: List[FunctionDefinition]) -> None:
    grammar = function_name_grammar([f.name for f in functions])
    assert grammar.matches(b'{"name": "fn_add"')
    assert grammar.matches(b'{"name": "fn_add_numbers"')
    assert not grammar.matches(b'{"name": "fn_ad"')
    assert not grammar.matches(b'{"name": "fn_unknown"')
    assert not grammar.matches(b'{"name": "fn_add" ')


def test_function_names_needing_escapes() -> None:
    grammar = function_name_grammar(['we"ird', "ñandú"])
    assert grammar.matches(b'{"name": "we\\"ird"')
    assert grammar.matches(b'{"name": "\\u00f1and\\u00fa"')


def test_arguments_grammar(functions: List[FunctionDefinition]) -> None:
    add, greet, _, complex_, no_args = functions
    assert arguments_grammar(add).matches(
        b', "parameters": {"a": 2, "b": 3.0}}'
    )
    assert not arguments_grammar(add).matches(b', "parameters": {"a": 2}}')
    assert not arguments_grammar(add).matches(
        b', "parameters": {"a": "2", "b": 3}}'
    )
    assert not arguments_grammar(add).matches(
        b', "parameters": {"a": 2, "b": 3, "c": 4}}'
    )
    assert arguments_grammar(greet).matches(
        b', "parameters": {"name": "shrek"}}'
    )
    assert arguments_grammar(no_args).matches(b', "parameters": {}}')
    assert arguments_grammar(complex_).matches(
        b', "parameters": {"count": 2, "tags": ["a"], '
        b'"point": {"x": 0.5, "label": "high"}, "nothing": null}}'
    )


def test_forced_bytes(functions: List[FunctionDefinition]) -> None:
    names = function_name_grammar([f.name for f in functions])
    state = names.initial_state()
    assert names.forced_bytes(state) == b'{"name": "'
    after_prefix = names.run(state, b'{"name": "fn_')
    assert after_prefix is not None
    assert names.forced_bytes(after_prefix) == b""  # several candidates
    unique = names.run(state, b'{"name": "fn_g')
    assert unique is not None
    assert names.forced_bytes(unique) == b'reet"'
    # The space before a value is left for the model to write.
    arguments = arguments_grammar(functions[0])
    assert arguments.forced_bytes(arguments.initial_state()) == (
        b', "parameters": {"a":'
    )
    greeting = arguments_grammar(functions[1])
    # ... and so is the opening quote of a string (' "*' is one token).
    assert greeting.forced_bytes(greeting.initial_state()) == (
        b', "parameters": {"name":'
    )
    opened = greeting.run(
        greeting.initial_state(), b', "parameters": {"name": "'
    )
    assert opened is not None and greeting.forced_bytes(opened) == b""


def test_states_are_hashable_and_stable() -> None:
    grammar = value_grammar(ParameterSpec(type="string"))
    first = grammar.run(grammar.initial_state(), b'"ab')
    second = grammar.run(grammar.initial_state(), b'"xyz')
    assert first is not None and first == second
    assert hash(first) == hash(second)
    assert not grammar.is_complete(first)
    assert grammar.advance(first, 0x0A) is None


def test_base_node_accepts_nothing() -> None:
    node = Node()
    assert node.step(node.start(), ord("a")) is None
    assert node.skip(node.start()) is None
    assert node.open(node.start()) is None
    assert node.forced(node.start()) == b""
    assert node.text_mode(node.start()) == NOT_TEXT


def test_literal_node() -> None:
    node = LiteralNode(text=b"ab")
    assert node.forced(node.start()) == b"ab"
    middle = node.step(node.start(), ord("a"))
    assert middle is not None and node.forced(middle) == b"b"
    assert node.step(middle, ord("b")) == DONE
    assert node.step(middle, ord("x")) is None
    assert LiteralNode(text=b"").start() == DONE


def test_enum_node() -> None:
    node = EnumNode(options=(b"a", b"ab", b"cd"))
    assert node.forced(node.start()) == b""          # three candidates
    assert node.skip(node.start()) is None           # nothing read yet
    after_a = node.step(node.start(), ord("a"))
    assert after_a is not None
    assert node.skip(after_a) == DONE                # "a" is an option
    assert node.forced(after_a) == b""               # but "ab" is possible
    assert node.step(after_a, ord("b")) == DONE
    after_c = node.step(node.start(), ord("c"))
    assert after_c is not None and node.forced(after_c) == b"d"
    assert node.skip(after_c) is None
    assert node.step(node.start(), ord("z")) is None


def test_string_node_text_modes() -> None:
    node = StringNode()
    assert node.text_mode(node.start()) == TEXT_START
    inside = node.step(node.start(), ord("a"))
    assert inside is not None and node.text_mode(inside) == TEXT_INSIDE
    escape = node.step(inside, ord("\\"))
    assert escape is not None and node.text_mode(escape) == TEXT_PARTIAL
    half = node.step(inside, "é".encode("utf-8")[0])
    assert half is not None and node.text_mode(half) == TEXT_PARTIAL
    assert node.step(inside, ord('"')) == DONE
    assert node.skip(inside) is None  # a string needs its closing quote


def test_number_node_can_be_skipped_only_when_complete() -> None:
    node = NumberNode()
    assert node.skip(node.start()) is None
    one = node.step(node.start(), ord("1"))
    assert one is not None and node.skip(one) == DONE
    dot = node.step(one, ord("."))
    assert dot is not None and node.skip(dot) is None
    assert NumberNode(integer_only=True).step(one, ord(".")) is None


def test_sequence_and_array_open_their_children() -> None:
    sequence = SequenceNode(children=(4, 9))
    assert sequence.open(sequence.start()) == ((1,), 4)
    assert sequence.open((1,)) == (DONE, 9)
    assert SequenceNode(children=()).start() == DONE
    array = ArrayNode(item=3)
    assert array.open(array.start()) is None
    opened = array.step(array.start(), ord("["))
    assert opened is not None and array.step(opened, ord("]")) == DONE
    first_item = array.skip(opened)
    assert first_item is not None and array.open(first_item) is not None


def test_grammar_keeps_node_subclasses() -> None:
    grammar = Grammar(nodes=(LiteralNode(text=b"x"), NumberNode()), root=0)
    assert isinstance(grammar.nodes[0], LiteralNode)
    assert isinstance(grammar.nodes[1], NumberNode)
    assert grammar.matches(b"x") and not grammar.matches(b"y")
