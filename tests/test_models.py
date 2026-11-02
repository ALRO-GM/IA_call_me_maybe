"""Tests of the pydantic models and of value coercion."""

from typing import Any, Dict

import pytest
from pydantic import ValidationError

from src.models import (
    FunctionCall,
    FunctionDefinition,
    ParameterSpec,
    Settings,
    coerce_properties,
)


@pytest.mark.parametrize(
    "written, canonical",
    [
        ("number", "number"), ("Float", "number"), ("int", "integer"),
        ("STRING", "string"), ("str", "string"), ("bool", "boolean"),
        (" list ", "array"), ("dict", "object"), ("None", "null"),
    ],
)
def test_type_aliases(written: str, canonical: str) -> None:
    assert ParameterSpec(type=written).type == canonical


@pytest.mark.parametrize("bad", ["datetime", "", 3, None, ["number"]])
def test_unknown_type_is_rejected(bad: Any) -> None:
    with pytest.raises(ValidationError):
        ParameterSpec.model_validate({"type": bad})


def test_array_gets_a_default_item_type() -> None:
    spec = ParameterSpec(type="array")
    assert spec.items is not None and spec.items.type == "string"


def test_enum_must_be_non_empty_scalars() -> None:
    with pytest.raises(ValidationError):
        ParameterSpec.model_validate({"type": "string", "enum": []})
    with pytest.raises(ValidationError):
        ParameterSpec.model_validate({"type": "string", "enum": [[1]]})


@pytest.mark.parametrize(
    "raw",
    [
        {"name": ""},
        {"name": "   "},
        {"name": "a\nb"},
        {"description": "no name"},
        {"name": "f", "parameters": {"a": {"type": "banana"}}},
        {"name": "f", "parameters": {"a": {}}},
        {"name": "f", "parameters": [1, 2]},
        {"name": 12},
    ],
)
def test_invalid_function_definitions(raw: Dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        FunctionDefinition.model_validate(raw)


def test_function_definition_defaults() -> None:
    function = FunctionDefinition.model_validate(
        {"name": "f", "description": None, "parameters": None, "extra": 1}
    )
    assert function.description == "" and function.parameters == {}


def test_number_is_always_a_float() -> None:
    value = ParameterSpec(type="number").coerce(2)
    assert value == 2.0 and isinstance(value, float)


def test_integer_stays_an_integer() -> None:
    value = ParameterSpec(type="integer").coerce(7)
    assert value == 7 and isinstance(value, int)


@pytest.mark.parametrize(
    "value, kind",
    [
        ("3", "number"), (True, "number"), (1.5, "integer"),
        (True, "integer"), (3, "string"), ("true", "boolean"),
        (0, "boolean"), (0, "null"), ("x", "array"), ([], "object"),
    ],
)
def test_wrong_types_are_refused(value: Any, kind: str) -> None:
    with pytest.raises(ValueError):
        ParameterSpec(type=kind).coerce(value)


def test_nested_values_are_coerced() -> None:
    spec = ParameterSpec.model_validate({
        "type": "object",
        "properties": {
            "xs": {"type": "array", "items": {"type": "number"}},
            "ok": {"type": "boolean"},
        },
    })
    assert spec.coerce({"xs": [1, 2], "ok": True}) == {
        "xs": [1.0, 2.0], "ok": True,
    }


def test_enum_values_are_checked() -> None:
    spec = ParameterSpec(type="string", enum=["a", "b"])
    assert spec.coerce("a") == "a"
    with pytest.raises(ValueError):
        spec.coerce("c")


def test_missing_and_extra_keys_are_refused() -> None:
    properties = {"a": ParameterSpec(type="number")}
    with pytest.raises(ValueError):
        coerce_properties({}, properties)
    with pytest.raises(ValueError):
        coerce_properties({"a": 1, "b": 2}, properties)
    with pytest.raises(ValueError):
        coerce_properties([1], properties)


def test_function_call_has_exactly_three_keys() -> None:
    call = FunctionCall(prompt="p", name="f", parameters={"a": 1.0})
    assert list(call.model_dump()) == ["prompt", "name", "parameters"]
    with pytest.raises(ValidationError):
        FunctionCall.model_validate(
            {"prompt": "p", "name": "f", "parameters": {}, "extra": 1}
        )


def test_settings_reject_bad_limits() -> None:
    with pytest.raises(ValidationError):
        Settings(max_new_tokens=0)


def test_descriptions_shown_to_the_model() -> None:
    assert ParameterSpec(type="number").describe() == "number"
    assert ParameterSpec(type="list").describe() == "array of string"
    nested = ParameterSpec.model_validate({
        "type": "object",
        "properties": {
            "x": {"type": "int"},
            "u": {"type": "str", "enum": ["a"]},
        },
    })
    assert nested.describe() == "object {x: integer, u: one of 'a'}"


def test_function_signature_and_arguments() -> None:
    function = FunctionDefinition.model_validate({
        "name": "fn_add",
        "description": "Add  two\nnumbers.",
        "parameters": {"a": {"type": "number"}, "b": {"type": "int"}},
    })
    assert function.signature() == (
        "fn_add(a: number, b: integer) - Add two numbers."
    )
    assert FunctionDefinition(name="f").signature() == "f()"
    assert function.coerce_arguments({"a": 1, "b": 2}) == {"a": 1.0, "b": 2}
    with pytest.raises(ValueError):
        function.coerce_arguments({"a": 1})
