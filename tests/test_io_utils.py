"""Tests of input loading and output writing."""

import json
import os
from pathlib import Path
from typing import Any, Callable

import pytest

from src.io_utils import load_functions, load_prompts, write_results
from src.models import FunctionCall

Writer = Callable[[str, Any], str]

GOOD_FUNCTION = {"name": "f", "parameters": {"a": {"type": "number"}}}


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="not found"):
        load_functions(str(tmp_path / "nope.json"))


def test_directory_instead_of_file(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="directory"):
        load_prompts(str(tmp_path))


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read everything")
def test_unreadable_file(write_json: Writer) -> None:
    path = write_json("locked.json", [GOOD_FUNCTION])
    os.chmod(path, 0)
    try:
        with pytest.raises(OSError, match="permission"):
            load_functions(path)
    finally:
        os.chmod(path, 0o600)


@pytest.mark.parametrize("text", ["", "   \n", "{oops", "[1, 2,]", "//x\n[]"])
def test_broken_json(write_json: Writer, text: str) -> None:
    with pytest.raises(ValueError):
        load_functions(write_json("bad.json", text))


def test_binary_file(tmp_path: Path) -> None:
    path = tmp_path / "bin.json"
    path.write_bytes(b"\xff\xfe\x00\x01")
    with pytest.raises(ValueError, match="UTF-8"):
        load_prompts(str(path))


@pytest.mark.parametrize("content", [{"a": 1}, "text", 3, None])
def test_root_must_be_an_array(write_json: Writer, content: Any) -> None:
    with pytest.raises(ValueError, match="array"):
        load_prompts(write_json("root.json", json.dumps(content)))


def test_invalid_functions_are_skipped(
    write_json: Writer, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_json("f.json", [
        GOOD_FUNCTION,
        {"name": "g", "parameters": {"a": {"type": "banana"}}},
        "not an object",
        {"description": "nameless"},
        GOOD_FUNCTION,
    ])
    functions = load_functions(path)
    assert [function.name for function in functions] == ["f"]
    warnings = capsys.readouterr().err
    assert warnings.count("warning:") == 4
    assert "unsupported type 'banana'" in warnings
    assert "already defined" in warnings


def test_no_usable_function(write_json: Writer) -> None:
    with pytest.raises(ValueError, match="no usable function"):
        load_functions(write_json("f.json", []))
    with pytest.raises(ValueError, match="no usable function"):
        load_functions(write_json("g.json", [{"nope": 1}]))


def test_invalid_prompts_are_skipped(
    write_json: Writer, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_json("p.json", [
        {"prompt": "ok"}, {"prompt": 3}, {"text": "x"}, "bare", None,
        {"prompt": ""}, {"prompt": "é", "extra": True},
    ])
    prompts = load_prompts(path)
    assert [item.prompt for item in prompts] == ["ok", "", "é"]
    assert capsys.readouterr().err.count("warning:") == 4


def test_empty_prompt_list_is_allowed(write_json: Writer) -> None:
    assert load_prompts(write_json("p.json", [])) == []


def test_write_creates_directories(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "out.json"
    calls = [FunctionCall(prompt="é ü", name="f", parameters={"a": 1.0})]
    write_results(str(target), calls)
    assert json.loads(target.read_text(encoding="utf-8")) == [
        {"prompt": "é ü", "name": "f", "parameters": {"a": 1.0}}
    ]
    assert os.listdir(target.parent) == ["out.json"]  # no temporary left


def test_write_replaces_previous_content(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    target.write_text("old")
    write_results(str(target), [])
    assert json.loads(target.read_text()) == []


def test_write_to_impossible_location(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(OSError, match="cannot write"):
        write_results(str(blocker / "out.json"), [])
