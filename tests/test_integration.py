"""End to end tests with the real model (run with RUN_SLOW=1)."""

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src import __main__ as cli

EXPECTED_NAMES = {
    "What is the sum of 2 and 3?": "fn_add_numbers",
    "Greet shrek": "fn_greet",
    "Reverse the string 'hello'": "fn_reverse_string",
    "What is the square root of 16?": "fn_get_square_root",
    "Substitute the word 'cat' with 'dog' in "
    "'The cat sat on the mat with another cat'":
        "fn_substitute_string_with_regex",
}


@pytest.mark.slow
def test_provided_examples(tmp_path: Path) -> None:
    output = tmp_path / "results.json"
    assert cli.main(["--output", str(output)]) == cli.EXIT_OK
    results: List[Dict[str, Any]] = json.loads(output.read_text("utf-8"))
    with open("data/input/function_calling_tests.json") as handle:
        prompts = [item["prompt"] for item in json.load(handle)]
    assert [entry["prompt"] for entry in results] == prompts
    by_prompt = {entry["prompt"]: entry for entry in results}
    for prompt, name in EXPECTED_NAMES.items():
        assert by_prompt[prompt]["name"] == name
    assert by_prompt["What is the sum of 2 and 3?"]["parameters"] == {
        "a": 2.0, "b": 3.0,
    }
    assert by_prompt["Greet shrek"]["parameters"] == {"name": "shrek"}
    assert by_prompt["Reverse the string 'hello'"]["parameters"] == {
        "s": "hello"
    }
