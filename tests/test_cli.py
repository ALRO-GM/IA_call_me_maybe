"""Tests of the command line: every failure must be a clean message."""

import json
from pathlib import Path
from typing import Any, Callable, List

import pytest

from src import __main__ as cli
from src import io_utils
from src.models import DEFAULT_OUTPUT, Settings
from tests.conftest import SAMPLE_FUNCTIONS

Writer = Callable[[str, Any], str]


def test_default_arguments() -> None:
    settings = cli.parse_arguments([])
    assert settings == Settings()
    assert settings.output == DEFAULT_OUTPUT
    assert settings.functions_definition.startswith("data/input/")


def test_custom_arguments() -> None:
    settings = cli.parse_arguments([
        "--functions_definition", "f.json", "--input", "i.json",
        "--output", "o.json", "--model", "some/model", "--visualize",
    ])
    assert (settings.functions_definition, settings.input) == (
        "f.json", "i.json"
    )
    assert settings.output == "o.json" and settings.model == "some/model"
    assert settings.visualize is True


def test_unknown_option_exits_cleanly(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as stop:
        cli.parse_arguments(["--nope"])
    assert stop.value.code == 2
    assert "usage:" in capsys.readouterr().err


def test_invalid_limit_is_reported(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["--max_new_tokens", "0"]) == cli.EXIT_ERROR
    assert "error: invalid option" in capsys.readouterr().err


@pytest.mark.parametrize(
    "functions, prompts, message",
    [
        (None, [], "functions definition file not found"),
        ("{broken", [], "not valid JSON"),
        ({"a": 1}, [], "must contain a JSON array"),
        ([], [], "no usable function"),
        (SAMPLE_FUNCTIONS, None, "input file not found"),
        (SAMPLE_FUNCTIONS, "[1,", "not valid JSON"),
        (SAMPLE_FUNCTIONS, "", "input file is empty"),
    ],
)
def test_bad_inputs_give_one_line_errors(
    functions: Any,
    prompts: Any,
    message: str,
    tmp_path: Path,
    write_json: Writer,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments: List[str] = ["--output", str(tmp_path / "out.json")]
    arguments += ["--functions_definition", (
        str(tmp_path / "missing_f.json") if functions is None
        else write_json("f.json", functions)
    )]
    arguments += ["--input", (
        str(tmp_path / "missing_p.json") if prompts is None
        else write_json("p.json", prompts)
    )]
    assert cli.main(arguments) == cli.EXIT_ERROR
    captured = capsys.readouterr().err
    assert message in captured and "Traceback" not in captured
    assert captured.strip().splitlines()[-1].startswith("error: ")


def test_no_prompt_writes_an_empty_array(
    tmp_path: Path, write_json: Writer
) -> None:
    output = tmp_path / "deep" / "out.json"
    status = cli.main([
        "--functions_definition", write_json("f.json", SAMPLE_FUNCTIONS),
        "--input", write_json("p.json", [{"nope": 1}]),
        "--output", str(output),
    ])
    assert status == cli.EXIT_OK
    assert json.loads(output.read_text()) == []


def test_unwritable_output_fails_before_loading_the_model(
    tmp_path: Path,
    write_json: Writer,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def explode(name: str, device: str) -> Any:
        raise AssertionError("the model must not be loaded")

    monkeypatch.setattr(cli, "load_model", explode)
    blocker = tmp_path / "file"
    blocker.write_text("x")
    status = cli.main([
        "--functions_definition", write_json("f.json", SAMPLE_FUNCTIONS),
        "--input", write_json("p.json", [{"prompt": "hi"}]),
        "--output", str(blocker / "out.json"),
    ])
    assert status == cli.EXIT_ERROR
    assert "cannot write output file" in capsys.readouterr().err


def test_model_failure_is_reported(
    tmp_path: Path,
    write_json: Writer,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def broken(name: str, device: str) -> Any:
        raise RuntimeError(f"cannot load model '{name}': offline")

    monkeypatch.setattr(cli, "load_model", broken)
    status = cli.main([
        "--functions_definition", write_json("f.json", SAMPLE_FUNCTIONS),
        "--input", write_json("p.json", [{"prompt": "hi"}]),
        "--output", str(tmp_path / "out.json"),
    ])
    assert status == cli.EXIT_ERROR
    assert "error: cannot load model" in capsys.readouterr().err


def test_full_run_with_a_fake_model(
    tmp_path: Path,
    write_json: Writer,
    monkeypatch: pytest.MonkeyPatch,
    tokenizer: Any,
    random_model: Callable[[int], Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeModel:
        get_logits_from_input_ids = staticmethod(random_model(5))

    monkeypatch.setattr(
        cli, "load_model", lambda name, device: (FakeModel(), tokenizer)
    )
    output = tmp_path / "out.json"
    prompts = [{"prompt": "one"}, {"prompt": ""}, {"prompt": "three 你好"}]
    status = cli.main([
        "--functions_definition", write_json("f.json", SAMPLE_FUNCTIONS),
        "--input", write_json("p.json", prompts),
        "--output", str(output), "--visualize", "--max_string_tokens", "4",
    ])
    assert status == cli.EXIT_OK
    results = json.loads(output.read_text(encoding="utf-8"))
    assert [entry["prompt"] for entry in results] == ["one", "", "three 你好"]
    names = {function["name"] for function in SAMPLE_FUNCTIONS}
    for entry in results:
        assert list(entry) == ["prompt", "name", "parameters"]
        assert entry["name"] in names
    captured = capsys.readouterr()
    assert "Results written to" in captured.out
    assert "[3/3]" in captured.err and "JSON: " in captured.err


def test_recovery_retries_then_skips(
    tmp_path: Path,
    write_json: Writer,
    monkeypatch: pytest.MonkeyPatch,
    tokenizer: Any,
    random_model: Callable[[int], Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeModel:
        get_logits_from_input_ids = staticmethod(random_model(9))

    monkeypatch.setattr(
        cli, "load_model", lambda name, device: (FakeModel(), tokenizer)
    )
    output = tmp_path / "out.json"
    status = cli.main([
        "--functions_definition", write_json("f.json", SAMPLE_FUNCTIONS),
        "--input", write_json("p.json", [{"prompt": "a"}]),
        "--output", str(output), "--max_new_tokens", "1",
    ])
    assert status == cli.EXIT_OK
    assert json.loads(output.read_text()) == []
    errors = capsys.readouterr().err
    assert "retrying with stricter limits" in errors
    assert "prompt 1 skipped" in errors and "Traceback" not in errors


def test_interrupt_keeps_partial_results(
    tmp_path: Path,
    write_json: Writer,
    monkeypatch: pytest.MonkeyPatch,
    tokenizer: Any,
    random_model: Callable[[int], Any],
) -> None:
    logits = random_model(2)
    seen: List[int] = []

    class FakeModel:
        @staticmethod
        def get_logits_from_input_ids(ids: List[int]) -> List[float]:
            seen.append(1)
            if cli_state["done"] >= 1:
                raise KeyboardInterrupt
            result: List[float] = logits(ids)
            return result

    cli_state = {"done": 0}
    real_write = io_utils.write_results

    def counting_write(path: str, results: Any) -> None:
        cli_state["done"] = len(results)
        real_write(path, results)

    monkeypatch.setattr(cli, "write_results", counting_write)
    monkeypatch.setattr(
        cli, "load_model", lambda name, device: (FakeModel(), tokenizer)
    )
    output = tmp_path / "out.json"
    status = cli.main([
        "--functions_definition", write_json("f.json", SAMPLE_FUNCTIONS[:2]),
        "--input", write_json("p.json", [{"prompt": "a"}, {"prompt": "b"}]),
        "--output", str(output), "--max_string_tokens", "3",
    ])
    assert status == cli.EXIT_INTERRUPTED
    assert len(json.loads(output.read_text())) == 1
