"""Tests of the prompt builder, the function caller and the visualizer."""

from typing import Any, Callable, List

import pytest

from src.decoder import ConstrainedDecoder
from src.function_caller import FunctionCaller
from src.models import FunctionCall, FunctionDefinition
from src.prompt_builder import PromptBuilder
from src.tokenizer import Tokenizer
from src.visualizer import Visualizer

DecoderFactory = Callable[..., ConstrainedDecoder]


def test_plain_prompts(
    tokenizer: Tokenizer, functions: List[FunctionDefinition]
) -> None:
    builder = PromptBuilder(tokenizer=tokenizer, functions=functions)
    selection = tokenizer.decode(builder.selection_prompt("Greet bob"))
    assert all(function.name in selection for function in functions)
    assert selection.endswith("Request: Greet bob\nAnswer: ")
    focused = tokenizer.decode(
        builder.arguments_prompt("Greet bob", functions[1])
    )
    assert "fn_greet(name: string)" in focused
    assert "fn_add_numbers(" not in focused
    assert focused.endswith("Request: Greet bob\nAnswer: ")


@pytest.mark.parametrize("seed", range(25))
def test_caller_output_always_matches_the_schema(
    seed: int,
    tokenizer: Tokenizer,
    functions: List[FunctionDefinition],
    random_model: Callable[[int], Any],
    make_decoder: DecoderFactory,
) -> None:
    caller = FunctionCaller(
        functions=functions,
        prompt_builder=PromptBuilder(tokenizer=tokenizer, functions=functions),
        decoder=make_decoder(random_model(seed), max_string_tokens=5),
    )
    prompt = "whatever \"quoted\" \\ text 你好"
    call, trace = caller.call(prompt)
    assert isinstance(call, FunctionCall) and call.prompt == prompt
    function = next(f for f in functions if f.name == call.name)
    assert list(call.parameters) == list(function.parameters)
    for key, spec in function.parameters.items():
        value = call.parameters[key]
        expected: Any = {
            "number": float, "integer": int, "string": str,
            "boolean": bool, "array": list, "object": dict,
            "null": type(None),
        }[spec.type]
        assert type(value) is expected
    assert trace.model_calls <= trace.generated_tokens
    assert trace.raw_json.startswith('{"name": "')


def test_caller_needs_functions(
    tokenizer: Tokenizer,
    random_model: Callable[[int], Any],
    make_decoder: DecoderFactory,
) -> None:
    with pytest.raises(ValueError):
        FunctionCaller(
            functions=[],
            prompt_builder=PromptBuilder(tokenizer=tokenizer, functions=[]),
            decoder=make_decoder(random_model(0)),
        )


def test_fallback_decoder_is_used_after_a_failure(
    tokenizer: Tokenizer,
    functions: List[FunctionDefinition],
    random_model: Callable[[int], Any],
    make_decoder: DecoderFactory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    builder = PromptBuilder(tokenizer=tokenizer, functions=functions)
    too_short = make_decoder(random_model(4), max_new_tokens=1)
    alone = FunctionCaller(
        functions=functions, prompt_builder=builder, decoder=too_short
    )
    with pytest.raises(RuntimeError, match="stopped after 1 tokens"):
        alone.call("anything")
    assert "retrying" not in capsys.readouterr().err
    helped = FunctionCaller(
        functions=functions,
        prompt_builder=builder,
        decoder=too_short,
        fallback_decoder=make_decoder(random_model(4), max_string_tokens=4),
    )
    call, _ = helped.call("anything")
    assert call.name in {function.name for function in functions}
    assert "retrying with stricter limits" in capsys.readouterr().err


def test_visualizer_output(
    tokenizer: Tokenizer,
    functions: List[FunctionDefinition],
    random_model: Callable[[int], Any],
    make_decoder: DecoderFactory,
) -> None:
    caller = FunctionCaller(
        functions=functions,
        prompt_builder=PromptBuilder(tokenizer=tokenizer, functions=functions),
        decoder=make_decoder(random_model(1), trace=True, max_string_tokens=4),
    )
    call, trace = caller.call("Add 1 and 2")
    plain = Visualizer(color=False).render("Add 1 and 2", trace)
    assert "Prompt: Add 1 and 2" in plain and "grammar" in plain
    assert "model calls" in plain and "\033[" not in plain
    assert f"JSON: {trace.raw_json}" in plain
    assert "\033[" in Visualizer(color=True).render("Add 1 and 2", trace)
