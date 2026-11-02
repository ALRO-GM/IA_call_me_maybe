"""Command line entry point: ``uv run python -m src``."""

import argparse
import json
import os
import sys
import time
from typing import Any, List, Optional, Sequence, Tuple

from pydantic import ValidationError

from src.decoder import (
    ConstrainedDecoder,
    LogitsCache,
    LogitsFunction,
    TokenIndex,
)
from src.function_caller import FunctionCaller
from src.hardware import configure_threads
from src.io_utils import (
    describe_validation_error,
    info,
    load_functions,
    load_prompts,
    warn,
    write_results,
)
from src.models import FunctionCall, FunctionDefinition, Settings
from src.prompt_builder import PromptBuilder
from src.tokenizer import Tokenizer
from src.visualizer import Visualizer

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INTERRUPTED = 130

FALLBACK_TOKEN_FACTOR = 4
FALLBACK_STRING_TOKENS = 12


def parse_arguments(argv: Optional[Sequence[str]] = None) -> Settings:
    """Read the command line.

    The defaults shown in ``--help`` come from the ``Settings`` model, so
    they are defined in a single place.

    Args:
        argv: Arguments to parse; defaults to ``sys.argv[1:]``.

    Returns:
        The validated run settings.

    Raises:
        ValueError: If an option has an invalid value.
    """
    defaults = Settings()
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description=(
            "Translate natural language prompts into function calls using "
            "a small language model and constrained decoding."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--functions_definition",
        default=defaults.functions_definition,
        metavar="FILE",
        help="JSON file listing the available functions",
    )
    parser.add_argument(
        "--input",
        default=defaults.input,
        metavar="FILE",
        help="JSON file listing the prompts",
    )
    parser.add_argument(
        "--output",
        default=defaults.output,
        metavar="FILE",
        help="JSON file to write",
    )
    parser.add_argument(
        "--model",
        default=defaults.model,
        metavar="NAME",
        help="Hugging Face model to use",
    )
    parser.add_argument(
        "--device",
        default=defaults.device,
        choices=["auto", "cpu", "cuda", "mps"],
        help="where to run the model; auto uses a GPU when one is usable "
        "and the CPU otherwise",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="show every decoding step (allowed tokens, probabilities)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=defaults.max_new_tokens,
        metavar="N",
        help="maximum number of generated tokens per prompt",
    )
    parser.add_argument(
        "--max_string_tokens",
        type=int,
        default=defaults.max_string_tokens,
        metavar="N",
        help="maximum number of tokens per string value",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=defaults.threads,
        metavar="N",
        help="CPU threads used by the model; 0 uses only the performance "
        "cores of a hybrid processor (or the library default)",
    )
    try:
        return Settings(**vars(parser.parse_args(argv)))
    except ValidationError as error:
        raise ValueError(
            f"invalid option: {describe_validation_error(error)}"
        ) from None


def load_model(name: str, device: str = "auto") -> Tuple[Any, Tokenizer]:
    """Load the language model and build our own tokenizer for it.

    With ``auto`` the SDK picks the device itself: Apple GPU, then NVIDIA
    GPU, then CPU, depending on what the installed drivers support.

    Only two SDK methods are used by the program: ``get_path_to_vocab_file``
    to build the tokenizer and ``get_logits_from_input_ids`` to generate.

    Args:
        name: Hugging Face identifier of the model.
        device: ``auto``, ``cpu``, ``cuda`` or ``mps``.

    Returns:
        The SDK model object and the matching tokenizer.

    Raises:
        RuntimeError: If the model or its vocabulary cannot be loaded.
    """
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from llm_sdk import Small_LLM_Model
    except Exception as error:
        raise RuntimeError(
            f"cannot import llm_sdk ({error}); run 'uv sync' first"
        ) from None
    # Try the requested device first; if the GPU cannot be used (missing
    # driver, missing library, not enough memory) fall back to the CPU.
    attempts = [device] if device == "cpu" else [device, "cpu"]
    model: Any = None
    vocab_path = ""
    for position, target in enumerate(attempts):
        try:
            options = {} if target == "auto" else {"device": target}
            model = Small_LLM_Model(model_name=name, **options)
            vocab_path = model.get_path_to_vocab_file()
            break
        except Exception as error:
            reason = str(error).strip().splitlines()[0] if str(error) else ""
            problem = f"{type(error).__name__}: {reason}"
            if position + 1 < len(attempts):
                warn(
                    f"model could not be loaded on device '{target}' "
                    f"({problem}); trying the CPU"
                )
                continue
            raise RuntimeError(
                f"cannot load model '{name}': {problem}\n"
                "       check the model name and your internet connection "
                "(the first run downloads the model)"
            ) from None
    try:
        tokenizer = Tokenizer.from_file(vocab_path)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"cannot use the vocabulary of '{name}': {error}"
        ) from None
    if len(tokenizer.token_bytes) < 256:
        raise RuntimeError(
            f"model '{name}' does not use a byte-level BPE vocabulary, "
            "which this program requires"
        )
    return model, tokenizer


def build_caller(
    logits_function: LogitsFunction,
    tokenizer: Tokenizer,
    functions: List[FunctionDefinition],
    settings: Settings,
) -> FunctionCaller:
    """Assemble the pipeline that processes one prompt.

    Two decoders share the same token index. The fallback one closes
    strings much sooner and may generate more tokens: it is only used when
    the main one cannot finish an answer.

    Args:
        logits_function: Returns the next-token logits of a context.
        tokenizer: Tokenizer of the model.
        functions: Available functions.
        settings: Run settings.

    Returns:
        The function caller.
    """
    decoder = ConstrainedDecoder(
        tokenizer=tokenizer,
        index=TokenIndex(tokenizer=tokenizer),
        logits_function=logits_function,
        max_new_tokens=settings.max_new_tokens,
        max_string_tokens=settings.max_string_tokens,
        trace=settings.visualize,
    )
    fallback = decoder.model_copy(update={
        "max_new_tokens": settings.max_new_tokens * FALLBACK_TOKEN_FACTOR,
        "max_string_tokens": min(
            settings.max_string_tokens, FALLBACK_STRING_TOKENS
        ),
    })
    return FunctionCaller(
        functions=functions,
        prompt_builder=PromptBuilder(tokenizer=tokenizer, functions=functions),
        decoder=decoder,
        fallback_decoder=fallback,
    )


def process_prompts(
    caller: FunctionCaller, prompts: List[str], settings: Settings
) -> List[FunctionCall]:
    """Process every prompt, saving the results after each one.

    A prompt that cannot be processed is reported and skipped.

    Args:
        caller: The pipeline processing one prompt.
        prompts: The natural language requests.
        settings: Run settings.

    Returns:
        The function calls obtained.

    Raises:
        OSError: If the output file cannot be written.
        KeyboardInterrupt: If the user interrupts the run; the results
            obtained so far are already saved.
    """
    visualizer = Visualizer.for_stream(sys.stderr)
    results: List[FunctionCall] = []
    for number, prompt in enumerate(prompts, start=1):
        started = time.perf_counter()
        try:
            call, trace = caller.call(prompt)
        except RuntimeError as error:
            warn(f"prompt {number} skipped ({prompt!r}): {error}")
            continue
        results.append(call)
        write_results(settings.output, results)
        if settings.visualize:
            visualizer.show(prompt, trace)
        info(
            f"[{number}/{len(prompts)}] "
            f"{time.perf_counter() - started:5.1f}s  {call.name}"
            f"({json.dumps(call.parameters, ensure_ascii=False)})"
        )
    return results


def run(settings: Settings) -> int:
    """Execute the whole program.

    Args:
        settings: Run settings.

    Returns:
        The process exit status.

    Raises:
        OSError: If a file cannot be read or written.
        ValueError: If an input file has an invalid content.
        RuntimeError: If the model cannot be loaded.
    """
    functions = load_functions(settings.functions_definition)
    prompts = [item.prompt for item in load_prompts(settings.input)]
    info(f"{len(functions)} function(s), {len(prompts)} prompt(s) loaded")
    # Written first: an unwritable output fails before loading the model.
    write_results(settings.output, [])
    if not prompts:
        warn("no prompt to process, an empty result was written")
        return EXIT_OK

    started = time.perf_counter()
    threads = configure_threads(settings.threads)
    if threads is not None:
        info(f"using {threads} CPU thread(s)")
    info(f"loading model {settings.model} (device: {settings.device}) ...")
    model, tokenizer = load_model(settings.model, settings.device)
    # Every model call goes through the cache, which also counts them.
    cache = LogitsCache(logits_function=model.get_logits_from_input_ids)
    caller = build_caller(cache, tokenizer, functions, settings)
    info(f"model ready in {time.perf_counter() - started:.1f}s")

    started = time.perf_counter()
    try:
        results = process_prompts(caller, prompts, settings)
    except KeyboardInterrupt:
        info(f"\ninterrupted: partial results are in {settings.output}")
        return EXIT_INTERRUPTED
    info(
        f"done: {len(results)} call(s) in "
        f"{time.perf_counter() - started:.1f}s ({cache.misses} model calls, "
        f"{cache.hits} from cache, {len(prompts) - len(results)} failure(s))"
    )
    print(f"Results written to {settings.output}")
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the program and turn every failure into a clean message.

    Args:
        argv: Command line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        The process exit status.
    """
    try:
        return run(parse_arguments(argv))
    except KeyboardInterrupt:
        info("\ninterrupted")
        return EXIT_INTERRUPTED
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR
    except Exception as error:  # last resort: never show a traceback
        print(
            f"error: unexpected failure ({type(error).__name__}: {error})",
            file=sys.stderr,
        )
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
