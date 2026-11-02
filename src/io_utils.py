"""Reading the input files and writing the results, without crashing.

Every problem a user can cause (missing file, broken JSON, wrong shape) is
reported as ``OSError`` or ``ValueError`` with a message meant for humans.
Entries that are individually wrong are skipped with a warning so that one
bad line does not waste the whole run.
"""

import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Tuple, Type, TypeVar

from pydantic import BaseModel, ValidationError

from src.models import FunctionCall, FunctionDefinition, PromptItem

ModelT = TypeVar("ModelT", bound=BaseModel)


def info(message: str) -> None:
    """Print a progress message on the standard error stream.

    Standard output is kept for the final result line only.

    Args:
        message: Text to display.
    """
    print(message, file=sys.stderr, flush=True)


def warn(message: str) -> None:
    """Print a warning on the standard error stream.

    Args:
        message: Text of the warning.
    """
    info(f"warning: {message}")


def describe_validation_error(error: ValidationError) -> str:
    """Summarise a pydantic error in a single readable line.

    Args:
        error: The validation error raised by pydantic.

    Returns:
        A short description such as ``parameters.a.type: unsupported``.
    """
    parts: List[str] = []
    for issue in error.errors():
        where = ".".join(str(step) for step in issue["loc"])
        text = issue["msg"].replace("Value error, ", "")
        parts.append(f"{where}: {text}" if where else text)
    return "; ".join(parts)


def read_json(path: str, label: str) -> Any:
    """Load a JSON document from disk.

    Args:
        path: Location of the file.
        label: Name of the file's role, used in error messages.

    Returns:
        The decoded JSON document.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the content is not valid JSON.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except FileNotFoundError:
        raise OSError(f"{label} file not found: {path}") from None
    except IsADirectoryError:
        raise OSError(f"{label} path is a directory: {path}") from None
    except PermissionError:
        raise OSError(f"no permission to read {label} file: {path}") from None
    except UnicodeDecodeError:
        raise ValueError(f"{label} file is not UTF-8 text: {path}") from None
    except OSError as error:
        raise OSError(f"cannot read {label} file {path}: {error}") from None
    if not text.strip():
        raise ValueError(f"{label} file is empty: {path}")
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{label} file is not valid JSON: {path} "
            f"(line {error.lineno}, column {error.colno}: {error.msg})"
        ) from None
    except RecursionError:
        raise ValueError(
            f"{label} file is nested too deeply: {path}"
        ) from None


def read_json_array(path: str, label: str) -> List[Any]:
    """Load a JSON document that must be an array.

    Args:
        path: Location of the file.
        label: Name of the file's role, used in error messages.

    Returns:
        The decoded list.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the content is not a JSON array.
    """
    data = read_json(path, label)
    if not isinstance(data, list):
        raise ValueError(
            f"{label} file must contain a JSON array, "
            f"found {type(data).__name__}: {path}"
        )
    return data


def load_entries(
    path: str, label: str, noun: str, model: Type[ModelT], strict: bool
) -> List[Tuple[int, ModelT]]:
    """Read a JSON array and validate each entry with a pydantic model.

    Entries that fail validation are skipped with a warning, so one bad
    entry does not waste the whole file.

    Args:
        path: Location of the file.
        label: Name of the file's role, used in error messages.
        noun: Name of one entry, used in warnings.
        model: Pydantic model every entry must satisfy.
        strict: Refuse type conversions (a number is not a string).

    Returns:
        The valid entries with their position in the file.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the file is not a JSON array.
    """
    entries: List[Tuple[int, ModelT]] = []
    for index, raw in enumerate(read_json_array(path, label)):
        try:
            entries.append((index, model.model_validate(raw, strict=strict)))
        except ValidationError as error:
            warn(
                f"skipping {noun} #{index} in {path}: "
                f"{describe_validation_error(error)}"
            )
    return entries


def load_functions(path: str) -> List[FunctionDefinition]:
    """Read and validate the functions definition file.

    Invalid or duplicated definitions are skipped with a warning.

    Args:
        path: Location of the functions definition file.

    Returns:
        The usable function definitions, in file order.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If no usable function remains.
    """
    label = "functions definition"
    first_seen: Dict[str, int] = {}
    functions: List[FunctionDefinition] = []
    entries = load_entries(path, label, "function", FunctionDefinition, False)
    for index, function in entries:
        if function.name in first_seen:
            warn(
                f"skipping function #{index} in {path}: name '{function.name}'"
                f" already defined at #{first_seen[function.name]}"
            )
            continue
        first_seen[function.name] = index
        functions.append(function)
    if not functions:
        raise ValueError(f"no usable function found in {label} file: {path}")
    return functions


def load_prompts(path: str) -> List[PromptItem]:
    """Read and validate the prompts file.

    Entries without a string ``prompt`` are skipped with a warning.

    Args:
        path: Location of the prompts file.

    Returns:
        The prompts to process, in file order (possibly empty).

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the file is not a JSON array.
    """
    entries = load_entries(path, "input", "prompt", PromptItem, True)
    return [prompt for _, prompt in entries]


def write_results(path: str, results: List[FunctionCall]) -> None:
    """Write the results as a JSON array, replacing the file atomically.

    The data is first written to a temporary file in the same directory and
    then renamed, so an interrupted run never leaves a half written file.

    Args:
        path: Destination file.
        results: Function calls to serialise.

    Raises:
        OSError: If the destination cannot be written.
    """
    directory = os.path.dirname(os.path.abspath(path))
    payload = [result.model_dump() for result in results]
    temporary = ""
    try:
        os.makedirs(directory, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=directory,
            prefix=".results-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    except OSError as error:
        if temporary and os.path.exists(temporary):
            os.remove(temporary)
        raise OSError(f"cannot write output file {path}: {error}") from None
