"""Pydantic models for the input files, the results and the settings."""

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

TYPE_ALIASES: Dict[str, str] = {
    "number": "number",
    "float": "number",
    "double": "number",
    "integer": "integer",
    "int": "integer",
    "string": "string",
    "str": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "null": "null",
    "none": "null",
    "array": "array",
    "list": "array",
    "object": "object",
    "dict": "object",
}
"""Accepted spellings of a parameter type and their canonical name."""

SCALAR_TYPES: Dict[str, Tuple[type, ...]] = {
    "number": (int, float),
    "integer": (int,),
    "string": (str,),
    "boolean": (bool,),
    "null": (type(None),),
}
"""Python types accepted for each scalar JSON type."""

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_FUNCTIONS = "data/input/functions_definition.json"
DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT = "data/output/function_calling_results.json"


class ParameterSpec(BaseModel):
    """Type description of one value (argument, array item or property).

    Attributes:
        type: Canonical JSON type name (see ``TYPE_ALIASES``).
        description: Optional human readable explanation.
        enum: Optional closed list of allowed scalar values.
        items: Item type, only meaningful when ``type`` is ``array``.
        properties: Nested fields, only meaningful for ``object``.
    """

    model_config = ConfigDict(extra="ignore")

    type: str
    description: Optional[str] = None
    enum: Optional[List[Any]] = None
    items: Optional["ParameterSpec"] = None
    properties: Optional[Dict[str, "ParameterSpec"]] = None

    @field_validator("type", mode="before")
    @classmethod
    def _normalise_type(cls, value: Any) -> str:
        """Lower-case the type and map aliases to the canonical name."""
        if not isinstance(value, str):
            raise ValueError("type must be a string")
        key = value.strip().lower()
        if key not in TYPE_ALIASES:
            known = ", ".join(sorted(set(TYPE_ALIASES.values())))
            raise ValueError(
                f"unsupported type '{value}' (supported: {known})"
            )
        return TYPE_ALIASES[key]

    @field_validator("enum")
    @classmethod
    def _check_enum(cls, value: Optional[List[Any]]) -> Optional[List[Any]]:
        """Ensure an enum is a non-empty list of JSON scalars."""
        if value is None:
            return None
        if not value:
            raise ValueError("enum must not be empty")
        for option in value:
            if option is not None and not isinstance(
                option, (str, int, float, bool)
            ):
                raise ValueError("enum values must be JSON scalars")
        return value

    @model_validator(mode="after")
    def _check_shape(self) -> "ParameterSpec":
        """Fill in defaults so every array has an item type."""
        if self.type == "array" and self.items is None:
            self.items = ParameterSpec(type="string")
        if self.type == "object" and self.properties is None:
            self.properties = {}
        return self

    def describe(self) -> str:
        """Write the type the way it is shown to the model.

        Returns:
            A compact label such as ``number`` or ``array of string``.
        """
        if self.enum is not None:
            return "one of " + ", ".join(repr(option) for option in self.enum)
        if self.type == "array" and self.items is not None:
            return f"array of {self.items.describe()}"
        if self.type == "object":
            fields = ", ".join(
                f"{key}: {spec.describe()}"
                for key, spec in (self.properties or {}).items()
            )
            return "object {" + fields + "}"
        return self.type

    def coerce(self, value: Any) -> Any:
        """Convert a decoded JSON value to the Python type of this spec.

        Constrained decoding already guarantees the JSON shape; this step
        is an independent check that also settles what JSON leaves open,
        such as writing every ``number`` as a float.

        Args:
            value: Value obtained from ``json.loads``.

        Returns:
            The value converted to the type described by this spec.

        Raises:
            ValueError: If the value does not match the spec.
        """
        if self.enum is not None:
            for option in self.enum:
                if type(option) is type(value) and option == value:
                    return value
            raise ValueError(f"{value!r} is not one of {self.enum!r}")
        if self.type == "array":
            if not isinstance(value, list) or self.items is None:
                raise ValueError(f"expected an array, got {value!r}")
            return [self.items.coerce(item) for item in value]
        if self.type == "object":
            return coerce_properties(value, self.properties or {})
        expected = SCALAR_TYPES[self.type]
        # bool is a subclass of int in Python, but not a JSON number.
        stray_boolean = isinstance(value, bool) and bool not in expected
        right_type = isinstance(value, expected)
        if stray_boolean or not right_type:
            raise ValueError(f"expected {self.type}, got {value!r}")
        return float(value) if self.type == "number" else value


def coerce_properties(
    value: Any, properties: Dict[str, ParameterSpec]
) -> Dict[str, Any]:
    """Check an object has exactly the expected keys and coerce them.

    Args:
        value: Decoded JSON value, expected to be an object.
        properties: Expected keys and their types.

    Returns:
        A new dictionary with every value coerced, in schema order.

    Raises:
        ValueError: If keys are missing, unexpected or badly typed.
    """
    if not isinstance(value, dict):
        raise ValueError(f"expected an object, got {value!r}")
    missing = [key for key in properties if key not in value]
    extra = [key for key in value if key not in properties]
    if missing or extra:
        raise ValueError(
            f"wrong keys (missing: {missing}, unexpected: {extra})"
        )
    return {key: spec.coerce(value[key]) for key, spec in properties.items()}


class FunctionDefinition(BaseModel):
    """One callable function advertised to the model.

    Attributes:
        name: Unique function name.
        description: What the function does, shown to the model.
        parameters: Mapping from argument name to its type.
        returns: Return type description (kept only for documentation).
    """

    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)
    description: str = ""
    parameters: Dict[str, ParameterSpec] = Field(default_factory=dict)
    returns: Optional[Dict[str, Any]] = None

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        """Reject names that are blank or contain control characters."""
        if not value.strip():
            raise ValueError("name must not be blank")
        if any(ord(char) < 32 for char in value):
            raise ValueError("name must not contain control characters")
        return value

    @field_validator("description", mode="before")
    @classmethod
    def _default_description(cls, value: Any) -> Any:
        """Treat a null description as an empty one."""
        return "" if value is None else value

    @field_validator("parameters", mode="before")
    @classmethod
    def _default_parameters(cls, value: Any) -> Any:
        """Treat null parameters as a function without arguments."""
        return {} if value is None else value

    def signature(self) -> str:
        """Write the function as a single line of the prompt.

        Returns:
            A line such as ``fn_add(a: number, b: number) - Add numbers.``
        """
        arguments = ", ".join(
            f"{name}: {spec.describe()}"
            for name, spec in self.parameters.items()
        )
        description = " ".join(self.description.split())
        line = f"{self.name}({arguments})"
        return f"{line} - {description}" if description else line

    def coerce_arguments(self, value: Any) -> Dict[str, Any]:
        """Check and convert the arguments generated for this function.

        Args:
            value: The decoded ``parameters`` object.

        Returns:
            The arguments, typed as the definition requires.

        Raises:
            ValueError: If the arguments do not match the definition.
        """
        return coerce_properties(value, self.parameters)


class PromptItem(BaseModel):
    """One natural language request read from the tests file.

    Attributes:
        prompt: The user request.
    """

    model_config = ConfigDict(extra="ignore")

    prompt: str


class FunctionCall(BaseModel):
    """One line of the output file.

    Attributes:
        prompt: The original natural language request.
        name: Name of the selected function.
        parameters: Arguments, typed as the function definition requires.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: Dict[str, Any]


class Settings(BaseModel):
    """Run configuration built from the command line.

    Attributes:
        functions_definition: Path of the functions definition file.
        input: Path of the prompts file.
        output: Path of the results file.
        model: Hugging Face identifier of the model to load.
        device: Where the model runs; ``auto`` lets the SDK choose.
        visualize: Print every decoding step to the terminal.
        max_new_tokens: Upper bound of generated tokens per prompt.
        max_string_tokens: Upper bound of tokens per string value.
        threads: CPU threads of the model; 0 chooses automatically.
    """

    model_config = ConfigDict(extra="forbid")

    functions_definition: str = DEFAULT_FUNCTIONS
    input: str = DEFAULT_INPUT
    output: str = DEFAULT_OUTPUT
    model: str = DEFAULT_MODEL
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    visualize: bool = False
    max_new_tokens: int = Field(default=400, ge=1)
    max_string_tokens: int = Field(default=96, ge=1)
    threads: int = Field(default=0, ge=0)
