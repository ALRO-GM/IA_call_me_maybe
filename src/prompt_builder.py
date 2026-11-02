"""Builds the text shown to the model before it starts answering.

The prompt only has to make the right answer *likely*: validity does not
depend on it, since the decoder enforces the JSON structure. It is kept
short on purpose because the model re-reads the whole prompt for every
generated token.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, PrivateAttr

from src.models import FunctionDefinition
from src.tokenizer import Tokenizer

SELECT_INSTRUCTIONS = (
    "You are a function calling engine. Pick the one function that best "
    "fulfils the user request. Answer with a single JSON object."
)

ARGUMENT_INSTRUCTIONS = (
    "You are a function calling engine. Extract from the user request the "
    "arguments of the function below. Answer with a single JSON object."
)

EXAMPLES = (
    "Examples of answers for other functions:\n"
    "Request: Replace each lowercase letter in 'Ab-cD' with underscores\n"
    'Answer: {"name": "fn_replace_pattern", "parameters": '
    '{"text": "Ab-cD", "pattern": "[a-z]", "replacement": "_"}}\n'
    "Request: Is 12.5 greater than -3?\n"
    'Answer: {"name": "fn_greater", "parameters": {"a": 12.5, "b": -3}}'
)

REQUEST_LABEL = "\n\nRequest: "
ANSWER_LABEL = "\nAnswer: "


class PromptBuilder(BaseModel):
    """Turns a user request into the token ids fed to the model.

    Two prompts are used, one per decoding stage:

    * the *selection* prompt lists every function, so the model can choose;
    * the *arguments* prompt shows only the chosen function plus a couple
      of worked examples, which is shorter (faster) and less distracting.

    Both are plain text ending with ``Request: <request>`` and ``Answer: ``.
    Everything before the request is tokenized once and reused.

    Attributes:
        tokenizer: Tokenizer of the model.
        functions: Functions the model may choose from.
    """

    tokenizer: Tokenizer
    functions: List[FunctionDefinition]

    _selection_head: List[int] = PrivateAttr(default_factory=list)
    _argument_heads: Dict[str, List[int]] = PrivateAttr(default_factory=dict)
    _tail: List[int] = PrivateAttr(default_factory=list)

    def model_post_init(self, context: Any) -> None:
        """Tokenize once the parts shared by every prompt.

        Args:
            context: Unused pydantic validation context.
        """
        listing = "\n".join(
            function.signature() for function in self.functions
        )
        self._selection_head = self.tokenizer.encode(
            f"{SELECT_INSTRUCTIONS}\n\nFunctions:\n{listing}{REQUEST_LABEL}"
        )
        self._tail = self.tokenizer.encode(ANSWER_LABEL)

    def selection_prompt(self, prompt: str) -> List[int]:
        """Build the prompt used to choose the function.

        Args:
            prompt: The natural language request.

        Returns:
            Token ids ending right where the JSON answer starts.
        """
        request = self.tokenizer.encode(prompt)
        return self._selection_head + request + self._tail

    def arguments_prompt(
        self, prompt: str, function: FunctionDefinition
    ) -> List[int]:
        """Build the prompt used to fill the arguments of a function.

        Args:
            prompt: The natural language request.
            function: The function already chosen by the model.

        Returns:
            Token ids ending right where the JSON answer starts.
        """
        head = self._argument_heads.get(function.name)
        if head is None:
            head = self.tokenizer.encode(
                f"{ARGUMENT_INSTRUCTIONS}\n\nFunction:\n"
                f"{function.signature()}\n\n{EXAMPLES}{REQUEST_LABEL}"
            )
            self._argument_heads[function.name] = head
        return head + self.tokenizer.encode(prompt) + self._tail
