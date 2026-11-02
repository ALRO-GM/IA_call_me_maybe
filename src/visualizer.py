"""Shows, step by step, how the constrained decoder built an answer."""

import os
import sys
from typing import List, TextIO

from pydantic import BaseModel

from src.function_caller import CallTrace

RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"


class Visualizer(BaseModel):
    """Prints a readable trace of each generation.

    For every step it shows whether the text was imposed by the grammar or
    chosen by the model, how many tokens the grammar allowed, how confident
    the model was, the runner-up tokens, and whether the model's favourite
    token had to be vetoed because it would have broken the JSON.

    Attributes:
        color: Use ANSI colours in the output.
    """

    color: bool = False

    @classmethod
    def for_stream(cls, stream: TextIO) -> "Visualizer":
        """Create a visualizer suited to an output stream.

        Args:
            stream: Stream the trace will be written to.

        Returns:
            A visualizer using colours only on an interactive terminal.
        """
        interactive = stream.isatty() and "NO_COLOR" not in os.environ
        return cls(color=interactive)

    def _paint(self, text: str, style: str) -> str:
        """Wrap text in an ANSI style when colours are enabled.

        Args:
            text: Text to style.
            style: ANSI escape sequence.

        Returns:
            The styled (or untouched) text.
        """
        return f"{style}{text}{RESET}" if self.color else text

    def render(self, prompt: str, trace: CallTrace) -> str:
        """Format the trace of one function call.

        Args:
            prompt: The natural language request.
            trace: Trace returned by the function caller.

        Returns:
            A multi-line, human readable description.
        """
        lines: List[str] = [
            "",
            self._paint(f"Prompt: {prompt}", BOLD),
            "  step  source   allowed  chosen token      p(chosen)  "
            "other candidates",
        ]
        vetoed = 0
        pieces: List[str] = []
        for number, step in enumerate(trace.steps, start=1):
            token = repr(step.text)
            if step.forced:
                lines.append(self._paint(
                    f"  {number:>4}  grammar  {step.allowed:>7}  {token}",
                    DIM,
                ))
                pieces.append(self._paint(step.text, DIM))
                continue
            confidence = step.alternatives[0][1] if step.alternatives else 0
            others = "  ".join(
                f"{text!r} {share:.1%}"
                for text, share in step.alternatives[1:]
            )
            lines.append(
                f"  {number:>4}  model    {step.allowed:>7}  "
                f"{token:<16}  {confidence:>8.1%}  {others}"
            )
            if not step.unconstrained_valid:
                vetoed += 1
                lines.append(self._paint(
                    f"        ^ the model preferred "
                    f"{step.unconstrained!r}, forbidden by the grammar",
                    YELLOW,
                ))
            colour = GREEN if step.unconstrained_valid else YELLOW
            pieces.append(self._paint(step.text, colour))
        lines.append("  JSON: " + "".join(pieces))
        lines.append(self._paint(
            f"  {trace.generated_tokens} tokens, {trace.model_calls} model "
            f"calls, {vetoed} choice(s) corrected by the grammar",
            CYAN,
        ))
        if self.color:
            lines.append(
                f"  legend: {self._paint('imposed by the grammar', DIM)} | "
                f"{self._paint('chosen by the model', GREEN)} | "
                f"{self._paint('model overruled', YELLOW)}"
            )
        return "\n".join(lines)

    def show(self, prompt: str, trace: CallTrace) -> None:
        """Print the trace of one function call on standard error.

        Args:
            prompt: The natural language request.
            trace: Trace returned by the function caller.
        """
        print(self.render(prompt, trace), file=sys.stderr)
