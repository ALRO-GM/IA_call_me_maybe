"""Byte-level automata describing which output is allowed.

A JSON schema without recursion describes a *regular* language, so the
expected output can be followed with a plain state machine. A ``Grammar``
is a small tree of nodes, one class per kind of JSON fragment:

* ``LiteralNode``   fixed text such as ``{"a": `` (keys, braces, commas);
* ``EnumNode``      one text out of a closed list (function names, booleans);
* ``StringNode``    the inside of a JSON string up to the closing quote;
* ``NumberNode``    a JSON number, optionally restricted to integers;
* ``SequenceNode``  several nodes one after another (a JSON object);
* ``ArrayNode``     ``[item, item, ...]`` with a bounded number of items.

Every node class answers the same few questions about its own fragment
(``Node`` documents them), and ``Grammar`` combines the answers without
knowing which kind of node it is talking to.

The position inside the grammar is a ``State``: a stack of frames, each one
being ``(node index, local state)``. States are plain tuples of integers, so
they are immutable, cheap to copy and usable as dictionary keys for caching.

``Grammar.advance(state, byte)`` returns the next state, or ``None`` when
the byte is not allowed there. A token is valid when all of its bytes can be
consumed one after another.
"""

import json
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict

from src.models import FunctionDefinition, ParameterSpec

Local = Tuple[int, ...]
Frame = Tuple[int, Local]
State = Tuple[Frame, ...]
Part = Union[bytes, int]

DONE: Local = ()
"""Local state of a node that has matched its whole fragment."""

# What ``Node.text_mode`` can answer.
NOT_TEXT = 0      # not inside the free text of a string
TEXT_START = 1    # at the first character of a string
TEXT_INSIDE = 2   # between two characters of a string
TEXT_PARTIAL = 3  # in the middle of an escape or of a UTF-8 character

# Local states of a string node: (mode, a, b, c).
STR_CONTENT = 0   # plain characters
STR_ESCAPE = 1    # just after a backslash
STR_HEX = 2       # inside \uXXXX, ``a`` hex digits still expected
STR_UTF8 = 3      # ``a`` more bytes expected, the next one in [b, c]
STR_FIRST = 4     # first character: like content, but not a space
STRING_START: Local = (STR_FIRST, 0, 0, 0)
STRING_CONTENT: Local = (STR_CONTENT, 0, 0, 0)

# Local states of a number node: (phase, characters consumed).
NUM_START = 0       # nothing read yet
NUM_MINUS = 1       # after "-"
NUM_ZERO = 2        # integer part is exactly "0"
NUM_INT = 3         # inside the integer part
NUM_DOT = 4         # after ".", a digit is required
NUM_FRAC = 5        # inside the fractional part
NUM_EXP = 6         # after "e", a sign or a digit is required
NUM_EXP_SIGN = 7    # after "e+" / "e-", a digit is required
NUM_EXP_ONE = 8     # one exponent digit read
NUM_EXP_TWO = 9     # two exponent digits read (the maximum)
NUM_ACCEPTING = (NUM_ZERO, NUM_INT, NUM_FRAC, NUM_EXP_ONE, NUM_EXP_TWO)

# Local states of an array node: (phase, items so far).
ARR_OPEN = 0        # expects "["
ARR_FIRST = 1       # expects "]" or the first item
ARR_AFTER_ITEM = 2  # expects "," or "]"
ARR_SPACE = 3       # expects the space that follows a comma
ARR_ITEM = 4        # an item starts here (no byte consumed)

MAX_NUMBER_CHARS = 40
MAX_ARRAY_ITEMS = 32

_QUOTE = ord('"')
_BACKSLASH = ord("\\")
_DIGITS = frozenset(b"0123456789")
_HEX_DIGITS = frozenset(b"0123456789abcdefABCDEF")
# \b and \f are valid JSON escapes but are left out on purpose: a small
# model writes "\b" when it means the regex word boundary "\\b".
_SIMPLE_ESCAPES = frozenset(b'"\\/nrt')
# First byte of a multi-byte UTF-8 character: (lowest, highest, number of
# bytes that follow, lowest and highest value of the next byte). The ranges
# of the next byte reject overlong forms and surrogates.
_UTF8_LEADS = (
    (0xC2, 0xDF, 1, 0x80, 0xBF),
    (0xE0, 0xE0, 2, 0xA0, 0xBF),
    (0xE1, 0xEC, 2, 0x80, 0xBF),
    (0xED, 0xED, 2, 0x80, 0x9F),
    (0xEE, 0xEF, 2, 0x80, 0xBF),
    (0xF0, 0xF0, 3, 0x90, 0xBF),
    (0xF1, 0xF3, 3, 0x80, 0xBF),
    (0xF4, 0xF4, 3, 0x80, 0x8F),
)


class Node(BaseModel):
    """Base class of grammar nodes: one fragment of the expected output.

    A node never stores where the output currently is; it receives its
    *local state* (a tuple of integers) and returns the next one. The
    methods below are the whole interface ``Grammar`` relies on. The
    defaults describe a node that accepts nothing, so each subclass only
    overrides what applies to it.
    """

    model_config = ConfigDict(frozen=True)

    def start(self) -> Local:
        """Return the local state before any byte was read."""
        return (0,)

    def step(self, local: Local, byte: int) -> Optional[Local]:
        """Consume one byte.

        Args:
            local: Current local state.
            byte: Next byte of the output.

        Returns:
            The new local state, ``DONE`` if the byte ends the fragment, or
            ``None`` if this node cannot take the byte.
        """
        return None

    def skip(self, local: Local) -> Optional[Local]:
        """Say what happens when the next byte is not for this node.

        Args:
            local: Current local state.

        Returns:
            ``DONE`` if the fragment is already complete (a number ends
            when something else begins), another local state to try the
            byte from, or ``None`` if the byte is simply an error.
        """
        return None

    def open(self, local: Local) -> Optional[Tuple[Local, int]]:
        """Say whether a child node has to be entered before reading on.

        Args:
            local: Current local state.

        Returns:
            ``None``, or the local state to keep for this node (``DONE``
            when nothing is left after the child) and the child's index.
        """
        return None

    def forced(self, local: Local) -> bytes:
        """Return the bytes that are the only possible continuation.

        Args:
            local: Current local state.

        Returns:
            The compulsory bytes, empty when there is a real choice.
        """
        return b""

    def text_mode(self, local: Local) -> int:
        """Tell whether the position is inside the free text of a string.

        Args:
            local: Current local state.

        Returns:
            One of the ``NOT_TEXT`` / ``TEXT_*`` constants.
        """
        return NOT_TEXT


class LiteralNode(Node):
    """Fixed text. Local state: ``(bytes already matched,)``.

    Attributes:
        text: The bytes to match.
    """

    text: bytes

    def start(self) -> Local:
        """Return the initial state, ``DONE`` for an empty text."""
        return (0,) if self.text else DONE

    def step(self, local: Local, byte: int) -> Optional[Local]:
        """Accept the byte only if it is the next one of the text."""
        position = local[0]
        if self.text[position] != byte:
            return None
        return (position + 1,) if position + 1 < len(self.text) else DONE

    def forced(self, local: Local) -> bytes:
        """Return what is left of the text."""
        return self.text[local[0]:]


class EnumNode(Node):
    """One text out of several. Local state: ``(position, *candidates)``.

    The candidates are the indexes of the options that still match what
    was read. An option may be a prefix of another one.

    Attributes:
        options: The alternatives.
    """

    options: Tuple[bytes, ...]

    def start(self) -> Local:
        """Return the initial state: every option is a candidate."""
        return (0,) + tuple(range(len(self.options)))

    def step(self, local: Local, byte: int) -> Optional[Local]:
        """Keep the candidates whose next byte is the given one."""
        position = local[0]
        alive = tuple(
            index for index in local[1:]
            if len(self.options[index]) > position
            and self.options[index][position] == byte
        )
        if not alive:
            return None
        position += 1
        if any(len(self.options[index]) > position for index in alive):
            return (position,) + alive
        return DONE

    def skip(self, local: Local) -> Optional[Local]:
        """End here if what was read is exactly one of the options."""
        complete = any(
            len(self.options[index]) == local[0] for index in local[1:]
        )
        return DONE if complete else None

    def forced(self, local: Local) -> bytes:
        """Return the rest of the option once a single one is left."""
        if len(local) != 2:
            return b""
        return self.options[local[1]][local[0]:]


class StringNode(Node):
    """The inside of a JSON string, closing quote included.

    The opening quote belongs to the literal before the node. Local state:
    ``(mode, a, b, c)``, see the ``STR_*`` constants.
    """

    def start(self) -> Local:
        """Return the state at the first character."""
        return STRING_START

    def step(self, local: Local, byte: int) -> Optional[Local]:
        """Follow JSON string syntax and UTF-8 encoding byte by byte."""
        mode, pending = local[0], local[1]
        if mode == STR_UTF8:
            if not local[2] <= byte <= local[3]:
                return None
            if pending == 1:
                return STRING_CONTENT
            return (STR_UTF8, pending - 1, 0x80, 0xBF)
        if mode == STR_ESCAPE:
            if byte in _SIMPLE_ESCAPES:
                return STRING_CONTENT
            return (STR_HEX, 4, 0, 0) if byte == ord("u") else None
        if mode == STR_HEX:
            # "d" is refused as first digit so that no lone surrogate
            # (\ud800-\udfff) can be produced.
            if byte not in _HEX_DIGITS or (pending == 4 and byte in b"dD"):
                return None
            if pending == 1:
                return STRING_CONTENT
            return (STR_HEX, pending - 1, 0, 0)
        if byte == _QUOTE:
            return DONE
        if byte == _BACKSLASH:
            return (STR_ESCAPE, 0, 0, 0)
        if byte < 0x20:
            return None  # raw control characters are forbidden in JSON
        if byte == 0x20 and mode == STR_FIRST:
            # Many tokens carry a leading space (" dog"); right after the
            # opening quote that space would wrongly become part of the
            # value, so a string is not allowed to start with one.
            return None
        if byte < 0x80:
            return STRING_CONTENT
        for lowest, highest, count, low, high in _UTF8_LEADS:
            if lowest <= byte <= highest:
                return (STR_UTF8, count, low, high)
        return None

    def text_mode(self, local: Local) -> int:
        """Tell apart the first character, the others, and escapes."""
        if local[0] == STR_FIRST:
            return TEXT_START
        return TEXT_INSIDE if local[0] == STR_CONTENT else TEXT_PARTIAL


class NumberNode(Node):
    r"""A JSON number. Local state: ``(phase, characters consumed)``.

    The grammar is ``-? (0 | [1-9][0-9]*) (\.[0-9]+)? ([eE][+-]?[0-9]+)?``
    limited to ``MAX_NUMBER_CHARS`` characters and two exponent digits, so
    that the value always fits in a float.

    Attributes:
        integer_only: Refuse the fraction and the exponent.
    """

    integer_only: bool = False

    def start(self) -> Local:
        """Return the state before the first character."""
        return (NUM_START, 0)

    def step(self, local: Local, byte: int) -> Optional[Local]:
        """Move to the next phase if the byte can extend the number."""
        phase, count = local
        if count >= MAX_NUMBER_CHARS:
            return None
        is_digit = byte in _DIGITS
        # Opening a fraction or an exponent needs room for what follows.
        roomy = count < MAX_NUMBER_CHARS - 3 and not self.integer_only
        target = -1
        if phase == NUM_START and byte == ord("-"):
            target = NUM_MINUS
        elif phase in (NUM_START, NUM_MINUS) and is_digit:
            target = NUM_ZERO if byte == ord("0") else NUM_INT
        elif phase == NUM_INT and is_digit:
            target = NUM_INT
        elif phase in (NUM_ZERO, NUM_INT) and byte == ord(".") and roomy:
            target = NUM_DOT
        elif phase in (NUM_DOT, NUM_FRAC) and is_digit:
            target = NUM_FRAC
        elif phase in (NUM_ZERO, NUM_INT, NUM_FRAC) and byte in b"eE":
            target = NUM_EXP if roomy else -1
        elif phase == NUM_EXP and byte in b"+-":
            target = NUM_EXP_SIGN
        elif phase in (NUM_EXP, NUM_EXP_SIGN) and is_digit:
            target = NUM_EXP_ONE
        elif phase == NUM_EXP_ONE and is_digit:
            target = NUM_EXP_TWO
        return (target, count + 1) if target >= 0 else None

    def skip(self, local: Local) -> Optional[Local]:
        """End here if the characters read so far form a number.

        A number has no closing character: it ends when a byte that cannot
        belong to it shows up.
        """
        return DONE if local[0] in NUM_ACCEPTING else None


class SequenceNode(Node):
    """Several nodes in a row. Local state: ``(index of the next child,)``.

    Attributes:
        children: Indexes of the child nodes, in output order.
    """

    children: Tuple[int, ...]

    def start(self) -> Local:
        """Return the initial state, ``DONE`` for an empty sequence."""
        return (0,) if self.children else DONE

    def open(self, local: Local) -> Optional[Tuple[Local, int]]:
        """Enter the next child; nothing is kept after the last one."""
        position = local[0]
        more = position + 1 < len(self.children)
        return ((position + 1,) if more else DONE, self.children[position])


class ArrayNode(Node):
    """``[item, item, ...]``. Local state: ``(phase, items so far)``.

    Attributes:
        item: Index of the node matching one item.
    """

    item: int

    def start(self) -> Local:
        """Return the state expecting the opening bracket."""
        return (ARR_OPEN, 0)

    def step(self, local: Local, byte: int) -> Optional[Local]:
        """Read the brackets, commas and spaces around the items."""
        phase, count = local
        if phase == ARR_OPEN and byte == ord("["):
            return (ARR_FIRST, 0)
        if phase in (ARR_FIRST, ARR_AFTER_ITEM) and byte == ord("]"):
            return DONE
        if phase == ARR_AFTER_ITEM and byte == ord(","):
            return (ARR_SPACE, count) if count < MAX_ARRAY_ITEMS else None
        if phase == ARR_SPACE and byte == ord(" "):
            return (ARR_ITEM, count)
        return None

    def skip(self, local: Local) -> Optional[Local]:
        """After ``[``, a byte that is not ``]`` starts the first item."""
        return (ARR_ITEM, 0) if local[0] == ARR_FIRST else None

    def open(self, local: Local) -> Optional[Tuple[Local, int]]:
        """Enter the item node when an item starts."""
        if local[0] != ARR_ITEM:
            return None
        return (ARR_AFTER_ITEM, local[1] + 1), self.item


class Grammar(BaseModel):
    """A compiled grammar: a list of nodes and the index of the root.

    Attributes:
        nodes: Every node of the tree; children are referenced by index.
        root: Index of the node the output must match.
    """

    model_config = ConfigDict(frozen=True)

    nodes: Tuple[Node, ...]
    root: int

    def _settle(self, state: State) -> State:
        """Apply every transition that needs no input byte.

        Finished nodes are popped and nodes waiting for a child enter it,
        until the top frame is a node that reads bytes.

        Args:
            state: A state possibly resting on a finished node.

        Returns:
            The equivalent state that is ready to read a byte.
        """
        while state:
            index, local = state[-1]
            state = state[:-1]
            if local == DONE:
                continue
            opened = self.nodes[index].open(local)
            if opened is None:
                return state + ((index, local),)
            kept, child = opened
            if kept != DONE:
                state += ((index, kept),)
            state += ((child, self.nodes[child].start()),)
        return state

    def initial_state(self) -> State:
        """Return the state before any byte has been read."""
        return self._settle(((self.root, self.nodes[self.root].start()),))

    def advance(self, state: State, byte: int) -> Optional[State]:
        """Consume one byte.

        The byte is offered to the node on top of the stack. If that node
        does not take it but may be skipped, the byte is offered again to
        whatever comes next.

        Args:
            state: Current state.
            byte: Next byte of the output.

        Returns:
            The next state, or ``None`` when the byte is not allowed.
        """
        while state:
            index, local = state[-1]
            node = self.nodes[index]
            updated = node.step(local, byte)
            consumed = updated is not None
            if updated is None:
                updated = node.skip(local)
            if updated is None:
                return None
            state = self._settle(state[:-1] + ((index, updated),))
            if consumed:
                return state
        return None

    def is_complete(self, state: State) -> bool:
        """Tell whether the output read so far is a full match.

        Args:
            state: Current state.

        Returns:
            True when nothing more is required.
        """
        while state:
            index, local = state[-1]
            if self.nodes[index].skip(local) != DONE:
                return False
            state = self._settle(state[:-1])
        return True

    def text_mode(self, state: State) -> int:
        """Tell whether the position is inside the free text of a string.

        Args:
            state: Current state.

        Returns:
            One of the ``NOT_TEXT`` / ``TEXT_*`` constants.
        """
        if not state:
            return NOT_TEXT
        index, local = state[-1]
        return self.nodes[index].text_mode(local)

    def forced_bytes(self, state: State) -> bytes:
        """Return the bytes that are the only possible continuation.

        Inside a literal, or once a single enum option is left, the output
        is fully determined and the model does not need to be asked.

        Args:
            state: Current state.

        Returns:
            The compulsory bytes, possibly empty.
        """
        forced = b""
        while state:
            index, local = state[-1]
            chunk = self.nodes[index].forced(local)
            if not chunk:
                break
            forced += chunk
            state = self._settle(state[:-1])
        # The start of a value is left to the model, because the vocabulary
        # glues it to what follows: " -", " [" and " \"*" are single tokens.
        # Imposing the space (or the opening quote of a string) on its own
        # would leave the model in front of a sequence it has never seen,
        # and it could not write a minus sign or a lone symbol.
        if forced.endswith(b' "') and self.text_mode(state) == TEXT_START:
            return forced[:-2]
        return forced[:-1] if forced.endswith(b" ") else forced

    def run(self, state: State, data: bytes) -> Optional[State]:
        """Consume several bytes in a row.

        Args:
            state: Starting state.
            data: Bytes to consume.

        Returns:
            The final state, or ``None`` if some byte is not allowed.
        """
        current: Optional[State] = state
        for byte in data:
            if current is None:
                return None
            current = self.advance(current, byte)
        return current

    def matches(self, data: bytes) -> bool:
        """Tell whether a whole byte string matches the grammar.

        Args:
            data: Candidate output.

        Returns:
            True if ``data`` is a complete, valid output.
        """
        final = self.run(self.initial_state(), data)
        return final is not None and self.is_complete(final)


def json_text(value: object) -> bytes:
    """Serialise a value as compact ASCII-only JSON bytes.

    Args:
        value: Any JSON serialisable value.

    Returns:
        The JSON text encoded as bytes.
    """
    return json.dumps(value, ensure_ascii=True).encode("ascii")


class GrammarBuilder(BaseModel):
    """Incrementally builds the node list of a ``Grammar``.

    Attributes:
        nodes: Nodes created so far.
    """

    nodes: List[Node] = []

    def add(self, node: Node) -> int:
        """Store a node and return its index.

        Args:
            node: Node to store.

        Returns:
            Index of the node inside ``nodes``.
        """
        self.nodes.append(node)
        return len(self.nodes) - 1

    def parts(self, spec: ParameterSpec) -> List[Part]:
        """Translate a type description into literal bytes and node indexes.

        Returning loose parts (instead of one node) lets the caller glue
        neighbouring literals together, e.g. a key and the opening quote of
        its string value, so that forced text is tokenized naturally.

        Args:
            spec: Type description.

        Returns:
            The parts, in output order.
        """
        if spec.enum is not None:
            options = tuple(json_text(option) for option in spec.enum)
            return [self.add(EnumNode(options=options))]
        kind = spec.type
        if kind == "string":
            return [b'"', self.add(StringNode())]
        if kind in ("number", "integer"):
            return [self.add(NumberNode(integer_only=kind == "integer"))]
        if kind == "boolean":
            return [self.add(EnumNode(options=(b"true", b"false")))]
        if kind == "null":
            return [b"null"]
        if kind == "array":
            item = self.sequence(
                self.parts(spec.items or ParameterSpec(type="string"))
            )
            return [self.add(ArrayNode(item=item))]
        return self.object_parts(spec.properties or {})

    def object_parts(self, properties: Dict[str, ParameterSpec]) -> List[Part]:
        """Translate an object with fixed keys into parts.

        Args:
            properties: Keys and their types, in output order.

        Returns:
            The parts spelling ``{"key": value, ...}``.
        """
        parts: List[Part] = [b"{"]
        for position, (key, spec) in enumerate(properties.items()):
            separator = b", " if position else b""
            parts.append(separator + json_text(key) + b": ")
            parts.extend(self.parts(spec))
        parts.append(b"}")
        return parts

    def sequence(self, parts: List[Part]) -> int:
        """Turn parts into a single node, merging adjacent literals.

        Args:
            parts: Literal bytes and node indexes.

        Returns:
            Index of a node matching the parts in order.
        """
        children: List[int] = []
        pending = b""
        for part in parts + [b""]:  # the empty part flushes the last text
            if isinstance(part, bytes) and part:
                pending += part
                continue
            if pending:
                children.append(self.add(LiteralNode(text=pending)))
                pending = b""
            if isinstance(part, int):
                children.append(part)
        if len(children) == 1:
            return children[0]
        return self.add(SequenceNode(children=tuple(children)))

    def build(self, parts: List[Part]) -> Grammar:
        """Finish the grammar.

        Args:
            parts: Parts the whole output must match, in order.

        Returns:
            The compiled grammar.
        """
        root = self.sequence(parts)
        return Grammar(nodes=tuple(self.nodes), root=root)


NAME_PREFIX = b'{"name": "'
PARAMETERS_KEY = b', "parameters": '


def function_name_grammar(names: List[str]) -> Grammar:
    """Build the grammar of the first half of the answer.

    It matches ``{"name": "<one of the names>"``. The closing quote is part
    of every option, which tells apart names that are prefixes of others.

    Args:
        names: Names of the available functions.

    Returns:
        The grammar selecting a function.
    """
    builder = GrammarBuilder()
    options = tuple(json_text(name)[1:] for name in names)
    choice = builder.add(EnumNode(options=options))
    return builder.build([NAME_PREFIX, choice])


def arguments_grammar(function: FunctionDefinition) -> Grammar:
    """Build the grammar of the second half of the answer.

    It matches ``, "parameters": {...}}`` for the given function, with
    every argument present, in order and of the declared type.

    Args:
        function: The function selected by the model.

    Returns:
        The grammar generating the arguments.
    """
    builder = GrammarBuilder()
    parts: List[Part] = [PARAMETERS_KEY]
    parts.extend(builder.object_parts(function.parameters))
    parts.append(b"}")
    return builder.build(parts)


def value_grammar(spec: ParameterSpec) -> Grammar:
    """Build a grammar matching a single value of the given type.

    Args:
        spec: Type description.

    Returns:
        The grammar of one value.
    """
    builder = GrammarBuilder()
    return builder.build(builder.parts(spec))
