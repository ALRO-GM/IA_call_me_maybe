*This project has been created as part of the 42 curriculum by <your_42_login>.*

# call me maybe

Introduction to function calling in LLMs: turning a sentence into a typed function call
with a 0.6B-parameter model and **constrained decoding**.

## Contents

1. [Description](#description)
2. [Instructions](#instructions)
3. [Example usage](#example-usage)
4. [Program flow](#program-flow)
5. [Algorithm explanation](#algorithm-explanation)
6. [Design decisions](#design-decisions)
7. [Development steps](#development-steps)
8. [Challenges faced](#challenges-faced)
9. [Optimizations](#optimizations)
10. [Code review and refactoring](#code-review-and-refactoring)
11. [Performance analysis](#performance-analysis)
12. [Testing strategy](#testing-strategy)
13. [Bonus features](#bonus-features)
14. [Resources](#resources)

## Description

Given a list of available functions and a list of natural language requests, the program
answers each request with the function to call and its arguments. It never answers the
question itself.

```
"What is the sum of 2 and 3?"
        -> {"prompt": "What is the sum of 2 and 3?",
            "name": "fn_add_numbers",
            "parameters": {"a": 2.0, "b": 3.0}}
```

A small model asked politely for JSON often produces something that does not parse. Here
the JSON is valid **by construction**: at every generation step, all tokens that would
break the JSON syntax or the schema of the chosen function are removed before the next
token is picked. The model only decides *which function* and *which values*; it is never
able to decide *whether the output is valid*.

Features:

- constrained decoding driven by a byte-level automaton built from the function schema;
- function chosen by the LLM among the declared names only (no keyword heuristics);
- types: `number`, `integer`, `string`, `boolean`, `null`, enums, and nested `array` /
  `object` arguments;
- a byte-level tokenizer written from scratch (no `encode` / `decode` from the SDK);
- only two SDK methods used: `get_logits_from_input_ids` and `get_path_to_vocab_file`;
- works with other byte-level BPE models (`--model`);
- step-by-step visualization of the generation (`--visualize`);
- tolerant input loading, retries, atomic and incremental output writing;
- 311 tests.

## Instructions

Requirements: Python 3.10 or later and [uv](https://docs.astral.sh/uv/).

```bash
make install        # uv sync: creates .venv and installs everything
make run            # runs with the default files
```

The first run downloads the model (about 1.5 GB) from Hugging Face.

The program is run as a module, with three optional paths:

```bash
uv run python -m src \
    [--functions_definition <file>] [--input <file>] [--output <file>]
```

| Option | Default | Meaning |
|---|---|---|
| `--functions_definition` | `data/input/functions_definition.json` | available functions |
| `--input` | `data/input/function_calling_tests.json` | prompts to process |
| `--output` | `data/output/function_calling_results.json` | file to write |
| `--model` | `Qwen/Qwen3-0.6B` | Hugging Face model to load |
| `--device` | `auto` | `auto`, `cpu`, `cuda` or `mps` |
| `--visualize` | off | print every decoding step |
| `--max_new_tokens` | `400` | generated tokens allowed per prompt |
| `--max_string_tokens` | `96` | tokens after which a string value is closed |
| `--threads` | `0` | CPU threads of the model; `0` = only the performance cores of a hybrid CPU, otherwise the library default. An `OMP_NUM_THREADS` set by the user always wins |

Other Makefile rules:

| Rule | Action |
|---|---|
| `make debug` | run the program under `pdb` |
| `make test` | run the unit tests (`RUN_SLOW=1 make test` adds the real-model tests) |
| `make lint` | `flake8` and `mypy` with the mandatory flags |
| `make lint-strict` | `flake8` and `mypy --strict` |
| `make clean` | remove caches |

`make run ARGS="--visualize"` forwards options to the program.

**CPU or GPU.** With `--device auto` the SDK picks the device itself: Apple GPU (`mps`),
then NVIDIA GPU (`cuda`), then CPU. An NVIDIA card is only used when its proprietary
driver is installed; with the open-source `nouveau` driver PyTorch does not see it. If
the model cannot be loaded on the chosen device, the program prints a warning and loads
it on the CPU instead. Note that the provided SDK needs the `accelerate` package to load
on `cuda`, and does not declare it: without it a CUDA machine also ends up on the CPU.
All figures in this document were measured on CPU.

Progress and warnings go to standard error; the exit status is `0` on success, `1` on a
fatal error (unreadable input, model not loadable, output not writable) and `130` when
interrupted. No error ever produces a Python traceback.

## Example usage

```console
$ uv run python -m src
5 function(s), 11 prompt(s) loaded
using 8 CPU thread(s)
loading model Qwen/Qwen3-0.6B (device: auto) ...
model ready in 4.0s
[1/11]   2.1s  fn_add_numbers({"a": 2.0, "b": 3.0})
[2/11]   3.2s  fn_add_numbers({"a": 265.0, "b": 345.0})
[3/11]   1.5s  fn_greet({"name": "shrek"})
...
[10/11]   4.9s  fn_substitute_string_with_regex({"source_string": "Programming is fun", "regex": "([aeiou])", "replacement": "*"})
[11/11]   6.6s  fn_substitute_string_with_regex({"source_string": "The cat sat on the mat with another cat", "regex": "cat", "replacement": "dog"})
done: 11 call(s) in 35.0s (114 model calls, 0 from cache, 0 failure(s))
Results written to data/output/function_calling_results.json
```

Custom files, including nested arguments:

```console
$ uv run python -m src \
      --functions_definition tests/data/extended_functions.json \
      --input tests/data/extended_prompts.json \
      --output data/output/extended.json
...
[13/20]  10.8s  fn_create_user({"user": {"name": "Alice", "age": 30, "admin": true}, "tags": ["dev", "ops"]})
```

Watching the decoder work:

```console
$ uv run python -m src --visualize
Prompt: What is the sum of 265 and -3.5?
  step  source   allowed  chosen token      p(chosen)  other candidates
     1  grammar        1  '{"name": "'
     2  model          2  'fn'                100.0%  'f' 0.0%
     3  model         16  '_add'              100.0%  '_sub' 0.0%  '_g' 0.0%  '_get' 0.0%
     4  grammar        1  '_numbers"'
     5  grammar        1  ', "parameters": {"a":'
     6  model          2  ' '                 100.0%  ' -' 0.0%
     7  model         11  '2'                 100.0%  '3' 0.0%  '1' 0.0%  '4' 0.0%
    ...
    12  model          2  ' -'                100.0%  ' ' 0.0%
    13  model         10  '3'                 100.0%  '1' 0.0%  '0' 0.0%  '2' 0.0%
  JSON: {"name": "fn_add_numbers", "parameters": {"a": 265, "b": -3.5}}
  28 tokens, 12 model calls, 0 choice(s) corrected by the grammar
```

`allowed` is the number of tokens (out of about 151,000) that the grammar accepted at
that step. When the model's favourite token is not among them, the line is followed by
a note saying which token was vetoed.

Errors:

```console
$ uv run python -m src --input missing.json
error: input file not found: missing.json
$ uv run python -m src --functions_definition broken.json
error: functions definition file is not valid JSON: broken.json (line 2, column 1: Expecting value)
```

## Program flow

```
 command line
      |
      v
 parse_arguments ............ Settings (pydantic)                    __main__.py
      |
      v
 load_functions, load_prompts ... validate, skip bad entries         io_utils.py, models.py
      |
      v
 write an empty result ...... fails here if the output is unwritable
      |
      v
 configure_threads .......... performance cores of a hybrid CPU     hardware.py
      |
      v
 load_model ................. SDK model (GPU if usable, else CPU)    tokenizer.py
      |                       + our Tokenizer built from vocab.json
      v
 build once: TokenIndex (byte trie), PromptBuilder, name grammar     decoder.py, grammar.py
      |
      v
 for each prompt ------------------------------------------------------------------+
      |                                                                            |
      |  STAGE 1 - which function?                               function_caller.py |
      |    selection prompt (all functions) -> token ids                           |
      |    grammar:  {"name": "  +  one of the declared names                      |
      |    constrained decoding  ->  {"name": "fn_add_numbers"                     |
      |                                                                            |
      |  STAGE 2 - which arguments?                                                |
      |    arguments prompt (only that function + 2 examples) -> token ids         |
      |    grammar built from the function's parameters                            |
      |    constrained decoding  ->  , "parameters": {"a": 2, "b": 3}}             |
      |                                                                            |
      |  join both halves -> json.loads -> check types with pydantic               |
      |  on failure: retry once with stricter limits, else skip with a warning     |
      |  append the result and rewrite the output file (atomic)                    |
      +----------------------------------------------------------------------------+
      |
      v
 summary on stderr, exit status
```

The constrained decoding loop used by both stages (`ConstrainedDecoder.generate`):

```
 state = grammar.initial_state()
 while state is not finished:
     forced = grammar.forced_bytes(state)        # only one possible continuation?
     if forced:
         append its tokens                       # no model call
     else:
         allowed = token_index.valid_ids(grammar, state)   # trie walk, cached
         logits  = model.get_logits_from_input_ids(context + output)  # via LogitsCache
         logits[every token not in allowed] = -inf
         append argmax(logits)
     state = grammar.run(state, bytes of what was appended)
```

| Module | Role |
|---|---|
| `src/__main__.py` | command line, orchestration, error messages, exit status |
| `src/models.py` | pydantic models of the inputs, the output and the settings; each type knows how to describe itself to the model and how to check a decoded value |
| `src/io_utils.py` | reading the two input files, writing the result atomically, messages on stderr |
| `src/tokenizer.py` | byte-level tokenizer: `encode`, `decode`, byte trie of the vocabulary |
| `src/grammar.py` | one class per kind of JSON fragment; `Grammar` follows them byte by byte |
| `src/decoder.py` | `TokenIndex` (allowed tokens per state), `LogitsCache` (model calls already made) and `ConstrainedDecoder` (masking loop) |
| `src/hardware.py` | number of CPU threads given to the model |
| `src/prompt_builder.py` | selection prompt and arguments prompt |
| `src/function_caller.py` | the two stages, the retry and the final validation |
| `src/visualizer.py` | `--visualize` output |

## Algorithm explanation

### 1. The output is a regular language

Every answer has the same shape:

```
{"name": "<function>", "parameters": {"<arg1>": <value1>, "<arg2>": <value2>}}
```

Once the function is known, keys, order and types are fixed. A JSON schema without
recursion describes a *regular* language, so it can be followed by a finite state
machine. `src/grammar.py` compiles a schema into a small tree of nodes:

| Node class | Matches |
|---|---|
| `LiteralNode` | fixed text: braces, keys, commas |
| `EnumNode` | one text out of a closed list: function names, booleans, enum values |
| `StringNode` | the inside of a JSON string up to the closing quote, with escapes and UTF-8 checked byte by byte |
| `NumberNode` | `-? (0 \| [1-9][0-9]*) (\.[0-9]+)? ([eE][+-]?[0-9]+)?`, optionally integers only |
| `SequenceNode` | several nodes in a row (a JSON object with fixed keys) |
| `ArrayNode` | `[item, item, ...]`, bounded to 32 items |

All of them derive from `Node` and answer the same five questions about their own
fragment; `Grammar` combines the answers without knowing which class it is talking to:

| Method | Question |
|---|---|
| `start()` | what is your state before any byte? |
| `step(local, byte)` | can you take this byte, and what is your state after it? |
| `skip(local)` | if the byte is not for you, are you complete anyway? (a number is; a string is not) |
| `open(local)` | do you have to enter a child node now? (sequences and arrays) |
| `forced(local)` | is there only one possible continuation, and which? (literals) |

The position in the grammar is a **state**: a stack of `(node, local state)` frames,
stored as plain tuples of integers. `Grammar.advance(state, byte)` returns the next
state, or `None` if that byte is not allowed there.

The automaton works on **bytes**, not characters, because tokens are byte sequences: one
token may hold half of a UTF-8 character.

### 2. From allowed bytes to allowed tokens

A token is allowed if the automaton can consume all its bytes. `TokenIndex`
(`src/decoder.py`) stores the vocabulary as a **byte trie** and walks the trie and the
automaton together: when a byte is refused, every token sharing that prefix is discarded
at once. Inside a number only the few branches made of digits, signs and separators are
visited.

The one place where almost every token is valid is the inside of a string. That set is
computed once at start-up; only the 6,192 tokens containing a quote, a backslash or a
cut UTF-8 character need the trie walk. Results are cached per state.

### 3. Masking the logits

For each step:

1. if the grammar leaves a single possibility (inside a literal, or once one function
   name is left), that text is appended **without calling the model**;
2. otherwise the model returns its logits for the whole vocabulary;
3. the logits of all forbidden tokens are set to `-inf`;
4. the best remaining token is appended, and the automaton advances over its bytes.

Decoding is greedy (arg-max), so runs are deterministic.

### 4. Two stages

1. **Function selection.** The prompt lists every function. The grammar is
   `{"name": "` followed by an enum of the declared names. The model picks the name
   token by token; an unknown name is impossible. The closing quote is part of each
   option, which separates names that are prefixes of each other (`fn_add` /
   `fn_add_numbers`).
2. **Arguments.** The grammar is built from the chosen function's parameters. The
   prompt is rebuilt to show only that function plus two worked examples.

The two generated halves are concatenated, parsed with `json.loads`, and re-validated
against the schema with pydantic before being written. That last check is independent
from the decoder: it would catch a bug in the automaton.

### 5. Home-made tokenizer

`src/tokenizer.py` rebuilds the tokenizer from `vocab.json` alone, the only vocabulary
file the subject's SDK methods give access to. Each entry is decoded with the GPT-2
byte alphabet (`Ġ` is a space, `Ċ` a newline) into the raw bytes it stands for. The
project code never calls the SDK's `encode` or `decode`. The tokenizer is needed in both
directions by the decoder: *decoding* gives the bytes of each candidate token (to test
them against the automaton), *encoding* turns the prompt and the forced text into ids.

Encoding splits the text into words with the model's splitting rule, then cuts each
word by greedy **longest match** over the vocabulary, using the same byte trie as the
decoder. This is not exact BPE: BPE needs the merge table (`merges.txt`), which the two
SDK methods the program allows itself do not provide.

## Design decisions

- **Bytes, not characters.** Correct handling of multi-byte characters and of tokens
  that cut them; invalid UTF-8 cannot be generated.
- **Immutable integer states.** States are hashable, so the allowed-token sets can be
  cached and reused across prompts.
- **Forced text is not generated.** Structure (keys, braces, the rest of a function
  name) is appended directly; the model is asked only where there is a real choice.
- **The start of a value is left to the model.** In this vocabulary ` -`, ` [` and
  ` "*` are single tokens, so neither the space before a value nor the opening quote of
  a string is forced.
- **A string cannot start with a space.** The trade-off is that a value that really
  starts with a space cannot be produced.
- **`\b` and `\f` escapes are disabled**, although they are valid JSON.
- **Numbers are bounded** to 40 characters and a two-digit exponent, so they always fit
  in a float and generation always terminates. Leading zeros are refused, as JSON does.
- **`number` is written as a float** (`2.0`), like the subject's example; `integer`
  stays an integer.
- **Few-shot examples use functions that do not exist** in the input, so nothing is
  hardcoded for the provided tests and the examples cannot leak into the answer (their
  names are not in the grammar).
- **Only the two SDK methods named by the subject.** The provided SDK also has public
  methods returning `merges.txt` and `tokenizer.json`; the subject does not list them,
  so the program does not call them. The price: no exact BPE and no chat control tokens
  (their ids are only in `tokenizer.json`), so the prompt is plain text
  (`Request: ... Answer: `). Measured cost against a version that used them: nothing on
  the provided set (11 / 11 both ways), one prompt on the extended set (17 -> 16 of 19)
  and one symbol on the 26-symbol test (24 -> 23). That version's BPE and chat-layout
  code was then deleted, so nothing in the repository depends on those files.
- **One class per kind of JSON fragment.** See
  [Code review and refactoring](#code-review-and-refactoring).
- **A failed prompt is skipped, not invented.** After one retry with stricter limits the
  prompt is left out with a warning; the output never contains an unvalidated entry.
- **pydantic everywhere.** Every class is a pydantic model, the grammar nodes included; no custom exception class
  is defined (built-in `OSError`, `ValueError` and `RuntimeError` are used).

The reasons behind the string and number rules are in
[Challenges faced](#challenges-faced).

## Development steps

The project was built in this order. Each step ended with a check before moving on.

1. **Reading the subject.** The 20 pages were condensed into a short requirements list.
   Six contradictions were found and settled (output file name, `output/` versus
   `data/output/`, `function`/`arguments` versus `name`/`parameters`, single-dash flags,
   and `encode` returning a 2-D tensor).
2. **Environment.** `uv` installed; `pyproject.toml` with `numpy` and `pydantic`, the
   provided `llm_sdk` as a local path dependency, and `flake8`, `mypy`, `pytest` as
   development tools. `.flake8` and the mypy configuration exclude `llm_sdk` and `.venv`.
3. **Exploring the SDK before designing anything.** A throw-away script measured what
   the design depends on:

   | Context length | 50 tokens | 200 tokens | 400 tokens | 800 tokens |
   |---|---|---|---|---|
   | One `get_logits_from_input_ids` call | 0.23 s | 0.62 s | 1.09 s | 2.11 s |

   It also showed that the vocabulary has 151,643 entries, that the control tokens
   (`<|im_start|>`, `<think>`...) are not in `vocab.json` but in `tokenizer.json` with
   ids from 151,643, that the model returns 151,936 logits, and that numbers are
   tokenized one digit at a time. Conclusion: the time budget is spent in model calls,
   and each call costs in proportion to the prompt length.
4. **Data models and file handling**: pydantic models, loaders that report each kind of
   problem separately, atomic writer.
5. **Tokenizer**, compared with the SDK's on sample texts until identical.
6. **Grammar automata**, checked against tables of valid and invalid values with no
   model involved.
7. **Token index and decoder.**
8. **First end-to-end run**: valid JSON and the right function for all 11 provided
   prompts at the first attempt, in 72 s, with 9 of 11 argument sets right.
9. **Quality work.** A second data set was written (11 other functions with integers,
   booleans, enums, arrays, nested objects) so that changes were never judged on the
   provided examples alone. Failures were diagnosed by printing, for each step, the
   allowed tokens and the model's probabilities.
10. **Speed work**: the per-function arguments prompt.
    A later systematic test (the same sentence with 26 different symbols) exposed one
    more decoder bug, described in challenge 4.
    Last, the SDK surface was reduced to the two methods the subject names, and every
    data set was run again in that configuration.
13. **Code review and refactoring**, with the outputs of the three data sets compared
    byte for byte before and after
    (see [Code review and refactoring](#code-review-and-refactoring)).
11. **Command line, error recovery and the visualizer** (the debugging trace of step 9,
    cleaned up).
12. **Tests and final checks**: the test suite, `flake8`, `mypy --strict`, the unit tests
    under Python 3.10, a run from a clean copy with only `uv sync`, three other models,
    and a run with the SDK reduced to `vocab.json` only.

## Challenges faced

Each entry gives the symptom, the cause that was found, and the fix.

**1. Tokens do not line up with JSON.**
`",`, `"}}` and ` -` are single tokens that span two grammar elements. A decoder that
handles one JSON element at a time cannot use them, and those are the tokens the model
prefers.
*Fix:* the automaton keeps consuming bytes across node boundaries, so a token may close
a string and start the following literal. The forced text that comes next simply starts
from wherever the token stopped.

**2. A number has no end marker.**
A string ends with a quote; a number ends only when something else begins.
*Fix:* number nodes have "accepting" states. On a byte that cannot extend the number,
the node is popped and the same byte is offered to what follows.

**3. Negative numbers lost their sign and arrays came out empty.**
`3.5 times -2` gave `"y": 2`, and "the average of 4, 8, 15..." gave `"values": []`.
The trace showed the model 99.9 % sure of `2`. The cause was the decoder itself: it
forced the text `"y": ` including the final space, but in this vocabulary ` -` and ` [`
are single tokens. After a lone space token, a minus sign is a sequence the model has
never seen.
*Fix:* the space before a value is no longer forced; the model chooses between ` `,
` -`, ` [`... The extended set went from 15 to 17 correct out of 19, for one extra
model call per numeric argument.

**4. Symbols could not be written: `"asterisk"` instead of `"*"`.**
"Replace all vowels ... with asterisks" gave `"replacement": "asterisk"`. It looked like
a limit of the model, and four prompt variants changed nothing. A systematic test showed
otherwise: the same sentence with 26 different symbols was right only 7 times, and it
failed even when the symbol was typed literally (`*` gave `.*`, `#` gave `##`, `'-'`
gave the Unicode minus `−`). The 7 successes were the symbols whose text is tokenized
as ` "` followed by something else. For the others the vocabulary merges the opening
quote with the symbol: `"*"` is ` "*` + `"`, `"-"` is the single token ` "-"`. The
decoder forced the opening quote ` "` on its own, so the natural token was out of reach.
It is the same mistake as challenge 3, one character further.
*Fix:* the opening quote of a string is not forced either; the model chooses between
` "`, ` "*`, ` "-"`... Result: 7 -> 24 correct out of 26 (23 in the final configuration,
see challenge 9), and the provided set went from 10 to 11 out of 11. Cost: one more model call per string argument (106 -> 114 calls).

**5. A regex escape silently became a backspace.**
The model wrote `"(\b\\d+\\b)"`. `\b` is a legal JSON escape, so the output was valid,
but it decodes to a backspace character instead of the regex word boundary.
*Fix:* `\b` and `\f` are removed from the allowed escapes. After a backslash the model
must now write `\\`, `\"`, `\/`, `\n`, `\r`, `\t` or `\u`.

**6. Values starting with a space.**
`"replacement": " *"`: many tokens carry a leading space, and right after the opening
quote the model sometimes picks one.
*Fix:* the first byte of a string may not be a space.

**7. The model described a regex instead of writing one, then copied the example.**
`"regex": "numbers"` for "replace all numbers". Adding rules to the instructions did
give a pattern, but the one with the bug of challenge 5.
*Fix:* two worked examples in the arguments prompt, using functions that are not in the
input. The example itself then had to be tuned, because a 0.6B model imitates it
closely: with the pattern `\\d` in the example, every regex started with a backslash
class (vowels became `\\w`); with `[0-9]`, "numbers" was answered with `[0-9]`, one
digit at a time. The final example uses `[a-z]` on a sentence about letters, which
shows the format without overlapping the usual requests. The provided prompts now give
`\\d+`, `([aeiou])` and `cat`.

**8. Reproducing the model's word splitting.**
The model's tokenizer first splits text into words with a rule that uses the Unicode
classes `\p{L}` and `\p{N}`, which Python's `re` does not have. A first version with
`\d` and `\w` split `½` and `²` differently.
*Fix:* the rule was rewritten as a small scanner based on `unicodedata.category`. While
the project still had the merge table (see challenge 9), the complete tokenizer gave
exactly the SDK's ids on 4,000 random strings, which validated the scanner.

**9. The best tokenizer needs files the subject does not offer.**
Exact BPE needs `merges.txt`, and the chat layout needs the ids of `<|im_start|>` and
`<think>`, which are not in `vocab.json` but in `tokenizer.json`. The provided SDK has
public methods for both files, but the subject lists neither.
*Fix:* the program restricts itself to `get_path_to_vocab_file` and
`get_logits_from_input_ids`. Encoding uses longest match and the prompt is plain text.
The whole evaluation was repeated to measure what that costs (see Design decisions),
and the BPE and chat code was removed.

**10. `flake8 .` failed on code that is not ours.**
The provided `llm_sdk` does not follow flake8, and may not be modified.
*Fix:* it is excluded in `.flake8` and in the mypy configuration.

**11. Too slow with many functions.**
The extended set took 232 s for 20 prompts, close to the 5-minute limit.
*Fix:* see [Optimizations](#optimizations).

**12. The SDK cannot load the model on an NVIDIA GPU as shipped.**
Asking for `cuda` fails inside the SDK with "requires `accelerate`", a package it does
not list. On a machine with a working NVIDIA driver the program would have stopped
before processing anything.
*Fix:* if loading fails on the selected device, the program warns and loads the model
on the CPU.

**Not solved.** These remain:
`23:05` gives `minute: 0` because JSON forbids `05` and the model does not write `5`;
`She said "hi"` is copied with single quotes because the model avoids escaping;
the Windows path `C:\Users\test` loses a folder;
and three symbol names out of 26 are wrong ("question marks" and "at signs" are copied
as words, "dollar signs" gives `$$`).

## Optimizations

Almost all the time is spent in the model: 65.2 s of a 65.7 s run (measured just before
the last row of the table below; the last change does not alter these proportions).
So the optimizations aim first at **fewer model calls**, then at **cheaper model calls**,
and last at making our own code negligible.

| Optimization | Measured effect |
|---|---|
| **Forced text is not generated.** Keys, braces and the unambiguous rest of a function name are appended directly. | 293 output tokens need 114 model calls: 61 % of the tokens cost nothing. At 0.57 s per call, generating them would add about 100 s (estimate). |
| **Per-function arguments prompt.** Once the function is chosen, the prompt shows only that function. | Arguments prompt 257 -> 163 tokens (5 functions) and 385 -> 161 (11 functions). Run time 84.1 s -> 62.6 s and 231.7 s -> 128.5 s, with the same answers. |
| **Trie walk instead of testing every token.** | Allowed tokens inside a string: 415 ms -> 4.6 ms. Inside a number: 190 ms -> 0.5 ms. Same sets (checked by an assertion). |
| **Plain-text token set precomputed.** Of the 147,031 tokens allowed inside a string, only 6,192 need the walk. | Included in the 4.6 ms above. |
| **Cache of allowed sets per grammar state.** | A repeated state costs about 10 µs. All 114 lookups of a run take 50 ms in total, under 0.1 % of the run. |
| **Grammars and prompt prefixes built once per function**, and a cache of tokenized words. | Not measured separately; part of the 0.4 s that is not model time. |
| **Inputs and output path checked before loading the model.** | A wrong path fails at once instead of after the 5 s model load. |

How the provided 11 prompts evolved:

| Version | Time | Model calls | Argument sets right |
|---|---|---|---|
| First working version (rules in the prompt) | 71.8 s | 99 | 9 / 11 |
| Few-shot examples instead of rules (longer prompt) | 87.3 s | 97 | 10 / 11 |
| Shorter examples; space before a value left to the model | 84.1 s | 103 | 10 / 11 |
| Per-function arguments prompt | 62.6 s | 106 | 10 / 11 |
| Opening quote left to the model; neutral example | 66.2 s | 114 | 11 / 11 |
| SDK reduced to the two subject methods (plain prompt) | 63.8 s | 114 | 11 / 11 |

Quality changes cost time (a longer prompt, one more call per number, one more call
per string); the per-function prompt paid for them and more.

Not done: batching several prompts in one forward pass and reusing the attention cache
between steps. The SDK exposes neither, and its private attributes are off limits.

### Second pass: making each model call cheaper

A profile of the whole run showed 64.3 s of 64.5 s inside `get_logits_from_input_ids`,
and our own code (masks, trie walks, grammar) under 1 s. A benchmark of the model alone
gave the cost of one call: about `40 ms + 2.7 ms x context tokens` with the default
settings. So only two levers were left: shorter contexts, or faster forward passes.

Attempts, in order, all measured on the three data sets (45 prompts) against the
baseline (subject 64.7 s, extended 129.0 s, edge cases 67.3 s; 261 s in total):

| Attempt | Time (subject / extended / edge) | Result | Kept? |
|---|---|---|---|
| Shorter arguments instructions (26 -> 10 tokens) | 59.9 / 121.5 / 64.5 s (-6 %) | `"NUMBERS"` becomes `"NUMBER"` in a subject prompt | no |
| Shorter instructions in both prompts | 58.6 / 139.5 / 62.6 s | same miss, plus the empty prompt picks another function | no |
| Short instructions and one example instead of two (-25 tokens) | 59.1 / 107.9 / 51.0 s (-17 %) | regex `[0-9]+` becomes `[0-9]`, vowels regex becomes `aeiou` | no |
| No instructions at all | 54.0 / 115.0 / 59.0 s (-14 %) | `"shrek"` becomes `"shr"`, and 7 other changes | no |
| **Thread count**: only the performance cores of the hybrid CPU | **35.0 / 70.4 / 37.5 s (-45 %)** | **byte-identical output** on the 45 prompts | **yes** |
| **Cache of logits** for contexts already sent to the model | 0 % on a normal run; -27 % model calls when retries happen | byte-identical output | **yes** |

Every prompt change saved time but lost at least one answer of the provided set: the
current prompt is already the shortest one that keeps 11 / 11, so it stays as it is.

**Thread count.** The i7-12700KF has 8 performance cores (16 logical CPUs) and 4
efficiency cores. PyTorch starts 12 threads and splits every matrix product evenly
between them, so the fast cores wait for the slow ones. Measured on a 200-token
context:

| Threads | 4 | 6 | 7 | **8** | 9 | 10 | 12 (default) | 16 | 20 |
|---|---|---|---|---|---|---|---|---|---|
| Seconds per call | 0.54 | 0.40 | 0.38 | **0.31** | 0.70 | 0.65 | 0.62 | 0.47 | 0.44 |

`src/hardware.py` reads `/sys/devices/cpu_core/cpus` (Linux lists the performance
cores of a hybrid Intel CPU there), counts the physical cores, and sets
`OMP_NUM_THREADS` before the SDK is imported. Our code never imports PyTorch. On a
processor that is not hybrid, nothing changes. `--threads N` forces a count and a
user-defined `OMP_NUM_THREADS` always wins.

**Logits cache.** Decoding is greedy, so the same context always gives the same
answer. `LogitsCache` keeps the logits of the last 128 contexts (as `float32`, the
precision the model computes them in, so no result can change; about 78 MB). When the
first attempt fails and the fallback decoder retries, it replays the same steps until
the limits differ: with `--max_new_tokens 25`, three prompts are retried and 43 of 157
calls come from the cache, so the run takes as long as one without retries (35.5 s). A
prompt repeated in the input file is also answered from memory. The cache also counts
the real forward passes: the former counter forgot the calls of a failed attempt.

**Several prompts in parallel (tried, not kept).** Instead of giving all the cores to
one prompt, one process per prompt could split them. Measured as total model calls per
second (200-token context, each process pinned to its cores with `taskset`, 20 s):

| Layout | Calls / s |
|---|---|
| 1 process, 8 performance cores (current) | 3.25 |
| 1 process, 12 threads (library default) | 1.65 |
| 1 process on the performance cores + 1 on the 4 efficiency cores | 3.35 |
| 2 processes x 4 performance cores | 3.60 |
| 4 processes x 2 performance cores + 1 on the efficiency cores | 3.85 |
| 8 processes x 1 performance core | 4.00 |

The CPU is already saturated by one process, so parallel workers only add 10 to 23 %,
and the efficiency cores add 3 % at best. In return every process loads its own copy of
the model (about 2.5 GB of RAM each, 20 GB for 8), and prompts are very unequal (a regex
prompt needs 30 calls, a greeting 4): with one core per worker the longest prompt alone
would take about 60 s, more than the whole sequential run (35 s). Not worth it.

Still not possible: caching the attention keys and values between steps, or scoring
several drafted tokens in one call (speculative or prompt-lookup decoding). Both need
the logits of more than the last position, and the SDK returns only those.

## Code review and refactoring

Once everything worked, the whole code was reviewed for duplication and for functions
doing the same kind of work in different places. The rule for every change was that the
output must not move: the results of the three data sets (45 prompts) were saved before
and compared byte for byte after. They are identical, with the same number of model
calls.

| What was found | What was done |
|---|---|
| **Five functions of the grammar each had their own `if kind == ...` chain** over the six kinds of node (initial state, consuming a byte, ending early, forced text, entering a child). Adding a kind meant editing five places. | One class per kind (`LiteralNode`, `EnumNode`, `StringNode`, `NumberNode`, `SequenceNode`, `ArrayNode`) deriving from `Node`, each implementing the same five methods. `Grammar.advance` went from about 45 lines of dispatch to a dozen lines that know no node type. No `kind` test is left in the code. |
| **`number` and `integer` were two kinds** sharing one function with a flag. | One `NumberNode` with an `integer_only` attribute. |
| **Seven chained `if` for the first byte of a UTF-8 character.** | One table of byte ranges read by a loop. |
| **"Are we inside a string?" was worked out in three places**, from the node kind and its raw state, in the token index and twice in the decoder. | One method, `Grammar.text_mode`, answered by the node itself. |
| **Three places knew the list of types**: building the grammar, describing a type in the prompt, checking a decoded value. The last two were loose functions in different modules. | `describe()` and `coerce()` are methods of `ParameterSpec`; `signature()` and `coerce_arguments()` are methods of `FunctionDefinition`. Checking scalars uses one table of accepted Python types instead of five near-identical branches. |
| **Two loaders with the same skeleton** (read the array, validate each entry, warn and skip the bad ones). | One generic `load_entries(path, label, noun, model, strict)`; `load_functions` and `load_prompts` are now a few lines each. |
| **The same trie-building loop** in the tokenizer and in the token index. | One `build_trie` function. |
| **Retry on failure lived in the command line module**, which built two complete pipelines (two prompt builders, two sets of grammars, two caches). | `FunctionCaller` owns an optional `fallback_decoder`; one pipeline, shared grammars and caches. |
| **`run()` did everything**: loading, the loop, progress, interruption. | The loop moved to `process_prompts`; `run` reads as the list of steps. |
| **Default values written twice** (in `Settings` and in `argparse`). | `argparse` takes its defaults from a `Settings()` instance. |
| **The decoder built the same "forced step" object in two branches** and had a fallback for non-ASCII forced text that could never trigger (literals are ASCII by construction). | One code path: the chunk appended is always the decoded bytes of the step's tokens. |
| **Code that the program could no longer reach** after restricting the SDK: BPE merges, control tokens, chat layout. | Deleted (about 150 lines), with their tests. |

Checked after the changes:

- outputs identical on the provided, extended and edge-case sets;
- allowed-token computation as fast as before (4.3 ms for a string state, 0.5 ms for a
  number state, against 4.6 ms and 0.5 ms);
- `flake8`, `mypy --strict`, and 301 tests (294 before): the tests of the deleted code
  went away and tests of each node class, of the generic loader's users and of the
  fallback decoder were added.

Left as it is, on purpose: the number automaton is still a chain of conditions (it reads
like the JSON grammar; a transition table was shorter but harder to check by eye), and
the word-splitting scanner is long because the rule it reproduces has seven cases.

## Performance analysis

Measured on an Intel i7-12700KF, CPU only (no GPU), with `Qwen/Qwen3-0.6B`.

| Data set | Prompts | Right function | All arguments right | Valid JSON | Time |
|---|---|---|---|---|---|
| Provided examples (5 functions) | 11 | 11 / 11 | 11 / 11 | 11 / 11 | 35 s (64 s with 12 threads) |
| `tests/data/extended_*` (11 functions) | 19 + 1 empty | 19 / 19 | 16 / 19 | 20 / 20 | 70 s (129 s) |
| `tests/data/edge_case_prompts.json` | 14 | n/a | n/a | 14 / 14 | 38 s (67 s) |

- **Accuracy.** 27 of 30 meaningful prompts fully right (90 %), 30 of 30 with the right
  function. The three misses are listed at the end of
  [Challenges faced](#challenges-faced). A regex argument is counted as right when it
  matches what the request describes; the exact pattern can differ between runs of
  different models.
- **Speed.** Model loading takes about 4 s once downloaded. One model call costs roughly
  `70 ms + 1.25 ms x context tokens` on the 8 performance cores (twice that with the
  default 12 threads), so about 0.3 s here. Long values cost the most: one call per digit
  of a number; the prompt with a 20-digit number took 22 s with 12 threads.
- **Reliability.** Validity does not depend on the model: the test suite drives the
  decoder with *random* logits and with a model that always asks for a newline, and
  still gets schema-compliant JSON every time. `openai-community/gpt2`, which does not
  understand the task at all, also produces 11 valid entries (with wrong content).
- **Other models** (same 11 prompts): `Qwen/Qwen2.5-0.5B-Instruct` 38 s with 9 / 11
  functions right, `HuggingFaceTB/SmolLM2-360M-Instruct` 48 s with 10 / 11. All their
  output is valid JSON. They lose more than Qwen3 from the plain text prompt: with their
  chat layout both chose 11 / 11.

## Testing strategy

`make test` runs 309 tests in under a second, without the model. They use a tiny
256-byte vocabulary and fake models, so the logic is tested apart from the LLM.

- **Grammar**: tables of valid and invalid numbers, strings (escapes, UTF-8), arrays,
  objects, enums, and prefix function names; each node class on its own.
- **Decoder**: allowed-token sets for known states; 40 seeds of random logits must
  always give valid, schema-compliant JSON; an adversarial model; token budgets.
- **Tokenizer**: round trips, split rules, longest match, the byte trie, bad vocabulary files.
- **Input/output**: missing, unreadable, binary, empty and broken files; skipped
  entries; atomic writing.
- **Hardware**: CPU lists, hyper-threads counted once, a fake `/sys` tree, the
  user's `OMP_NUM_THREADS` respected.
- **Logits cache**: hits, least-recently-used eviction, disabled cache.
- **Command line**: every bad input gives a one-line `error:` and never a traceback;
  full runs with a fake model; retry and skip; `Ctrl+C` keeps partial results.

`RUN_SLOW=1 make test` adds two tests with the real model: a round trip of 1,500 random
strings through the real vocabulary, and an end-to-end run on the provided examples.

Three data sets were used for manual review: the provided one, an extended one written
for this project (other names, integers, booleans, enums, arrays, nested objects, a
function without arguments), and an edge-case list (empty prompt, empty string, 20-digit
number, quotes, backslashes, emoji, prompt injection, no matching function, another
language).

## Bonus features

| Bonus | Where |
|---|---|
| Multiple models | `--model`; any byte-level BPE model that the SDK can load |
| Tokenizer recoded, `encode` / `decode` not used | `src/tokenizer.py` (`Tokenizer.encode`, `Tokenizer.decode`); the program calls only `get_logits_from_input_ids` and `get_path_to_vocab_file` |
| Integration of encoding/decoding with constrained decoding | `TokenIndex` and `ConstrainedDecoder` in `src/decoder.py`; see "Algorithm explanation" |
| Error recovery | invalid entries skipped with a warning, retry with stricter limits, results saved after every prompt, atomic writes, `Ctrl+C` handling |
| Performance optimizations | forced text, trie walk, caches of allowed tokens and of logits, CPU threads; see [Optimizations](#optimizations) |
| Test suite | `tests/` |
| Visualization | `--visualize` |
| Nested arguments | `array` and `object` types, at any depth |

## Resources

- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [RFC 8259 - The JSON Data Interchange Format](https://www.rfc-editor.org/rfc/rfc8259)
- Willard and Louf, [Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702) (token masks from finite state machines)
- Sennrich, Haddow and Birch, [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (BPE)
- [Hugging Face tokenizers: BPE summary](https://huggingface.co/docs/transformers/tokenizer_summary)
- [OpenAI function calling guide](https://platform.openai.com/docs/guides/function-calling)
- [pydantic documentation](https://docs.pydantic.dev/)

### Use of AI

Claude Code (Anthropic) was used throughout this project:

- condensing the subject into working notes and a development plan;
- designing the architecture (byte-level automaton, token trie, two-stage generation);
- writing the source code in `src/`, the tests in `tests/`, the Makefile and this README;
- running the experiments reported in this document and diagnosing the model's failures
  from decoding traces.

Every part was reviewed and must be understood by the author before the defense.
