# call me maybe — subject condensado (v1.5)

Resumen de `en.subject.pdf` sin prólogo, capítulo de IA ni texto introductorio.
Si algo aquí contradice al PDF, manda el PDF.

## 1. Objetivo

Programa que traduce prompts en lenguaje natural a llamadas a función estructuradas.
No responde la pregunta: devuelve **qué función llamar y con qué argumentos tipados**.

```
"What is the sum of 40 and 2?"  ->  name: fn_add_numbers, parameters: {"a": 40.0, "b": 2.0}
```

La técnica obligatoria es **constrained decoding**: en cada paso de generación se
ponen a `-inf` los logits de los tokens que romperían el JSON o el esquema, y se
elige solo entre los válidos. Prohibido confiar en que el modelo "saque JSON bien"
solo con el prompt.

## 2. Reglas duras

### Prohibido
- `dspy`, `pytorch`, `transformers`, `huggingface`, `outlines` o similares **en tu código**
  (el `llm_sdk` sí los usa por dentro; tú no los importas).
- Métodos o atributos privados de `llm_sdk` (`_model`, `_tokenizer`, `_device`...).
- Elegir la función con heurísticas (palabras clave, regex sobre el prompt, etc.):
  **la elige el LLM**.
- Hardcodear soluciones para los ejemplos: los ficheros de entrada cambian en la evaluación.
- Subir el directorio de salida (`data/output/`) al repo.

### Obligatorio
- Python >= 3.10, estándar `flake8`, `mypy` sin errores.
- Type hints en parámetros, retornos y variables donde aplique.
- Docstrings PEP 257 (estilo Google o NumPy) en funciones y clases.
- **Todas las clases usan `pydantic`** para validación.
- Paquetes permitidos: `numpy`, `json` (stdlib), `pydantic`.
- Entorno virtual con `uv`; el evaluador y la moulinette solo ejecutan `uv sync`.
- Modelo por defecto `Qwen/Qwen3-0.6B`. Otros modelos valen solo si además funciona con este.
- Nunca crashear: `try/except`, context managers para ficheros, mensajes de error claros.
  Un crash por excepción no controlada = proyecto no funcional.

## 3. Ejecución

```
uv run python -m src [--functions_definition <file>] [--input <file>] [--output <file>]
```

| Flag | Por defecto |
|---|---|
| `--functions_definition` | `data/input/functions_definition.json` |
| `--input` | `data/input/function_calling_tests.json` |
| `--output` | `data/output/function_calling_results.json` |

`src` es un paquete ejecutable (necesita `src/__main__.py`).

## 4. Entradas

`function_calling_tests.json` — array de objetos con una clave `prompt`:

```json
[{"prompt": "What is the sum of 2 and 3?"}, {"prompt": "Greet shrek"}]
```

`functions_definition.json` — array de funciones disponibles:

```json
[{
  "name": "fn_add_numbers",
  "description": "Add two numbers together and return their sum.",
  "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
  "returns": {"type": "number"}
}]
```

Ambos ficheros pueden **faltar o contener JSON inválido**: hay que gestionarlo sin crashear.

### Datos de ejemplo entregados (`data/input/`)

| Función | Parámetros |
|---|---|
| `fn_add_numbers` | `a: number`, `b: number` |
| `fn_greet` | `name: string` |
| `fn_reverse_string` | `s: string` |
| `fn_get_square_root` | `a: number` |
| `fn_substitute_string_with_regex` | `source_string: string`, `regex: string`, `replacement: string` |

11 prompts: 2 sumas, 2 saludos, 2 reverse, 2 raíces cuadradas y 3 de sustitución con regex
(números -> `NUMBERS`, vocales -> asteriscos, `cat` -> `dog`). Los de regex son los difíciles:
el modelo tiene que inventar el patrón y copiar literalmente una frase con comillas y apóstrofos.

## 5. Salida

Un único fichero JSON: array con un objeto por prompt, con **exactamente** estas claves:

```json
[
  {"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}},
  {"prompt": "Reverse the string 'hello'", "name": "fn_reverse_string", "parameters": {"s": "hello"}}
]
```

Validación:
- JSON válido (sin comas finales ni comentarios).
- Claves y tipos idénticos al esquema de `functions_definition.json`.
- Ninguna clave extra ni texto fuera del JSON.
- Todos los argumentos requeridos presentes.
- Tipos correctos (`number`, `string`, `boolean`, etc.). `number` admite entero o float;
  el ejemplo oficial los escribe como float (`2.0`).

## 6. LLM SDK (`llm_sdk.Small_LLM_Model`)

| Método público | Devuelve |
|---|---|
| `get_logits_from_input_ids(input_ids: list[int])` | `list[float]`: logits crudos del **siguiente** token |
| `get_path_to_vocab_file()` | ruta a `vocab.json` (token -> id) |
| `encode(text: str)` | tensor 2-D `[[ids]]`, sin tokens especiales añadidos |
| `decode(ids)` | `str`, elimina tokens especiales (uso opcional) |
| `get_path_to_merges_file()` | ruta a `merges.txt` (no citado en el PDF, pero es público) |
| `get_path_to_tokenizer_file()` | ruta a `tokenizer.json` (ídem) |

Detalles del código entregado que condicionan el diseño:
- `get_logits_from_input_ids` hace un forward completo de toda la secuencia en cada llamada
  (sin caché) y convierte ~150k logits a lista de Python. Es el coste dominante.
- El constructor acepta `model_name` y `device`; elige `mps` > `cuda` > `cpu`.
- La primera ejecución descarga el modelo desde Hugging Face.
- El vocabulario usa `Ġ` para representar el espacio inicial de un token.

Bucle de generación: prompt -> tokens -> ids -> logits -> **máscara de tokens válidos**
-> elegir token -> añadirlo a la secuencia -> repetir.

## 7. Objetivos de rendimiento

- >= 90 % de acierto en función y argumentos.
- 100 % de salidas parseables y conformes al esquema.
- Todos los prompts en menos de 5 minutos en hardware estándar.
- Casos límite a probar: strings vacíos, números grandes, caracteres especiales, tipos
  incorrectos, prompts ambiguos, funciones con varios parámetros.

## 8. Entregables

En el repositorio:
- `src/` con la implementación.
- `pyproject.toml` y `uv.lock`.
- `llm_sdk/` copiado al mismo nivel que `src/`.
- `data/input/` con los ficheros de prueba.
- `README.md`.
- `Makefile`.
- `.gitignore` (artefactos de Python y la salida).

### Makefile

| Regla | Qué hace |
|---|---|
| `install` | instala dependencias (`uv sync`) |
| `run` | ejecuta el programa |
| `debug` | ejecuta con `pdb` |
| `clean` | borra `__pycache__`, `.mypy_cache`, etc. |
| `lint` | `flake8 .` y `mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs` |
| `lint-strict` (opcional) | `flake8 .` y `mypy . --strict` |

### README.md (en inglés)

Primera línea, en cursiva:
`*This project has been created as part of the 42 curriculum by <login>.*`

Secciones: Description, Instructions, Resources (referencias + cómo se usó la IA y en qué
partes), Algorithm explanation, Design decisions, Performance analysis, Challenges faced,
Testing strategy, Example usage.

### Tests
Recomendados (`pytest` o `unittest`), no se entregan ni puntúan.

## 9. Bonus (opcional, deben funcionar de verdad)

- Soporte para más modelos además de Qwen3-0.6B.
- Tokenizador propio: no usar `encode`/`decode` en el código principal, solo
  `get_logits_from_input_ids` y `get_path_to_vocab_file`; exponer `encode` (y `decode`) propios.
- Demostrar cómo se integran encode/decode con el constrained decoding.
- Recuperación de errores avanzada.
- Optimizaciones de rendimiento (caché, batching).
- Suite de tests completa.
- Visualización del proceso de generación.
- Argumentos anidados complejos.

## 10. Evaluación

- Solo se revisa lo que hay en el repositorio.
- Pueden pedir una **modificación en directo** de unos minutos (cambiar un comportamiento,
  reescribir unas líneas, añadir algo pequeño): hay que entender todo el código.
- Los ficheros de entrada pueden cambiar durante la evaluación.

## 11. Incoherencias del subject

| Tema | Lo que dice | Qué seguir |
|---|---|---|
| Nombre de salida | V.4: `function_calling_results.json`; ejemplo de uso: `function_calls.json` | `function_calling_results.json` por defecto; el flag `--output` cubre el otro |
| Carpeta de salida | `data/output/` (IV.3.2, V.4) y `output/` (V.6, VIII) | `data/output/`; ignorar ambas en `.gitignore` |
| Claves del resultado | Intro: `function` / `arguments`; V.4: `name` / `parameters` | `prompt`, `name`, `parameters` |
| Nombre de función | Intro: `add_numbers`; resto: `fn_add_numbers` | El que venga en `functions_definition.json` |
| Flags | V.6 los escribe con un guion (`-input`) | Doble guion (`--input`), como en IV.3.2 |
| `encode` | El PDF dice que devuelve un tensor; el código devuelve uno 2-D | Usar `encode(text)[0].tolist()` sin importar `torch` |
