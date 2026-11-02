# call me maybe — plan de desarrollo

Pasos ordenados para construir el proyecto. Cada fase termina con una comprobación
concreta; no pases a la siguiente hasta que se cumpla.
Requisitos y restricciones: ver `SUBJECT_RESUMEN.md`.

## Estado de partida (comprobado en este PC el 2026-10-10)

- Hay: `en.subject.pdf`, `data/input/*.json`, `llm_sdk/`. Python 3.12.3. 31 GB de RAM, 20 GB libres en disco.
- **`uv` no está instalado.**
- **No hay GPU NVIDIA**: el modelo correrá en CPU, así que el límite de 5 minutos es el riesgo principal.
- El modelo no está descargado (`~/.cache/huggingface` no existe). Hay conexión a huggingface.co.
- **El repo git actual es `~/42_core` entero**, no esta carpeta. La entrega necesita su propio
  repositorio con `src/`, `llm_sdk/`, etc. en la raíz.

## Arquitectura propuesta

```
call_me_maybe/
├── pyproject.toml  uv.lock  Makefile  README.md  .gitignore  .flake8
├── llm_sdk/                    (entregado, no se toca)
├── data/input/                 (entregado)
├── src/
│   ├── __init__.py
│   ├── __main__.py             CLI: argumentos, orquestación, códigos de salida
│   ├── models.py               modelos pydantic de entrada y salida
│   ├── io_utils.py             lectura/escritura de JSON con errores controlados
│   ├── vocabulary.py           vocab.json -> texto real de cada token
│   ├── constraints.py          autómatas: qué texto es válido en cada estado
│   ├── decoder.py              bucle de generación con máscara de logits
│   ├── prompt_builder.py       texto que se le pasa al modelo
│   └── function_caller.py      une todo: prompt -> nombre -> parámetros
└── tests/                      pytest (no se entrega)
```

### Idea central del algoritmo

El JSON de salida tiene forma fija: `{"name": "<fn>", "parameters": {"<p1>": <v1>, ...}}`.
Solo hay dos cosas que el modelo decide de verdad: **qué función** y **el valor de cada parámetro**.

1. Se escribe el prompt (funciones disponibles + petición del usuario) y se deja abierta
   la respuesta en `{"name": "`.
2. **Nombre**: en cada paso solo son válidos los tokens cuyo texto continúa el prefijo de
   algún nombre de función existente. El modelo elige entre ellos por logits. Es imposible
   que salga un nombre inexistente.
3. Conocida la función, se conocen sus parámetros y tipos. Las partes fijas
   (`", "parameters": {"a": `) se añaden a la secuencia directamente: solo hay una
   continuación válida, no hace falta consultar al modelo.
4. **Valor**: se genera con el autómata de su tipo (`number`, `string`, `boolean`...),
   enmascarando a `-inf` todo token que lo saque de un estado válido.
5. Se repite por cada parámetro, se cierra el JSON, se parsea y se valida con pydantic.

El fichero final se escribe con `json.dump` a partir de objetos Python ya validados, así
que la validez del JSON de salida no depende del modelo.

---

## Fase 0 — Entorno y esqueleto

1. Instalar `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
2. Decidir el repositorio: crear uno propio para esta carpeta (o clonar el del intra y mover aquí el contenido).
3. `pyproject.toml` en la raíz: `requires-python >= 3.10`, dependencias `numpy` y `pydantic`,
   `llm-sdk` como dependencia local por ruta (`[tool.uv.sources]`), y grupo dev con
   `flake8`, `mypy`, `pytest`.
4. `uv sync` y commitear el `uv.lock` generado. Ojo: arrastra `torch` y `transformers`
   (varios GB de descarga).
5. `.gitignore`: `.venv/`, `__pycache__/`, `.mypy_cache/`, `.pytest_cache/`, `data/output/`, `output/`.
6. `.flake8` y config de mypy excluyendo `.venv` y `llm_sdk` (el código del SDK no cumple
   flake8 y no se puede modificar). Sin esto `flake8 .` falla por código ajeno.
7. `Makefile` con `install`, `run`, `debug`, `clean`, `lint`, `lint-strict`.
8. `src/__init__.py` y un `src/__main__.py` mínimo.

**Hecho cuando:** `uv sync` funciona desde cero, `uv run python -m src` arranca y `make lint` pasa.

## Fase 1 — Explorar el SDK antes de diseñar

Script suelto en `tests/` o en un scratch, no en `src/`. Responder con datos reales:

1. Cargar `Small_LLM_Model()` y medir cuánto tarda la carga.
2. Medir una llamada a `get_logits_from_input_ids` con 50, 200 y 500 tokens.
   **Este número decide el presupuesto**: llamadas totales = 5 min / tiempo por llamada.
3. Abrir `vocab.json`: tamaño, formato, cómo aparecen el espacio (`Ġ`), el salto de línea,
   los dígitos, las comillas y los caracteres no ASCII.
4. Comprobar si los tokens especiales (`<|im_start|>`, `<|im_end|>`, `<think>`) están en
   `vocab.json` o solo en `tokenizer.json`, y qué ids les asigna `encode`.
5. Ver cómo tokeniza `encode` textos como `{"name": "`, `12.5`, `-3`, `"hello"`.
6. Generación greedy sin restricciones con un prompt de prueba para ver qué produce el
   modelo "en crudo" y si entra en modo `<think>`.

**Hecho cuando:** tienes apuntados el tiempo por llamada y el formato exacto del vocabulario.

## Fase 2 — Modelos y E/S

1. `models.py` (pydantic): `ParameterSpec` (type), `FunctionDefinition` (name, description,
   parameters, returns), `PromptItem` (prompt), `FunctionCallResult` (prompt, name, parameters).
   Validar: nombres no vacíos, nombres de función únicos, tipos soportados.
2. `io_utils.py`: cargar ambos JSON con context manager y errores diferenciados:
   fichero inexistente, sin permisos, JSON inválido, raíz que no es array, elemento que no
   cumple el esquema, lista vacía.
3. Escritura: crear el directorio de salida si no existe y volcar con `json.dump`.
4. `__main__.py`: `argparse` con los tres flags y sus valores por defecto. Cualquier error
   termina con mensaje claro por `stderr` y código de salida distinto de 0, sin traceback.

**Hecho cuando:** los tests de pytest cubren fichero ausente, JSON roto, clave que falta y
tipo desconocido, y ninguno produce traceback.

## Fase 3 — Vocabulario

1. `vocabulary.py`: leer `vocab.json` (ruta de `get_path_to_vocab_file()`) y construir
   `id -> texto real`. Los tokens están en codificación byte-level BPE: hay que deshacer el
   mapeo de caracteres a bytes (`Ġ` -> espacio, `Ċ` -> `\n`, etc.) para obtener el texto verdadero.
2. Tener en cuenta que un carácter UTF-8 multibyte puede estar repartido entre varios tokens:
   trabajar a nivel de bytes o descartar los tokens con UTF-8 incompleto.
3. Identificar los ids de tokens especiales para excluirlos siempre de la máscara.
4. Precalcular con numpy lo que se pueda reutilizar: conjunto de tokens puramente
   numéricos, tokens sin comillas ni barras ni caracteres de control, etc.

**Hecho cuando:** para una muestra de textos, concatenar el texto de los ids de `encode(t)`
reproduce `t` exactamente.

## Fase 4 — Restricciones (sin modelo)

Todo esto es lógica pura y se prueba sin cargar el LLM.

1. Interfaz común: dado el texto generado hasta ahora, decir si un token candidato es
   válido, y si el valor puede o debe terminar.
2. **Nombre de función**: trie o comprobación de prefijos sobre los nombres disponibles.
   Cuidado con nombres que son prefijo de otros (`fn_add` y `fn_add_numbers`): la comilla de
   cierre es lo que desambigua.
3. **number**: autómata `-? dígitos (. dígitos)? ((e|E) (+|-)? dígitos)?`. Sin ceros a la
   izquierda inválidos en JSON, sin `.` final, con longitud máxima.
4. **integer**: lo mismo sin parte decimal ni exponente.
5. **boolean**: solo `true` o `false`.
6. **string**: cualquier contenido hasta la comilla de cierre. Decidir cómo tratar `"` y `\`
   dentro del valor (escapes JSON) y prohibir caracteres de control crudos. Longitud máxima.
7. Tipo desconocido en el esquema: error claro en la validación de la Fase 2, no aquí.

**Hecho cuando:** hay tests por tipo con secuencias válidas e inválidas, incluidos
`-0.5`, `1e10`, `007`, `1.`, string vacío y string con comillas.

## Fase 5 — Decodificador restringido

1. `decoder.py`: bucle que recibe los ids actuales y una restricción, y devuelve los ids
   generados y su texto.
2. En cada paso: logits -> array numpy -> máscara (`-inf` en los inválidos) -> `argmax`.
3. Si solo hay un token válido, añadirlo sin llamar al modelo.
4. Si no queda ningún token válido: no crashear; error controlado para ese prompt.
5. Límite de tokens por valor para garantizar que termina siempre.
6. Recalcular la máscara sobre ~150k tokens en cada paso con un bucle Python es lento:
   usar las máscaras precalculadas de la Fase 3 y filtrar solo los candidatos necesarios.

**Hecho cuando:** con un modelo falso (logits aleatorios) el decodificador siempre produce
un valor válido para cada tipo. Ese test demuestra que la validez no depende del modelo.

## Fase 6 — Prompt y selección de función

1. `prompt_builder.py`: instrucción breve, lista de funciones (nombre, descripción,
   parámetros con tipo) y la petición del usuario, terminando en el inicio del JSON.
2. Probar el formato de chat de Qwen (`<|im_start|>system ... <|im_start|>assistant`) frente
   a texto plano, y cómo evitar el bloque `<think>`. Elegir con datos de la Fase 1.
3. Mantener el prompt corto: cada token de más se paga en todas las llamadas al modelo.
4. `function_caller.py`: generar el nombre restringido -> buscar la definición -> generar
   cada parámetro con su restricción -> construir `FunctionCallResult`.
5. Para los parámetros, incluir en el contexto el nombre de la clave que se está rellenando
   (ya ocurre de forma natural al escribir `"a": ` en la secuencia).

**Hecho cuando:** los 11 prompts de ejemplo eligen la función correcta.

## Fase 7 — Calidad de los argumentos

Aquí se gana o se pierde el 90 %.

1. Ejecutar los 11 prompts y revisar a mano cada argumento.
2. Casos difíciles esperados:
   - números: que copie `265` y no lo trunque o alargue;
   - strings: que pare en el sitio correcto y no añada texto;
   - regex: `fn_substitute_string_with_regex` exige inventar el patrón (`\d+`, `[aeiou]`)
     y copiar literalmente una frase con comillas y apóstrofos.
3. Ajustar solo cosas generales: redacción del prompt, ejemplos few-shot **con funciones
   distintas a las de prueba**, condición de fin de string. Nada específico de los ejemplos.
4. Crear un segundo juego de funciones y prompts propio (otros nombres, booleanos, enteros,
   más parámetros) para comprobar que no has sobreajustado a los datos entregados.

**Hecho cuando:** >= 90 % de aciertos en los datos entregados y en tu juego propio.

## Fase 8 — Rendimiento

1. Medir el tiempo total de los 11 prompts. Objetivo: holgura clara bajo los 5 minutos.
2. Palancas, por orden de impacto:
   - menos llamadas al modelo (saltar tokens forzados, parar el nombre en cuanto es único);
   - prompt más corto;
   - cálculo de la máscara vectorizado con numpy;
   - cargar el modelo y el vocabulario una sola vez.
3. Mostrar progreso por prompt en `stderr` para que el evaluador vea que avanza.

**Hecho cuando:** ejecución completa bajo 5 minutos en este PC, con margen.

## Fase 9 — Robustez

Probar cada caso y confirmar mensaje claro, sin traceback:

- fichero de funciones o de prompts inexistente, vacío, con JSON roto, o que no es un array;
- función sin parámetros, con muchos parámetros, con tipo desconocido;
- prompt vacío, muy largo, con emojis o caracteres no ASCII, con comillas y barras;
- prompt que no encaja con ninguna función (aun así debe salir una llamada válida);
- lista de funciones vacía;
- ruta de salida en un directorio inexistente o sin permisos;
- sin red y sin modelo descargado;
- `Ctrl+C` a mitad de ejecución.

Decidir y documentar qué pasa si un prompt falla: seguir con el resto y dejar constancia.

**Hecho cuando:** ninguna de estas entradas produce un traceback.

## Fase 10 — Entrega

1. `make lint` limpio; probar también `make lint-strict`.
2. Docstrings y type hints en todo; todas las clases son modelos pydantic.
3. Buscar que no haya `import torch`, `import transformers` ni accesos a `._` del SDK en `src/`.
4. `README.md` en inglés con la primera línea obligatoria y todas las secciones exigidas
   (incluye las cifras medidas en las fases 7 y 8 y el uso que se hizo de la IA).
5. Prueba final desde un clon limpio del repo: `uv sync` y `uv run python -m src`.
6. Confirmar que `data/output/` no está en el repo.
7. Preparar la defensa: poder explicar cada módulo y hacer un cambio pequeño en directo
   (añadir un tipo nuevo, cambiar el formato de salida, cambiar un valor por defecto).

## Bonus (solo con lo obligatorio perfecto)

Por relación esfuerzo/valor:

1. Suite de tests completa (ya la tienes casi hecha con las fases 2 a 5).
2. Visualización del proceso: flag `--verbose` que muestre por paso los tokens válidos y el elegido.
3. Soporte multi-modelo: flag `--model`; depende de que la Fase 3 no asuma detalles de Qwen.
4. Tokenizador propio (BPE con `vocab.json` y `merges.txt`) para no usar `encode`/`decode`.
5. Optimizaciones de caché.
6. Argumentos anidados (arrays y objetos): amplía los autómatas de la Fase 4.

## Decisiones pendientes

- Repositorio: ¿repo nuevo para esta carpeta o clonar el del intra?
- Salida de `number`: siempre float (`2.0`, como en el ejemplo oficial) o conservar enteros.
- Qué escribir en la salida cuando un prompt falla de forma irrecuperable.
