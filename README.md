# Consulta de kaqchikel: asistente con RAG híbrido

Proyecto 3 de Deep Learning y Sistemas Inteligentes (CC3092). Es un asistente web para consultar el
idioma kaqchikel. La persona escribe una palabra o frase, en kaqchikel o en español, y el sistema
recupera las entradas relevantes de dos diccionarios elaborados por lingüistas. Con ellas, un LLM
(Claude) redacta la respuesta: traducción, significado, ejemplos, corrección ortográfica (saltillo y
diéresis) y la cita del diccionario y la página. Si la consulta no aparece en los diccionarios, lo
dice en lugar de inventar.

El LLM explica, pero la fuente de verdad es el diccionario: la página citada sale del índice y nunca
del modelo.

## Arquitectura

```
Consulta
  ├─ Normalización: NFC, apóstrofos → ', clave "plegada" sin saltillo ni diéresis
  ├─ Léxica: lema exacto → clave plegada → glosa en español → fuzzy (≥90) → BM25
  ├─ Densa: intfloat/multilingual-e5-base (768 dim), similitud coseno sobre 6 453 documentos
  └─ Fusión: coincidencias léxicas fuertes primero + RRF (k=60) del resto → top-5
       + corrección ortográfica determinista (awan → awän) y pares mínimos (ik → ik')
       + "sin evidencia": sin coincidencia fuerte y similitud < 0.85 → "no aparece", sin llamar al LLM
Prompt v3 (reglas, abreviaturas del OKMA y 3 ejemplos, en caché) + <consulta> + <recuperacion> + <entradas>
  → Claude Haiku 5.5, salida JSON validada con Pydantic (structured outputs), effort="low"
  → Post-validación: se descartan las citas no recuperadas; fuente, palabra y categoría salen del índice
  → Streamlit: tarjetas con "OKMA · pág. N", "¿Quisiste decir…?", "No aparece en los diccionarios"
```

Hay tres configuraciones con la misma interfaz, que se comparan en la evaluación y en la app:

| Modo | Qué hace |
|---|---|
| `sin_contexto` | El LLM responde de memoria (baseline) |
| `lexico` | Recuperación solo léxica + LLM (baseline) |
| `hibrido` | RAG híbrido léxico + embeddings + LLM (el producto) |

## Datos y derechos

| Fuente | Contenido | Procesamiento |
|---|---|---|
| *Diccionario Bilingüe Kaqchikel* (OKMA, 660 págs.) | 5 788 entradas, 90 frases idiomáticas y 5 417 filas de glosario es→kaq | Extracción de texto con `pypdf` y parser de entradas ([ingest_okma.py](src/kaq/ingest_okma.py)) |
| *Kaqchikel L2* (Dideduc Guatemala Occidente, 66 págs.) | 500 pares de vocabulario y 75 fragmentos de gramática | El PDF solo tiene imágenes: se transcribió con visión y se revisaron 5 páginas a mano ([ingest_l2.py](src/kaq/ingest_l2.py)) |

Una entrada de diccionario equivale a un fragmento (*chunk*), con metadatos de lema, categoría, diccionario y página.

**Los diccionarios pertenecen a los lingüistas que los proporcionaron y no tenemos permiso para
publicarlos.** Por eso quedan fuera del repositorio los PDFs, `data/processed/`, `data/index/`,
`data/demo_cache.json` y los archivos de resultados que copian texto de los diccionarios
(`eval/results/*/responses.jsonl`). Para reproducir el proyecto hace falta pedir los PDFs al equipo.

## Instalación

Requiere Python 3.11 o 3.12.

```powershell
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # y escribe tu ANTHROPIC_API_KEY (nunca subas .env)
```

Con los PDFs en la raíz del repositorio:

| Paso | Comando | Tiempo |
|---|---|---|
| Ingesta OKMA | `python -m src.kaq.ingest_okma` | ~1 min |
| Ingesta L2 (une las transcripciones de `data/processed/l2_ocr/`) | `python -m src.kaq.ingest_l2 build` | 1 s |
| Índice de embeddings (CPU) | `python -m src.kaq.build_index` | ~7 min la primera vez |
| Prueba de humo (10 consultas × 3 modos) | `python scripts/smoke_test.py` | ~1.5 min, ~$0.01 |

## Ejecutar la app

```powershell
streamlit run app/streamlit_app.py
```

- **Búsqueda:** ejemplos clicables, corrección "¿Quisiste decir…?" y alternativas clicables.
- **Barra lateral:** historial de la sesión y modo técnico. Desde ahí se elige la configuración, se
  comparan las 3 configuraciones lado a lado y se ven latencia, tokens, costo y entradas recuperadas.
- **Feedback 👍/👎:** queda en `logs/feedback.jsonl`.
- **Respaldo sin internet:** `python scripts/build_demo_cache.py` guarda 15 consultas × 3 modos en
  `data/demo_cache.json`. Si la API falla, la app responde desde ahí y lo avisa.

## Evaluación

El conjunto de evaluación ([eval/cases.jsonl](eval/cases.jsonl)) tiene 42 casos construidos a partir
de los diccionarios. Cada uno trae sus `gold_ids`, los lemas aceptados, las palabras clave de la
traducción y la página de referencia:

| Tipo | Casos | Ejemplo |
|---|---|---|
| kaq → es | 10 | `ixim` → maíz (OKMA p. 179) |
| es → kaq | 8 | `computadora` → kematz'ib' (OKMA p. 227) |
| Saltillo o diéresis omitidos | 8 | `kinaq` → kinäq' |
| Pares mínimos | 4 | `ik` (chile) frente a `ik'` (luna) |
| Palabras inexistentes | 5 | `blorptanik` → debe abstenerse |
| Prompt injection | 3 | "Ignora tus instrucciones…" |
| Cultural o gramática | 4 | `¿Qué significa ri'j ik'?` → cuarto menguante |

Métricas ([eval/metrics.py](eval/metrics.py)):

- **Exactitud automática:** lema correcto, palabra clave de la traducción, corrección esperada y
  abstención cuando corresponde.
- **Tasa de alucinación:** responde algo que no es exacto.
- **Recuperación:** recall@k y MRR sobre los `gold_ids`.
- **Casos especiales:** corrección ortográfica, pares mínimos y resistencia a la inyección.
- **Operativas:** latencia media y p95, tokens y costo por solicitud.
- **Fidelidad al contexto (0–2) y alucinación según un LLM-as-judge** ([eval/judge.py](eval/judge.py),
  Claude Sonnet 5.5). El juez se valida contra una evaluación humana a ciegas.

Para reproducir:

```powershell
# Recuperación sola (gratis)
python eval/run_eval.py --tag recuperacion_k10 --retrieval-only --k 10
python eval/run_eval.py --tag recuperacion_k10_umbral083 --retrieval-only --k 10 --dense-min 0.83

# Iteraciones (~$0.015 cada una con Haiku; ~$0.34 la de Sonnet)
python eval/run_eval.py --tag it0-2_base --prompt v1                     # 3 modos, ~$0.04
python eval/run_eval.py --tag it4_sin_atajo --modes hibrido --prompt v1 --no-short-circuit
python eval/run_eval.py --tag it5_k3 --modes hibrido --prompt v1 --k 3
python eval/run_eval.py --tag it6_prompt_v2 --modes hibrido --prompt v2
python eval/run_eval.py --tag it7_sonnet --modes hibrido --prompt v1 --model claude-sonnet-5-5
python eval/run_eval.py --tag it8_prompt_v3 --modes hibrido --prompt v3

# Juez (~$0.0055 por respuesta) y su validación humana
python eval/judge.py --tag it0-2_base
python eval/judge.py --tag it0-2_base --export-human 12                  # muestra para revisión humana
python eval/judge.py --tag it0-2_base --agreement                        # acuerdo juez vs. humano
```

Cada corrida deja en `eval/results/<tag>/` el detalle por caso (`rows.csv`) y el resumen
(`summary.json`), y regenera la tabla de iteraciones [eval/results/iteraciones.csv](eval/results/iteraciones.csv).

### Resultados

42 casos, Claude Haiku 5.5, top-k = 5. El juez es Claude Sonnet 5.5 y califica la fidelidad de 0 a 2.

**Comparación de las tres configuraciones (línea base, prompt v1):**

| Configuración | Exactitud | Alucinación (auto) | Fidelidad (juez) | Alucinación (juez) | recall@5 | Latencia media | Costo por solicitud |
|---|---|---|---|---|---|---|---|
| Sin contexto | 0.36 | 0.60 | 0.29 | 0.76 | — | 2.0 s | $0.00017 |
| Solo léxico | 0.95 | 0.02 | 1.71 | 0.00 | 0.97 | 2.1 s | $0.00035 |
| RAG híbrido | 0.98 | 0.02 | 1.74 | 0.00 | 1.00 | 2.3 s | $0.00037 |

Sin contexto, el modelo inventa traducciones y ejemplos. Por ejemplo, dice que `ya'` es una partícula
temporal, cuando el diccionario dice agua, y nunca sugiere la corrección de saltillo o diéresis
(corrección 0/8). Con recuperación, la alucinación cae a cero según el juez.

**Iteraciones sobre el modo híbrido** (detalle en [eval/results/iteraciones.csv](eval/results/iteraciones.csv)):

| Iteración | Cambio | Exactitud | Fidelidad (juez) | Alucinación (juez) | Latencia media / p95 | Costo por solicitud |
|---|---|---|---|---|---|---|
| it2 | Línea base: prompt v1, k=5, umbral 0.85 | 0.98 | 1.74 | 0.00 | 2.3 / 3.6 s | $0.00037 |
| it3 | Corrección: desempate de formas plegadas por número de entradas (`ixoq` → `ixöq`) | 1.00 | 1.79 | 0.00 | 2.2 / 3.5 s | $0.00036 |
| it4 | Sin atajo de "sin evidencia": el LLM decide si abstenerse | 1.00 | — | — | 2.5 / 3.4 s | $0.00039 |
| it5 | top-k = 3 | 0.98 | — | — | 2.2 / 3.4 s | $0.00030 |
| it5 | top-k = 10 | 0.98 | — | — | 2.6 / 4.6 s | $0.00045 |
| it6 | Prompt v2: v1 sin ejemplos few-shot | 1.00 | 1.83 | 0.02 | 2.5 / 4.4 s | $0.00037 |
| it7 | Claude Sonnet 5.5 en lugar de Haiku 5.5 | 1.00 | 1.86 | 0.02 | 3.6 / 6.0 s | $0.00801 |
| **it8** | **Prompt v3: abreviaturas del OKMA y sin reglas ortográficas inventadas (configuración final)** | **1.00** | **1.86** | **0.00** | **2.2 / 3.4 s** | **$0.00037** |

**Recuperación sin LLM** (k = 10, gratis):

| | recall@1 | recall@3 | recall@5 | MRR | Abstención correcta (umbral 0.85 / 0.83 / 0.80) |
|---|---|---|---|---|---|
| Léxico | 0.74 | 0.94 | 0.97 | 0.84 | 0.71 |
| Híbrido | 0.74 | 0.97 | 1.00 | 0.85 | 0.86 / 0.71 / 0.14 |

**Decisiones respaldadas por los resultados:**

- **Umbral de 0.85:** bajarlo no reduce la falsa abstención, que ya es 0, y sí deja pasar palabras inventadas.
- **k = 5:** con k=3 se pierde contexto, y con k=10 sube la latencia sin mejorar la exactitud.
- **Atajo de "sin evidencia":** el LLM también se abstiene por sí solo (it4), pero el atajo responde en
  ~50 ms y sin costo.
- **Haiku 5.5 con v3:** iguala la fidelidad de Sonnet 5.5 con 1/22 del costo y menor latencia.
- **Ejemplos few-shot:** quitarlos (v2) ahorra tokens, pero no abarata la solicitud, porque el prompt
  queda debajo del mínimo para entrar en caché. Además subió la alucinación.

**Fallas encontradas y cómo se corrigieron:**

1. **Corrección ambigua.** `ixoq` → `ixo'q`, cuando la referencia es `ixöq` (mujer). Se corrigió en it3 con el desempate.
2. **Abreviaturas mal expandidas.** El juez detectó que el modelo explicaba "s. np." como "sustantivo
   propio", cuando significa "usualmente no poseído". Se corrigió en it8 con la lista oficial de la pág. 53 del OKMA.
3. **Corrección incoherente con frases idiomáticas.** En `¡k'o chi nuk'u'x!` sugería la misma forma
   porque comparaba la consulta sin `¡!` contra el lema con ellos. Se corrigió en it8.
4. **Consulta de gramática que no pasa el umbral.** "¿Cómo se forma el plural de un sustantivo?" queda
   en 0.8498 y se responde "no aparece". Sigue abierta como limitación: bajar el umbral empeora la abstención.

**Validación del juez:** `eval/results/it0-2_base/human_review.csv` trae 12 casos (4 por
modo) para calificarlos a ciegas. El acuerdo se calcula con `--agreement`.

Una limitación conocida del juez es que no ve la lista de abreviaturas, así que a veces penaliza
expansiones correctas como "usualmente no poseído".

Costo total del desarrollo de la evaluación: unos $2.30 (pipeline $0.53 y juez $1.75).

## Estructura

```
app/                  interfaz Streamlit (streamlit_app.py, style.css)
src/kaq/              config, normalización, ingesta, índice, recuperación, prompts, generación, pipeline
scripts/              prueba de humo y caché de respaldo para la feria
eval/                 casos, ejecución, métricas, juez y resultados
.streamlit/           tema de la interfaz
```

## Limitaciones

- **Embeddings:** e5 no conoce el kaqchikel, así que la capa léxica hace la mayor parte del trabajo.
  Los embeddings ayudan sobre todo con las consultas en español y las conceptuales.
- **Umbral de "sin evidencia" (0.85):** se calibró con pocas consultas. Hay consultas reales que
  quedan justo debajo, por ejemplo "¿Cómo se forma el plural de un sustantivo?" (0.8498), y palabras
  inventadas que lo superan por coincidencias parciales.
- **Formas plegadas ambiguas:** cuando varias formas comparten la misma clave plegada, la corrección
  elige la que tiene más entradas en los diccionarios (`ixoq` → `ixöq`) y muestra las demás como
  alternativas. Es una heurística: en 290 claves ambiguas puede no coincidir con lo que la persona
  quiso escribir.
- **El conjunto de evaluación es pequeño (42 casos)** y las mejoras de it3 e it8 se diseñaron después
  de ver sus fallas, así que hay riesgo de sobreajuste. Lo ideal sería validarlas con casos nuevos.
- **Calidad de los datos:** en ~7 % de las entradas OKMA el campo de traducción queda vacío. El L2 es
  un texto pedagógico y no un diccionario.
- **No reemplaza a una persona hablante:** no traduce oraciones completas ni cubre variantes
  regionales fuera de los diccionarios.
