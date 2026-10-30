"""Configuración central: rutas de datos y rangos de páginas de las fuentes."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Fuentes (los PDFs están en la raíz y fuera de git)
OKMA_PDF = ROOT / "Diccionario Bilingüe Kaqchikel OKMA.pdf"
L2_PDF = ROOT / "kaqchikel L2 Dideduc Guate Occidente.pdf"

DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
LOGS_DIR = ROOT / "logs"

OKMA_ENTRIES = PROCESSED_DIR / "okma_entries.jsonl"
OKMA_GLOSARIO = PROCESSED_DIR / "okma_glosario.jsonl"
OKMA_STATS = PROCESSED_DIR / "stats.json"
OKMA_SAMPLE = PROCESSED_DIR / "muestra_revision.md"

L2_PAGES_DIR = DATA_DIR / "raw" / "l2_pages"  # imágenes de página extraídas del PDF
L2_OCR_DIR = PROCESSED_DIR / "l2_ocr"  # transcripción por página (JSON)
L2_ENTRIES = PROCESSED_DIR / "l2_entries.jsonl"

# Rangos de páginas del OKMA (1-indexadas, inclusivas). El número de página
# del PDF coincide con el impreso, así que se puede citar tal cual.
OKMA_DICT_PAGES = (57, 583)
OKMA_FRASES_PAGES = (585, 590)
OKMA_GLOSARIO_PAGES = (591, 660)

# Índice (se regenera con python -m src.kaq.build_index)
INDEX_DOCS = INDEX_DIR / "docs.jsonl"
INDEX_EMBEDDINGS = INDEX_DIR / "embeddings.npy"
INDEX_META = INDEX_DIR / "meta.json"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# Recuperación
TOP_K = 5
RRF_K = 60  # constante de Reciprocal Rank Fusion
CANDIDATES = 20  # candidatos por recuperador antes de fusionar
FUZZY_CUTOFF = 90  # similitud mínima (0-100)
BM25_MIN_SCORE = 7.5  # modo léxico
DENSE_MIN_SCORE = 0.85  # modo híbrido



## ANTRHOPIC LLM
MODEL = "claude-haiku-5-5"  # desarrollo y evaluación masiva
MODEL_ALT = "claude-sonnet-5-5"  # comparación en una iteración
PROMPT_VERSION = "v1"
MAX_TOKENS = 2000  # incluye el razonamiento adaptativo del modelo
EFFORT = "low"  # desde la API 2026-10-21 no se aceptan temperature/top_p/top_k
MAX_ENTRY_CHARS = 1500  # recorte de cada entrada en el prompt
# Si la recuperación no encuentra evidencia, se responde "no encontrado" sin
# llamar al LLM (costo y latencia cero, sin riesgo de alucinar). En False el
# LLM recibe las entradas débiles y decide; sirve para medir su abstención.
SHORT_CIRCUIT_NO_EVIDENCE = True

# USD por millón de tokens. Escritura en caché = 1.25x entrada; lectura = 0.1x.
PRICES = {
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_write": 0.125, "cache_read": 0.01},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_write": 2.50, "cache_read": 0.20},
}
REQUESTS_LOG = LOGS_DIR / "requests.jsonl"
