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

# Rangos de páginas del OKMA (1-indexadas, inclusivas). El número de página
# del PDF coincide con el impreso, así que se puede citar tal cual.
OKMA_DICT_PAGES = (57, 583)
OKMA_FRASES_PAGES = (585, 590)
OKMA_GLOSARIO_PAGES = (591, 660)
