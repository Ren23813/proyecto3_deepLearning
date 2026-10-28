"""Normalización ortográfica del kaqchikel.

- normalize(): forma canónica para guardar y mostrar (NFC, saltillo unificado).
- fold(): clave "plegada" para buscar sin importar saltillo, diéresis,
  tildes ni mayúsculas. Es la base de la corrección ortográfica: si una
  consulta coincide por clave plegada pero no exacta, se sugiere la forma
  correcta del diccionario.
"""
import re
import unicodedata

# Variantes tipográficas del saltillo que aparecen en PDFs y teclados
_APOSTROPHES = str.maketrans({c: "'" for c in "’‘´`ʼʹ′"})
_FOLD_DROP = re.compile(r"[' ¡!¿?.,;:\-]")
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(_APOSTROPHES)
    return _SPACES.sub(" ", text).strip()


def strip_marks(text: str) -> str:
    """Quita diéresis y tildes (ä→a, á→a, ñ→n)."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def fold(text: str) -> str:
    return _FOLD_DROP.sub("", strip_marks(normalize(text)).lower())
