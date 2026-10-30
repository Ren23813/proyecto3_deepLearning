"""Recuperación léxica, densa e híbrida sobre el índice de diccionarios

- lexical(): coincidencia exacta del lema -> clave plegada (sin saltillo ni
  diéresis) -> glosa en español -> fuzzy -> BM25
- dense(): similitud coseno con embeddings e5 ("query: " + consulta)
- hybrid(): las coincidencias léxicas fuertes (exacta, plegada, español, fuzzy)
  quedan ancladas al inicio; el resto se ordena por Reciprocal Rank Fusion de
  BM25 y densa. En un diccionario la coincidencia exacta debe ganar siempre
- suggest_correction(): forma correcta del diccionario cuando la consulta solo
  coincide por clave plegada o por fuzzy (p. ej. "ik" sin saltillo -> "ik'").

La función principal es search(query, mode, k).
"""
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz, process

from . import config
from .normalize import fold, normalize, strip_marks

MODES = ("sin_contexto", "lexico", "hibrido")
_STRONG = ("exacta", "plegada", "español", "fuzzy")
_LEMMA_TYPES = {"entrada", "frase", "vocabulario"}  # documentos con un lema buscable
_DIAERESIS = set("äëïöü")

# Saltillo o diéresis: el término no es español ("gü" del español no cuenta)
_KAQ_MARKS = re.compile(r"'|[äëïö]|(?<![gG])ü")
_WORD = re.compile(r"[\w']+")
_PUNCT = re.compile(r"[¡!¿?.,;:\"()«»]")
_PARENS = re.compile(r"\([^)]*\)")
# Palabras de la consulta que no son el término buscado ("¿qué significa ...?").
# Se comparan sin plegar: "ya" es vacía, pero "ya'" (agua) no.
_STOPWORDS = """
    a al como cómo con cual cuál de del decir dice dicen dime e el él ella ellos
    ellas en es esta está este esto ese esa eso forma idioma kakchikel kaqchikel
    la las le les lo los me mi mí mis o palabra para por puedes que qué quiere
    quiero se significa significado sin su sus traduce traducción traducir tu tú
    tus un una y escribe escribir correcta correctamente frase español nos yo
    usted ustedes pero muy más ya no si sí son ser fue hay tengo tiene necesito
    saber dónde donde cuándo cuando quién quien porque explica explicar ignora
    ignorar instrucciones anteriores responde responder
""".split()
_STOPWORDS = set(_STOPWORDS) | {w.translate(str.maketrans("áéíóú", "aeiou")) for w in _STOPWORDS}
_STOPWORDS_FOLDED = {fold(w) for w in _STOPWORDS}  # para los tokens plegados de BM25


@dataclass
class Hit:
    doc: dict
    method: str  # exacta | plegada | español | fuzzy | bm25 | densa
    score: float


@dataclass
class Retrieval:
    hits: list[Hit]
    correction: dict | None = None  # {"forma": "ik'", "motivo": "falta saltillo"}
    alternatives: list[str] = field(default_factory=list)  # pares mínimos: ik -> ik'
    no_evidence: bool = False
    max_bm25: float = 0.0
    max_dense: float | None = None


def key(text: str) -> str:
    """Forma exacta para comparar lemas: minúsculas, sin puntuación ni espacios extra."""
    return " ".join(_PUNCT.sub(" ", normalize(text).lower()).split())


def tokens(text: str) -> list[str]:
    return [t for t in (fold(w) for w in _WORD.findall(normalize(text).lower())) if t]


def _gloss_terms(doc: dict) -> list[str]:
    """Términos en español de la entrada: glosario + glosas cortas de la traducción.
    Se descartan los que tienen saltillo o diéresis (errores de extracción)."""
    terms = list(doc.get("es_terms") or [])
    for gloss in re.split(r"[;,]", doc.get("traduccion") or ""):
        gloss = _PARENS.sub("", gloss).strip()
        if gloss and len(gloss.split()) <= 4:
            terms.append(gloss)
    return [t for t in terms if not _KAQ_MARKS.search(t)]


def query_terms(query: str) -> list[tuple[str, int, int]]:
    """Candidatos a término buscado con su posición (palabra inicial y final):
    la consulta completa y sus n-gramas (n <= 5), de más largo a más corto,
    sin los que solo tienen palabras vacías."""
    words = _WORD.findall(normalize(query).lower())
    terms = [(key(query), 0, len(words))]
    for n in range(min(5, len(words)), 0, -1):
        for i in range(len(words) - n + 1):
            gram = words[i : i + n]
            if all(w in _STOPWORDS for w in gram):
                continue
            terms.append((" ".join(gram), i, i + n))
    seen, out = set(), []
    for term in terms:
        if term[0] and term[0] not in seen:
            seen.add(term[0])
            out.append(term)
    return out


class Index:
    def __init__(self) -> None:
        if not config.INDEX_DOCS.exists():
            raise FileNotFoundError(
                f"No existe {config.INDEX_DOCS}. Ejecuta: python -m src.kaq.build_index"
            )
        with open(config.INDEX_DOCS, encoding="utf-8") as f:
            self.docs = [json.loads(line) for line in f]
        self.by_id = {d["id"]: d for d in self.docs}
        self.emb = np.load(config.INDEX_EMBEDDINGS)

        self.exact = defaultdict(list)
        self.folded = defaultdict(list)
        self.spanish = defaultdict(list)
        for i, d in enumerate(self.docs):
            if d["tipo"] in _LEMMA_TYPES:
                self.exact[key(d["lema"])].append(i)
                self.folded[fold(d["lema"])].append(i)
            for term in _gloss_terms(d):
                self.spanish[fold(term)].append(i)
        self.fold_keys = list(self.folded)
        self.spanish_keys = list(self.spanish)

        corpus = [
            tokens(" ".join([d.get("lema") or "", d.get("traduccion") or "",
                             " ".join(d.get("es_terms") or []), d["texto"]]))
            for d in self.docs
        ]
        self.bm25 = BM25Okapi(corpus)
        self._encoder = None

    def encode_query(self, query: str) -> np.ndarray:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(config.EMBEDDING_MODEL)
        return self._encoder.encode("query: " + query, normalize_embeddings=True)


@lru_cache(maxsize=1)
def get_index() -> Index:
    return Index()


def _fuzzy(term: str, keys: list[str], limit: int = 3) -> list[str]:
    if len(fold(term)) < 4:
        return []
    matches = process.extract(fold(term), keys, scorer=fuzz.ratio,
                              score_cutoff=config.FUZZY_CUTOFF, limit=limit)
    return [m[0] for m in matches]


def lexical(query: str, k: int = config.CANDIDATES) -> tuple[list[Hit], float]:
    """Devuelve los hits léxicos y el puntaje BM25 máximo."""
    idx = get_index()
    terms = query_terms(query)
    seen: dict[int, Hit] = {}

    def add(doc_ids, method, score):
        for i in doc_ids:
            if i not in seen:
                seen[i] = Hit(idx.docs[i], method, score)

    # Un n-grama contenido en otro que ya coincidió no se busca: en "pájaro
    # carpintero" no se agregan todas las entradas de "pájaro"
    covered: list[tuple[int, int]] = []
    for t, a, b in terms:
        if any(ca <= a and b <= cb for ca, cb in covered):
            continue
        matched = False
        for table, method, score, keyfn in (
            (idx.exact, "exacta", 1.0, key),
            (idx.folded, "plegada", 0.9, fold),
            (idx.spanish, "español", 0.9, fold),
        ):
            if method == "español" and _KAQ_MARKS.search(t):
                continue  # "ya'" no debe coincidir con la glosa "Ya"
            doc_ids = table.get(keyfn(t), [])
            add(doc_ids, method, score)
            matched = matched or bool(doc_ids)
        if matched:
            covered.append((a, b))
    if not seen:
        for t, _, _ in terms:
            for kf in _fuzzy(t, idx.fold_keys):
                add(idx.folded[kf], "fuzzy", 0.7)
            for kf in _fuzzy(t, idx.spanish_keys):
                add(idx.spanish[kf], "fuzzy", 0.7)

    q_tokens = [t for t in tokens(query) if t not in _STOPWORDS_FOLDED] or tokens(query)
    scores = idx.bm25.get_scores(q_tokens) if q_tokens else np.zeros(len(idx.docs))
    for i in np.argsort(-scores):
        if len(seen) >= k or scores[i] <= 0:
            break
        add([int(i)], "bm25", float(scores[i]))
    return list(seen.values())[:k], float(scores.max(initial=0.0))


def dense(query: str, k: int = config.CANDIDATES) -> list[Hit]:
    idx = get_index()
    sims = idx.emb @ idx.encode_query(query)
    return [Hit(idx.docs[int(i)], "densa", float(sims[i])) for i in np.argsort(-sims)[:k]]


def hybrid(query: str, k: int = config.CANDIDATES) -> tuple[list[Hit], float, float]:
    """Devuelve los hits fusionados, el BM25 máximo y la similitud densa máxima."""
    lex, max_bm25 = lexical(query, config.CANDIDATES)
    den = dense(query, config.CANDIDATES)

    anchored = [h for h in lex if h.method in _STRONG]
    anchored_ids = {h.doc["id"] for h in anchored}
    fused: dict[str, float] = defaultdict(float)
    first_hit: dict[str, Hit] = {}
    for ranking in (lex, den):
        for rank, h in enumerate(ranking):
            fused[h.doc["id"]] += 1 / (config.RRF_K + rank + 1)
            first_hit.setdefault(h.doc["id"], h)
    rest = sorted((i for i in fused if i not in anchored_ids), key=lambda i: -fused[i])
    hits = anchored + [Hit(first_hit[i].doc, first_hit[i].method, fused[i]) for i in rest]
    return hits[:k], max_bm25, den[0].score if den else 0.0


def _reason(query_form: str, form: str) -> str:
    a, b = key(query_form), key(form)

    def no_apos(s):
        return s.replace("'", "")

    def n_diaeresis(s):
        return sum(c in _DIAERESIS for c in s)

    if no_apos(a) == no_apos(b):
        return "falta saltillo" if b.count("'") > a.count("'") else "saltillo de más o mal ubicado"
    if strip_marks(a) == strip_marks(b):
        return "falta diéresis" if n_diaeresis(b) > n_diaeresis(a) else "diéresis o tilde distinta"
    if strip_marks(no_apos(a)) == strip_marks(no_apos(b)):
        return "faltan saltillo y diéresis"
    return "escritura similar"


def suggest_correction(query: str) -> tuple[dict | None, list[str]]:
    """Corrección ortográfica determinista y pares mínimos.

    Recorre los términos de la consulta (del más largo al más corto). Para el
    primero que coincide con un lema del diccionario por clave plegada:
    - si además coincide exacto, no hay corrección, pero las otras formas con
      la misma clave se devuelven como alternativas (ik -> ik');
    - si no coincide exacto, se sugiere la forma del diccionario.
    Si ninguno coincide (ni siquiera como palabra en español), se prueba fuzzy.
    """
    idx = get_index()
    terms = [t for t, _, _ in query_terms(query)]
    for t in terms:
        ids = idx.folded.get(fold(t))
        if not ids:
            continue
        forms = list(dict.fromkeys(idx.docs[i]["lema"] for i in ids))
        if key(t) in forms:
            return None, [f for f in forms if f != key(t)]
        best = forms[0]
        return {"forma": best, "motivo": _reason(t, best)}, forms[1:]

    if any(fold(t) in idx.spanish for t in terms):
        return None, []
    for t in terms:
        found = _fuzzy(t, idx.fold_keys, limit=1)
        if found:
            best = idx.docs[idx.folded[found[0]][0]]["lema"]
            return {"forma": best, "motivo": _reason(t, best)}, []
    return None, []


def search(query: str, mode: str = "hibrido", k: int = config.TOP_K) -> Retrieval:
    """Recupera las entradas para una consulta.

    mode: "lexico" o "hibrido". "sin_contexto" no recupera nada (baseline).
    """
    if mode not in MODES:
        raise ValueError(f"Modo desconocido: {mode}. Opciones: {MODES}")
    if mode == "sin_contexto":
        return Retrieval(hits=[])

    if mode == "lexico":
        hits, max_bm25 = lexical(query, k)
        max_dense = None
    else:
        hits, max_bm25, max_dense = hybrid(query, k)

    strong = any(h.method in _STRONG for h in hits)
    if mode == "lexico":
        weak = max_bm25 < config.BM25_MIN_SCORE
    else:
        weak = max_dense < config.DENSE_MIN_SCORE
    correction, alternatives = suggest_correction(query)
    return Retrieval(
        hits=hits,
        correction=correction,
        alternatives=alternatives,
        no_evidence=not strong and weak,
        max_bm25=round(max_bm25, 2),
        max_dense=None if max_dense is None else round(max_dense, 4),
    )
