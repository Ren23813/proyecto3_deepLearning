"""Ingesta del Diccionario Bilingüe Kaqchikel OKMA (PDF con capa de texto).

Produce:
- okma_entries.jsonl: entradas kaq -> es (tipo "entrada") y frases idiomáticas (tipo "frase")
- okma_glosario.jsonl: glosario es -> kaq, enlazado a las entradas por clave plegada
- stats.json y muestra_revision.md: control de calidad

Formato de una entrada en el PDF:
    LEMA [morfemas] [transcripción] [-raíz] || cat. Definición. Ejemplo kaq. Traducción es.
    || otra acepción ... Sin. X. Var. Y. Véase: Z. Pl. W. Forma poseída: V. Campo Léxico: U.

Uso:  python -m src.kaq.ingest_okma
"""
import json
import random
import re
from collections import Counter

from pypdf import PdfReader

from . import config
from .normalize import fold, normalize

# --- Limpieza de página 

_PAGE_HEADER = re.compile(
    r"^\d*\s*(Rusoltzij ri Kaqchikel|Diccionario Estándar Bilingüe Kaqchikel-Español"
    r"|Frases idiomáticas|Glosario Español-Kaqchikel)"
)


def _is_top_noise(line: str) -> bool:
    """Encabezado, número de página decorativo o letra de sección."""
    s = line.strip()
    return (
        not s
        or len(s) <= 3
        or not re.search(r"[A-Za-zÁÉÍÓÚÄËÏÖÜÑáéíóúäëïöüñ]", s)
        or bool(_PAGE_HEADER.match(s))
    )


_CAPTION_SLASH = re.compile(r"/(\s|[A-ZÁÉÍÓÚÑ]|$)")


def _is_caption(line: str) -> bool:
    """Pies de imagen tipo "Kär/Pez" o "Ch’äj/ árbol de pino".

    En el cuerpo la barra solo aparece en "Piojoso/a" o "Ellos/as", seguida
    de minúscula; los pies además suelen usar ’ en vez de '.
    """
    if "/" not in line or "||" in line:
        return False
    return "’" in line or (len(line.strip()) < 60 and bool(_CAPTION_SLASH.search(line)))


def page_lines(reader: PdfReader, first: int, last: int):
    """Genera (página, línea) ya sin ruido de encabezados ni pies de imagen."""
    for page in range(first, last + 1):
        lines = (reader.pages[page - 1].extract_text() or "").split("\n")
        while lines and _is_top_noise(lines[0]):
            lines.pop(0)
        for line in lines:
            if line.strip() and not _is_caption(line):
                yield page, line.rstrip()


# --- Detección de inicio de entrada 

_PRE_REST = re.compile(r"^(?P<pre>[^\[|]*)(?P<rest>(\[|\|\|).*)?$")
# Morfemas en ortografía kaqchikel: "achik'-a-n-el", "-xib'al", "k-i'" (sin tildes del español)
_MORPH_TOKEN = re.compile(r"^-?[a-zäëïöü'\-]+$")


def _is_upper_token(tok: str) -> bool:
    return any(c.isalpha() for c in tok) and tok == tok.upper()


def _split_head(line: str):
    """Si la línea empieza con un LEMA en mayúsculas, devuelve (lema, morfemas, resto).

    El corchete de la transcripción es la señal más fiable de inicio de entrada;
    sin él, se exige "||" y que lo intermedio parezca morfología kaqchikel.
    """
    m = _PRE_REST.match(line.strip())
    if not m:
        return None
    tokens = m.group("pre").split()
    head = []
    while tokens and _is_upper_token(tokens[0]):
        head.append(tokens.pop(0))
    if not head or head[0].startswith("¿"):
        return None
    rest = m.group("rest") or ""
    if rest.startswith("["):
        if len(tokens) > 4:
            return None
    elif not all(_MORPH_TOKEN.match(t) for t in tokens):
        return None
    return " ".join(head), " ".join(tokens) or None, rest


def _is_entry_start(lines: list[tuple[int, str]], i: int) -> bool:
    parts = _split_head(lines[i][1])
    if not parts:
        return False
    if parts[2]:
        return True
    # La transcripción puede quedar en la línea siguiente, a veces precedida
    # por el final de los morfemas ("saq-a-" / "tura's [tewulAX ...")
    if i + 1 >= len(lines):
        return False
    m = _PRE_REST.match(lines[i + 1][1].strip())
    return bool(
        m
        and (m.group("rest") or "").startswith("[")
        and all(_MORPH_TOKEN.match(t) for t in m.group("pre").split())
    )


def _join(lines: list[str]) -> str:
    """Une líneas; si una termina en guion (raíz partida) no agrega espacio."""
    out = ""
    for line in lines:
        line = line.strip()
        if out.endswith("-"):
            out += line
        else:
            out = f"{out} {line}" if out else line
    return normalize(out)


# --- Extracción de campos 

_STOP = r"(?=\.\s|\.$|\||Véase|Sin\.|Var\.|Campo|Forma poseída|Pl\.|$)"
_FIELDS = {
    "sinonimos": re.compile(r"\bSin\.\s*(.+?)" + _STOP),
    "variantes": re.compile(r"\bVar\.\s*(.+?)" + _STOP),
    "vease": re.compile(r"Véase:\s*(.+?)" + _STOP),
    "plural": re.compile(r"\bPl\.\s*(.+?)" + _STOP),
    "forma_poseida": re.compile(r"Forma\s+poseída:\s*(.+?)" + _STOP),
    "campo_lexico": re.compile(r"Campo\s+Léxico:\s*(.+?)" + _STOP),
}
_LIST_FIELDS = {"sinonimos", "variantes", "vease", "campo_lexico"}
_LABELS = ("Forma poseída", "Sin.", "Var.", "Véase", "Pl.", "Campo Léxico", "Nota", "Lit.")
# Abreviaturas gramaticales al inicio de una acepción: "s. 1a.", "v.t. d.", "adj. y s.",
# incluidas erratas del PDF como "s 1a." o ".v.i."
_CATEGORY_TOKEN = re.compile(r"^(?:\.?[a-záéíóúü][a-záéíóúü.]{0,13}|\d[a-z]\.?)$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# Saltillo o diéresis: señal de texto en kaqchikel ("gü" del español no cuenta)
_KAQ_MARKS = re.compile(r"'|[äëïö]|(?<![gG])ü")
# Lo que queda entre la transcripción y la definición: raíz opcional y separador
# ("||", "|", "||." o la errata "ns")
_AFTER_TRANSCRIPTION = re.compile(r"^\s*(?P<root>-[^\s|]+)?\s*(?:\|\|?\.?|ns\b)?\s*")


def _extract_fields(body: str) -> dict:
    fields = {}
    for name, rx in _FIELDS.items():
        values = [v.strip(" .,") for v in rx.findall(body)]
        if name in _LIST_FIELDS:
            items = [x.strip(" .") for v in values for x in v.split(",")]
            fields[name] = list(dict.fromkeys(x for x in items if x))
        else:
            fields[name] = values[0] if values else None
    return fields


def _split_category(sense: str) -> tuple[str | None, str]:
    tokens = sense.split()
    n = 0
    while n < min(len(tokens), 5) and _CATEGORY_TOKEN.match(tokens[n]):
        n += 1
    if n == 0:
        return None, sense
    return " ".join(tokens[:n]), " ".join(tokens[n:])


def _glosses(definition: str) -> str | None:
    """Glosas en español al inicio de la acepción ("Pita. Cordel. Mecate.").

    Se salta un "Forma poseída:" inicial y se detiene en la primera oración
    larga, en otro campo etiquetado o en una oración con saltillo o diéresis
    (señal de que empieza el ejemplo en kaqchikel).
    """
    out = []
    for sent in _SENTENCE_SPLIT.split(definition):
        sent = sent.strip()
        if not out and sent.startswith("Forma poseída"):
            continue
        max_words = 10 if not out else 5
        if (
            not sent
            or len(sent.split()) > max_words
            or sent.startswith(_LABELS)
            or _KAQ_MARKS.search(sent)
            or len(out) == 4
        ):
            break
        out.append(sent.rstrip("."))
    return "; ".join(out) or None


def _has_example(senses: list[str]) -> bool:
    """Heurística: alguna oración de 4+ palabras escrita en kaqchikel."""
    for sense in senses:
        for sent in _SENTENCE_SPLIT.split(sense):
            if len(sent.split()) >= 4 and _KAQ_MARKS.search(sent) and not sent.startswith(_LABELS):
                return True
    return False


def parse_entry(lines: list[tuple[int, str]], tipo: str) -> dict:
    head, morph, _ = _split_head(lines[0][1])
    text = _join([l for _, l in lines])

    # La transcripción [...] usa una fuente rota y se descarta; lo que sigue es
    # la raíz opcional y el cuerpo. Sin transcripción, el cuerpo empieza en "||".
    bar = text.find("||")
    tr = re.search(r"\[[^\]]*\]", text)
    if tr and (bar == -1 or tr.start() < bar):
        morph = text[len(head) : tr.start()].strip() or None  # incluye morfemas partidos en 2 líneas
        after = text[tr.end() :]
        m = _AFTER_TRANSCRIPTION.match(after)
        root, body = m.group("root"), after[m.end() :]
    else:
        root, body = None, text[bar + 2 :] if bar != -1 else text[len(head) :]

    # lstrip quita el punto suelto de "||." y de erratas como ".v.i."
    senses = [s.strip().lstrip(". ") for s in body.split("||")]
    senses = [s for s in senses if s]
    category, first_def = _split_category(senses[0]) if senses else (None, "")

    return {
        "tipo": tipo,
        "lema": head.lower(),
        "lema_original": head,
        "lema_fold": fold(head),
        "morfologia": morph,
        "raiz": root or None,
        "categoria": category,
        "traduccion": _glosses(first_def),
        "acepciones": senses,
        **_extract_fields(body),
        "tiene_ejemplo": _has_example(senses),
        "texto": " ".join(filter(None, [head, morph, root, "||", " || ".join(senses)])),
        "pagina": lines[0][0],
        "pagina_fin": lines[-1][0],
    }


def parse_entries(reader: PdfReader, pages: tuple[int, int], tipo: str) -> list[dict]:
    lines = list(page_lines(reader, *pages))
    starts = [i for i in range(len(lines)) if _is_entry_start(lines, i)]
    entries = []
    for a, b in zip(starts, starts[1:] + [len(lines)]):
        entries.append(parse_entry(lines[a:b], tipo))
    return entries


# Letra de sección a mitad de página ("R", "CH"); en el glosario no hay otras
# líneas tan cortas en mayúsculas
_SECTION_LETTER = re.compile(r"^[A-ZÁÉÍÓÚÑ]{1,2}$")


def parse_glosario(reader: PdfReader, pages: tuple[int, int]) -> list[dict]:
    """Líneas "Español || LEMA || cat."; algunas se parten en dos renglones."""
    records, buf, buf_page = [], "", None

    def flush():
        parts = [p.strip() for p in buf.split("||")]
        if len(parts) >= 2 and parts[0] and parts[1]:
            records.append({
                "tipo": "glosario",
                "espanol": normalize(parts[0]),
                "kaqchikel": normalize(parts[1]).lower(),
                "kaqchikel_fold": fold(parts[1]),
                "categoria": normalize(" ".join(parts[2:])) or None,
                "pagina": buf_page,
            })

    for page, line in page_lines(reader, *pages):
        if _SECTION_LETTER.match(line.strip()):
            continue
        if buf and buf.count("||") >= 2 and "||" in line:
            flush()
            buf = ""
        if not buf:
            buf_page = page
        buf = f"{buf} {line.strip()}".strip()
    if buf:
        flush()
    return records


# --- Control de calidad 

# Letras iniciales del alfabeto kaqchikel (las compuestas primero)
_ALPHABET = sorted(
    "a ä b' ch ch' e ë i ï j k k' l m n o ö p q q' r s t t' tz tz' u ü w x y".split(),
    key=len, reverse=True,
)


def _initial(lema: str) -> str:
    s = lema.lstrip("¡-")
    return next((l for l in _ALPHABET if s.startswith(l)), s[:1])


def build_stats(entries: list[dict], glosario: list[dict]) -> dict:
    main = [e for e in entries if e["tipo"] == "entrada"]

    def pct(pred, items):
        return round(100 * sum(1 for x in items if pred(x)) / max(len(items), 1), 1)

    chars = [len(e["texto"]) for e in entries]
    return {
        "entradas": len(main),
        "frases_idiomaticas": sum(1 for e in entries if e["tipo"] == "frase"),
        "glosario_es_kaq": len(glosario),
        "glosario_enlazado_pct": pct(lambda g: g["ref_ids"], glosario),
        "lemas_unicos": len({e["lema"] for e in main}),
        "pct_con_categoria": pct(lambda e: e["categoria"], main),
        "pct_con_traduccion": pct(lambda e: e["traduccion"], main),
        "pct_con_ejemplo": pct(lambda e: e["tiene_ejemplo"], main),
        "pct_con_campo_lexico": pct(lambda e: e["campo_lexico"], main),
        "pct_multipagina": pct(lambda e: e["pagina_fin"] != e["pagina"], main),
        "longitud_media_caracteres": round(sum(chars) / len(chars)),
        # Aproximación de ~4 caracteres por token; se mide con la API más adelante
        "tokens_aprox_por_entrada": round(sum(chars) / len(chars) / 4),
        "max_caracteres": max(chars),
        "entradas_por_letra": dict(Counter(_initial(e["lema"]) for e in main).most_common()),
        "categorias_top": dict(Counter(e["categoria"] for e in main).most_common(15)),
    }


def write_sample(entries: list[dict], n: int = 30, seed: int = 42) -> None:
    sample = random.Random(seed).sample(entries, n)
    out = ["# Muestra para revisión manual contra el PDF", "",
           "Marcar cada entrada como OK o con el problema encontrado.", ""]
    for e in sorted(sample, key=lambda e: e["pagina"]):
        out += [
            f"## {e['id']} · {e['lema_original']} · pág. {e['pagina']}",
            f"- categoría: `{e['categoria']}` · traducción: `{e['traduccion']}`",
            f"- texto: {e['texto']}",
            "- revisión: ",
            "",
        ]
    config.OKMA_SAMPLE.write_text("\n".join(out), encoding="utf-8")


def _write_jsonl(path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    reader = PdfReader(config.OKMA_PDF)
    entries = parse_entries(reader, config.OKMA_DICT_PAGES, "entrada")
    entries += parse_entries(reader, config.OKMA_FRASES_PAGES, "frase")

    # El PDF repite algunas entradas tal cual (p. ej. "Ö"); los homónimos se conservan
    seen, unique = set(), []
    for e in entries:
        key = (e["lema"], e["texto"])
        if key not in seen:
            seen.add(key)
            unique.append(e)
    entries = [{"id": f"okma-{i:05d}", "diccionario": "OKMA", **e} for i, e in enumerate(unique, 1)]

    by_fold: dict[str, list[str]] = {}
    for e in entries:
        by_fold.setdefault(e["lema_fold"], []).append(e["id"])
    glosario = [
        {"id": f"okmag-{i:05d}", "diccionario": "OKMA", **g, "ref_ids": by_fold.get(g["kaqchikel_fold"], [])}
        for i, g in enumerate(parse_glosario(reader, config.OKMA_GLOSARIO_PAGES), 1)
    ]

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(config.OKMA_ENTRIES, entries)
    _write_jsonl(config.OKMA_GLOSARIO, glosario)
    stats = build_stats(entries, glosario)
    config.OKMA_STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    write_sample([e for e in entries if e["tipo"] == "entrada"])

    print(json.dumps({k: v for k, v in stats.items() if not isinstance(v, dict)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
