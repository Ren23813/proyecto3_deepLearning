"""Ingesta del texto pedagógico "Kaqchikel L2" (segundo PDF).

El PDF no tiene capa de texto: cada página es una imagen escaneada. El proceso
tiene tres pasos:

1. extract: guarda la imagen de cada página en data/raw/l2_pages/pNNN.jpg
2. Transcripción (OCR con Claude vision): por cada página se escribe
   data/processed/l2_ocr/pNNN.json con este formato:
       {
         "pagina_pdf": 21,
         "pagina_impresa": 21,            # null si no se ve en el escaneo
         "unidad": "Nab'ey Tanaj (Unidad No. 1)",
         "secciones": [{"titulo": "...", "texto": "..."}],
         "vocabulario": [{"kaq": "q'e'ël", "es": "recostado",
                          "categoria": null, "tema": "Posicionales"}]
       } 
3. build: une las páginas en data/processed/l2_entries.jsonl con el mismo
   formato de documento que las entradas del OKMA:
   - tipo "vocabulario": un par kaqchikel-español (tablas y listas del libro)
   - tipo "texto": una sección explicativa (cultura, gramática)

Uso:
    python -m src.kaq.ingest_l2 extract
    python -m src.kaq.ingest_l2 build
"""
import json
import sys

from pypdf import PdfReader

from . import config
from .normalize import fold, normalize

SOURCE = "L2 Dideduc"
MAX_CHARS = 1200  # las secciones largas se parten en fragmentos de este tamaño


def extract() -> None:
    config.L2_PAGES_DIR.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(config.L2_PDF)
    for n, page in enumerate(reader.pages, start=1):
        (config.L2_PAGES_DIR / f"p{n:03d}.jpg").write_bytes(page.images[0].data)
    print(f"{len(reader.pages)} páginas en {config.L2_PAGES_DIR}")


def _chunks(text: str) -> list[str]:
    """Parte un texto largo por oraciones en fragmentos de hasta MAX_CHARS."""
    out, current = [], ""
    for sent in text.replace("\n", " ").split(". "):
        if current and len(current) + len(sent) > MAX_CHARS:
            out.append(current.strip())
            current = ""
        current += sent + ". "
    out.append(current.strip().rstrip("."))
    return [c for c in out if c]


def _pages(page: dict) -> dict:
    """Se cita la página impresa (la que ve quien lee el libro). Esto porque las primeras
    páginas del PDF están desordenadas respecto al impreso: se guarda aparte."""
    cited = page.get("pagina_impresa") or page["pagina_pdf"]
    return {"pagina": cited, "pagina_fin": cited, "pagina_pdf": page["pagina_pdf"]}


def _vocab_doc(item: dict, page: dict) -> dict:
    kaq, es = normalize(item["kaq"]), normalize(item["es"])
    tema = item.get("tema") or page.get("unidad")
    texto = f"{kaq} || {es}." + (f" Tema: {tema}." if tema else "")
    return {
        "diccionario": SOURCE,
        "tipo": "vocabulario",
        "lema": kaq.lower(),
        "lema_original": kaq,
        "lema_fold": fold(kaq),
        "categoria": item.get("categoria"),
        "traduccion": es,
        "es_terms": [es],
        "tema": tema,
        "texto": texto,
        **_pages(page),
    }


def _section_docs(section: dict, page: dict) -> list[dict]:
    titulo = normalize(section.get("titulo") or page.get("unidad") or "")
    return [
        {
            "diccionario": SOURCE,
            "tipo": "texto",
            "lema": titulo.lower(),
            "lema_original": titulo,
            "lema_fold": fold(titulo),
            "categoria": None,
            "traduccion": None,
            "es_terms": [],
            "tema": page.get("unidad"),
            "texto": f"{titulo}: {normalize(chunk)}" if titulo else normalize(chunk),
            **_pages(page),
        }
        for chunk in _chunks(section["texto"])
    ]


def build() -> None:
    pages = sorted(config.L2_OCR_DIR.glob("p*.json"))
    if not pages:
        sys.exit(f"No hay transcripciones en {config.L2_OCR_DIR}")
    docs, seen = [], set()
    for path in pages:
        page = json.loads(path.read_text(encoding="utf-8"))
        for item in page.get("vocabulario", []):
            doc = _vocab_doc(item, page)
            pair = (doc["lema"], doc["traduccion"].lower())
            if pair not in seen:  # el libro repite pares en ejercicios
                seen.add(pair)
                docs.append(doc)
        for section in page.get("secciones", []):
            docs += _section_docs(section, page)
    for n, doc in enumerate(docs, start=1):
        doc["id"] = f"l2-{n:05d}"

    with open(config.L2_ENTRIES, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps({"id": doc.pop("id"), **doc}, ensure_ascii=False) + "\n")
    vocab = sum(1 for d in docs if d["tipo"] == "vocabulario")
    print(f"{len(pages)} páginas: {vocab} pares de vocabulario, {len(docs) - vocab} fragmentos de texto")


if __name__ == "__main__":
    {"extract": extract, "build": build}[sys.argv[1] if len(sys.argv) > 1 else "build"]()
