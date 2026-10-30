"""Construye el índice de recuperación a partir de las entradas procesadas

Salidas en data/index/:
- docs.jsonl: un documento por entrada, con los términos
  en español del glosario que apuntan a ella (es_terms).
- embeddings.npy: matriz (n_docs, 768) de embeddings e5 normalizados.
- meta.json: modelo, dimensión, número de documentos y fecha.


Uso: python -m src.kaq.build_index
"""
import json
import time
from collections import defaultdict
from datetime import datetime

import numpy as np

from . import config


def _read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_docs() -> list[dict]:
    docs = _read_jsonl(config.OKMA_ENTRIES)

    # Glosario es -> kaq: sus términos en español se agregan a la entrada enlazada
    es_terms = defaultdict(list)
    for g in _read_jsonl(config.OKMA_GLOSARIO):
        for ref in g["ref_ids"]:
            es_terms[ref].append(g["espanol"])
    for d in docs:
        d["es_terms"] = list(dict.fromkeys(es_terms.get(d["id"], [])))

    if config.L2_ENTRIES.exists():
        docs += _read_jsonl(config.L2_ENTRIES)
    else:
        print(f"Aviso: no existe {config.L2_ENTRIES.name}, se indexa solo OKMA")
    return docs


def passage(doc: dict) -> str:
    """Texto que se embebe. e5 requiere el prefijo "passage: " en los documentos."""
    text = doc["texto"]
    if doc.get("es_terms"):
        text += " Español: " + ", ".join(doc["es_terms"])
    return "passage: " + text


def _previous_embeddings() -> dict[str, np.ndarray]:
    """Embeddings del índice anterior por texto de pasaje, si el modelo no cambió.
    Con caché solo se embeben los documentos nuevos o modificados.
    """
    if not (config.INDEX_META.exists() and config.INDEX_DOCS.exists() and config.INDEX_EMBEDDINGS.exists()):
        return {}
    meta = json.loads(config.INDEX_META.read_text(encoding="utf-8"))
    if meta.get("modelo_embeddings") != config.EMBEDDING_MODEL:
        return {}
    old_docs = _read_jsonl(config.INDEX_DOCS)
    old_emb = np.load(config.INDEX_EMBEDDINGS)
    if len(old_docs) != len(old_emb):
        return {}
    return {passage(d): v for d, v in zip(old_docs, old_emb)}


def embed(texts: list[str]) -> np.ndarray:
    cache = _previous_embeddings()
    missing = [t for t in dict.fromkeys(texts) if t not in cache]
    print(f"Embeddings: {len(texts) - len(missing)} reutilizados, {len(missing)} nuevos")
    if missing:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(config.EMBEDDING_MODEL)
        new = model.encode(missing, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
        cache.update(zip(missing, new))
    return np.stack([cache[t] for t in texts]).astype(np.float32)


def main() -> None:
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    docs = load_docs()

    t0 = time.time()
    emb = embed([passage(d) for d in docs])
    seconds = round(time.time() - t0, 1)

    with open(config.INDEX_DOCS, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    np.save(config.INDEX_EMBEDDINGS, emb)
    meta = {
        "modelo_embeddings": config.EMBEDDING_MODEL,
        "dimension": int(emb.shape[1]),
        "documentos": len(docs),
        "por_diccionario": {
            src: sum(1 for d in docs if d["diccionario"] == src)
            for src in sorted({d["diccionario"] for d in docs})
        },
        "segundos_embeddings": seconds,
        "construido": datetime.now().isoformat(timespec="seconds"),
    }
    config.INDEX_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
