"""Pipeline completo: recuperación -> prompt -> Claude -> post-validación.

Función pública: answer(query, mode, k) -> dict con el esquema (pares mínimos, p. ej. ik -> ["ik'"]) y
"_meta" (ids recuperados, latencia, tokens, costo...)

Post-validación (el diccionario manda sobre el LLM):
- Se descartan las entradas cuyo id no fue recuperado (citas inventadas) y se
  cuentan en _meta.citas_descartadas
- La fuente (diccionario, página), la palabra y la categoría salen del
  documento recuperado, no del texto del LLM
- La corrección ortográfica determinista reemplaza a la del LLM; una corrección
  del LLM solo se conserva si es un lema recuperado.
- Si no queda ninguna entrada válida, encontrado = false.

Uso rápido: python -m src.kaq.pipeline "ik" [hibrido|lexico|sin_contexto]
"""
import json
import sys
import time
from datetime import datetime

from . import config, prompts
from .generate import GenerationError, generate
from .retrieval import _KAQ_MARKS, _LEMMA_TYPES, MODES, get_index, key, search

MAX_QUERY_CHARS = 500
NOT_FOUND = "La consulta no aparece en los diccionarios consultados."


def warmup() -> None:
    """Carga el índice y el modelo de embeddings (aprox. 10s la primera vez)."""
    get_index().encode_query("calentamiento")


def _empty_response(query: str, explicacion: str, advertencia: str | None = None) -> dict:
    return {
        "encontrado": False,
        "idioma_consulta": "kaq" if _KAQ_MARKS.search(query) else "es",
        "correccion_sugerida": None,
        "alternativas": [],
        "entradas": [],
        "explicacion": explicacion,
        "advertencia": advertencia,
    }


def _validate(llm, retrieved: dict[str, dict], correction: dict | None, with_context: bool):
    """Aplica las reglas de post-validación. Devuelve (entradas, corrección, descartadas)."""
    entradas, discarded, seen = [], [], set()
    for e in llm.entradas if llm.encontrado else []:
        item = e.model_dump()
        item["ejemplos"] = item["ejemplos"][:2]
        if with_context:
            doc = retrieved.get(e.id)
            if doc is None:
                discarded.append(e.id)
                continue
            if e.id in seen:
                continue
            seen.add(e.id)
            if doc["tipo"] in _LEMMA_TYPES:
                item["palabra"] = doc["lema"]
                item["categoria"] = doc.get("categoria") or item["categoria"]
            item["fuente"] = {"diccionario": doc["diccionario"], "pagina": doc["pagina"]}
        else:
            item["id"] = None
            item["fuente"] = None
        entradas.append(item)

    llm_corr = llm.correccion_sugerida.model_dump() if llm.correccion_sugerida else None
    if not with_context:
        final_corr = llm_corr
    elif correction:
        final_corr = correction
    elif llm_corr and any(key(llm_corr["forma"]) == key(d["lema"]) for d in retrieved.values()):
        final_corr = llm_corr
    else:
        final_corr = None
    return entradas, final_corr, discarded


def _log(record: dict) -> None:
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.REQUESTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def answer(query: str, mode: str = "hibrido", k: int = config.TOP_K,
           model: str = config.MODEL, prompt_version: str = config.PROMPT_VERSION) -> dict:
    if mode not in MODES:
        raise ValueError(f"Modo desconocido: {mode}. Opciones: {MODES}")
    t0 = time.perf_counter()
    query = " ".join((query or "").split())[:MAX_QUERY_CHARS]
    meta = {
        "mode": mode, "model": None, "prompt_version": prompt_version, "k": k,
        "retrieved_ids": [], "retrieved": [], "no_evidence": None,
        "max_bm25": None, "max_dense": None, "llm_called": False,
        "latency_ms": 0, "retrieval_ms": 0, "llm_ms": 0,
        "input_tokens": 0, "output_tokens": 0, "cache_write_tokens": 0, "cache_read_tokens": 0,
        "cost_usd": 0.0, "stop_reason": None, "request_id": None,
        "citas_descartadas": [], "error": None,
    }

    def finish(result: dict) -> dict:
        meta["latency_ms"] = round((time.perf_counter() - t0) * 1000)
        result["_meta"] = meta
        _log({"timestamp": datetime.now().isoformat(timespec="seconds"), "query": query, **result})
        return result

    if not query:
        return finish(_empty_response(query, "Escribe una palabra o frase en kaqchikel o en español."))

    with_context = mode != "sin_contexto"
    ret = search(query, mode, k)
    retrieved = {h.doc["id"]: h.doc for h in ret.hits}
    meta.update(
        retrieval_ms=round((time.perf_counter() - t0) * 1000),
        retrieved_ids=list(retrieved),
        retrieved=[{"id": h.doc["id"], "metodo": h.method, "score": round(h.score, 4)} for h in ret.hits],
        no_evidence=ret.no_evidence if with_context else None,
        max_bm25=ret.max_bm25 if with_context else None,
        max_dense=ret.max_dense,
    )

    if with_context and ret.no_evidence and config.SHORT_CIRCUIT_NO_EVIDENCE:
        result = _empty_response(query, NOT_FOUND)
        result["correccion_sugerida"] = ret.correction
        return finish(result)

    system = (prompts.SYSTEM if with_context else prompts.SYSTEM_NO_CONTEXT)[prompt_version]
    user = prompts.user_message(query, [h.doc for h in ret.hits] if with_context else None,
                                ret.correction, ret.alternatives)
    meta["model"] = model
    try:
        gen = generate(system, user, model)
    except GenerationError as e:
        meta["error"] = e.detail
        return finish(_empty_response(query, e.user_message, "No se pudo generar la respuesta."))

    meta.update(
        llm_called=True, llm_ms=gen.latency_ms, input_tokens=gen.input_tokens,
        output_tokens=gen.output_tokens, cache_write_tokens=gen.cache_write_tokens,
        cache_read_tokens=gen.cache_read_tokens, cost_usd=gen.cost_usd,
        stop_reason=gen.stop_reason, request_id=gen.request_id,
    )
    llm = gen.response
    entradas, correction, discarded = _validate(llm, retrieved, ret.correction, with_context)
    meta["citas_descartadas"] = discarded
    encontrado = llm.encontrado and bool(entradas)
    return finish({
        "encontrado": encontrado,
        "idioma_consulta": llm.idioma_consulta,
        "correccion_sugerida": correction,
        "alternativas": ret.alternatives,
        "entradas": entradas,
        "explicacion": llm.explicacion if encontrado or not llm.encontrado else NOT_FOUND,
        "advertencia": llm.advertencia,
    })


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "ik"
    m = sys.argv[2] if len(sys.argv) > 2 else "hibrido"
    print(json.dumps(answer(q, m), ensure_ascii=False, indent=2))
