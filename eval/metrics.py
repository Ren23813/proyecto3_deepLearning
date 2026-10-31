"""Métricas automáticas del conjunto de evaluación (eval/cases.jsonl).

Por caso (score_case):
- exacto: la respuesta es correcta según reglas verificables
    - si debe abstenerse: encontrado=false y sin contenido prohibido
    - si no: encontrado=true, algún lema de referencia entre las entradas,
      alguna palabra clave en el texto, la corrección esperada (si aplica) y
      ningún contenido prohibido en las entradas
- alucinacion: afirma algo (encontrado=true) y no es exacto
- abstencion_correcta / falsa_abstencion
- correccion_ok, alternativa_ok (pares mínimos), inyeccion_resistida
- rango del primer gold_id en retrieved_ids (para recall@k y MRR)

Por modo (summarize): promedios de lo anterior, recall@1/3/5, MRR, latencia
media y p95, tokens y costo por solicitud. Si hay juicios del LLM-as-judge,
también la fidelidad media (0-2) y la tasa de alucinación según el juez.
"""
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.kaq.normalize import normalize, strip_marks  # noqa: E402
from src.kaq.retrieval import key  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
CASES = EVAL_DIR / "cases.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
RECALL_KS = (1, 3, 5, 10)


def load_cases(path: Path = CASES) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _plain(text: str) -> str:
    """Minúsculas sin tildes ni diéresis, para buscar palabras clave en español."""
    return strip_marks(normalize(text or "")).lower()


def _has_word(text: str, words: list[str]) -> bool:
    """Alguna palabra aparece al inicio de una palabra del texto ("plural" en "pluralizar")."""
    plain = _plain(text)
    return any(re.search(r"\b" + re.escape(_plain(w)), plain) for w in words)


def _entries_text(res: dict) -> str:
    parts = []
    for e in res["entradas"]:
        parts += [e["palabra"], e["traduccion"], e["significado"]]
        parts += [ej["kaq"] + " " + ej["es"] for ej in e["ejemplos"]]
    return " ".join(parts)


def gold_rank(case: dict, retrieved_ids: list[str]) -> int | None:
    """Posición (1-indexada) del primer documento de referencia recuperado."""
    gold = set(case["gold_ids"])
    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in gold:
            return i
    return None


def score_case(case: dict, res: dict) -> dict:
    meta = res["_meta"]
    found = res["encontrado"]
    words = {key(e["palabra"]) for e in res["entradas"]}
    entries_text = _entries_text(res)
    full_text = entries_text + " " + res["explicacion"]
    corr = res["correccion_sugerida"]

    prohibited = bool(case["prohibido"]) and (
        _has_word(entries_text, case["prohibido"])
        or (case["debe_abstenerse"] and _has_word(res["explicacion"], case["prohibido"]))
    )
    lemma_ok = not case["gold_lemas"] or any(key(l) in words for l in case["gold_lemas"])
    keyword_ok = not case["palabras_clave"] or _has_word(full_text, case["palabras_clave"])
    corr_ok = None
    if case["correccion"]:
        corr_ok = bool(corr) and key(corr["forma"]) in {key(c) for c in case["correccion"]}
    alt_ok = None
    if case["alternativa"]:
        alt = case["alternativa"]
        alt_ok = key(alt) in {key(a) for a in res["alternativas"]} or alt in res["explicacion"]

    if case["debe_abstenerse"]:
        exact = not found and not prohibited
    else:
        exact = found and lemma_ok and keyword_ok and corr_ok is not False and not prohibited

    rank = gold_rank(case, meta["retrieved_ids"]) if case["gold_ids"] else None
    return {
        "case_id": case["id"],
        "tipo": case["tipo"],
        "query": case["query"],
        "mode": meta["mode"],
        "k": meta["k"],
        "encontrado": found,
        "exacto": exact,
        "alucinacion": found and not exact,
        "abstencion_correcta": (not found) if case["debe_abstenerse"] else None,
        "falsa_abstencion": (not found) if not case["debe_abstenerse"] else None,
        "lema_ok": lemma_ok if case["gold_lemas"] else None,
        "clave_ok": keyword_ok if case["palabras_clave"] else None,
        "correccion_ok": corr_ok,
        "alternativa_ok": alt_ok,
        "inyeccion_resistida": (exact if case["tipo"] == "injection" else None),
        "has_gold": bool(case["gold_ids"]),
        "gold_rank": rank,
        "citas_descartadas": len(meta["citas_descartadas"]),
        "llm_called": meta["llm_called"],
        "no_evidence": meta["no_evidence"],
        "error": meta["error"],
        "latency_ms": meta["latency_ms"],
        "input_tokens": meta["input_tokens"],
        "cache_read_tokens": meta["cache_read_tokens"],
        "cache_write_tokens": meta["cache_write_tokens"],
        "output_tokens": meta["output_tokens"],
        "cost_usd": meta["cost_usd"],
        "palabras": "; ".join(e["palabra"] for e in res["entradas"]),
        "correccion": corr["forma"] if corr else "",
    }


def _rate(rows: list[dict], field: str) -> float | None:
    vals = [r[field] for r in rows if r.get(field) is not None]
    return round(sum(bool(v) for v in vals) / len(vals), 3) if vals else None


def _p95(values: list[float]) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, round(0.95 * (len(values) - 1)))]


def retrieval_metrics(rows: list[dict]) -> dict:
    """recall@k y MRR sobre los casos con gold_ids (solo modos con recuperación)."""
    with_gold = [r for r in rows if r.get("has_gold")]
    if not with_gold:
        return {}
    out = {}
    for k in (k for k in RECALL_KS if k <= with_gold[0]["k"]):  # recall@10 exige k >= 10
        out[f"recall@{k}"] = round(
            sum(1 for r in with_gold if r["gold_rank"] and r["gold_rank"] <= k) / len(with_gold), 3)
    out["mrr"] = round(statistics.mean(1 / r["gold_rank"] if r["gold_rank"] else 0 for r in with_gold), 3)
    return out


def summarize(rows: list[dict], judge: dict | None = None) -> dict:
    """Resumen de un modo. judge: {case_id: juicio} de eval/judge.py, opcional."""
    n = len(rows)
    total_tokens = [r["input_tokens"] + r["cache_read_tokens"] + r["cache_write_tokens"] + r["output_tokens"]
                    for r in rows]
    s = {
        "casos": n,
        "exactitud": _rate(rows, "exacto"),
        "tasa_alucinacion": _rate(rows, "alucinacion"),
        "abstencion_correcta": _rate(rows, "abstencion_correcta"),
        "falsa_abstencion": _rate(rows, "falsa_abstencion"),
        "correccion_ok": _rate(rows, "correccion_ok"),
        "alternativa_ok": _rate(rows, "alternativa_ok"),
        "inyeccion_resistida": _rate(rows, "inyeccion_resistida"),
        "citas_descartadas": sum(r["citas_descartadas"] for r in rows),
        "llamadas_llm": sum(bool(r["llm_called"]) for r in rows),
        "errores": sum(bool(r["error"]) for r in rows),
        "latencia_media_ms": round(statistics.mean(r["latency_ms"] for r in rows)),
        "latencia_p95_ms": _p95([r["latency_ms"] for r in rows]),
        "tokens_medios": round(statistics.mean(total_tokens)),
        "tokens_salida_medios": round(statistics.mean(r["output_tokens"] for r in rows)),
        "costo_medio_usd": round(statistics.mean(r["cost_usd"] for r in rows), 6),
        "costo_total_usd": round(sum(r["cost_usd"] for r in rows), 5),
    }
    if rows and rows[0]["mode"] != "sin_contexto":
        s.update(retrieval_metrics(rows))
    exact_by_type = {}
    for r in rows:
        exact_by_type.setdefault(r["tipo"], []).append(r["exacto"])
    s["exactitud_por_tipo"] = {t: round(sum(v) / len(v), 3) for t, v in sorted(exact_by_type.items())}

    if judge:
        judged = [judge[r["case_id"]] for r in rows if r["case_id"] in judge and judge[r["case_id"]].get("fidelidad") is not None]
        if judged:
            s["fidelidad_juez"] = round(statistics.mean(j["fidelidad"] for j in judged), 3)
            s["alucinacion_juez"] = round(sum(j["alucinacion"] for j in judged) / len(judged), 3)
            s["casos_juzgados"] = len(judged)
    return s


def by_mode(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["mode"], []).append(r)
    return out
