"""Corre el conjunto de evaluación y guarda una iteración.

    python eval/run_eval.py --tag base                                 # 3 modos, Haiku, prompt v1, k=5
    python eval/run_eval.py --tag k10 --modes hibrido --k 10
    python eval/run_eval.py --tag sonnet --modes hibrido --model claude-sonnet-5-5
    python eval/run_eval.py --tag umbral083 --modes hibrido --dense-min 0.83
    python eval/run_eval.py --tag ret --retrieval-only --k 10          # solo recuperación: sin LLM ni costo
    python eval/run_eval.py --tag prueba --limit 2                     # confirmar antes de correr el lote

Salidas en eval/results/<tag>/:
- responses.jsonl: respuesta completa por caso y modo (trae texto de los diccionarios: fuera de git)
- rows.csv: métricas por caso
- summary.json: configuración y métricas por modo
Al final se regenera eval/results/iteraciones.csv (la tabla de la sección 2.5) con todos los summary.json.
"""
import argparse
import csv
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import (  # noqa: E402
    RECALL_KS, RESULTS_DIR, by_mode, gold_rank, load_cases, retrieval_metrics, score_case, summarize,
)
from src.kaq import config  # noqa: E402
from src.kaq.pipeline import answer, warmup  # noqa: E402
from src.kaq.retrieval import key, search  # noqa: E402

TABLE = RESULTS_DIR / "iteraciones.csv"
TABLE_COLUMNS = [
    "tag", "fecha", "modo", "modelo", "prompt", "k", "dense_min", "short_circuit", "nota",
    "exactitud", "tasa_alucinacion", "fidelidad_juez", "alucinacion_juez", "abstencion_correcta",
    "falsa_abstencion", "correccion_ok", "alternativa_ok", "inyeccion_resistida",
    *[f"recall@{k}" for k in RECALL_KS], "mrr", "citas_descartadas", "errores",
    "latencia_media_ms", "latencia_p95_ms", "tokens_medios", "costo_medio_usd", "costo_total_usd",
]


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    columns = columns or list(rows[0])
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_iterations_table() -> None:
    """Una fila por (iteración, modo) a partir de todos los summary.json."""
    rows = []
    for path in sorted(RESULTS_DIR.glob("*/summary.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        if s.get("solo_recuperacion"):
            continue
        cfg = s["config"]
        for mode, m in s["modos"].items():
            rows.append({
                "tag": s["tag"], "fecha": s["fecha"], "modo": mode,
                "modelo": cfg["model"], "prompt": cfg["prompt"],
                "k": cfg["k"], "dense_min": cfg["dense_min"], "short_circuit": cfg["short_circuit"],
                "nota": cfg.get("nota", ""), **m,
            })
    if rows:
        rows.sort(key=lambda r: (r["fecha"], r["tag"]))
        write_csv(TABLE, rows, TABLE_COLUMNS)
        print(f"Tabla de iteraciones: {TABLE} ({len(rows)} filas)")


def print_summary(summary: dict) -> None:
    for mode, m in summary["modos"].items():
        recall = " ".join(f"{k}={m[k]}" for k in m if k.startswith("recall@") or k == "mrr")
        print(f"\n[{mode}] exactitud={m['exactitud']} alucinación={m['tasa_alucinacion']} "
              f"abstención={m['abstencion_correcta']} falsa_abst={m['falsa_abstencion']} "
              f"corrección={m['correccion_ok']} inyección={m['inyeccion_resistida']}")
        if recall:
            print(f"  {recall}")
        print(f"  latencia media {m['latencia_media_ms']} ms (p95 {m['latencia_p95_ms']}), "
              f"{m['tokens_medios']} tokens/solicitud, ${m['costo_medio_usd']:.6f}/solicitud, "
              f"total ${m['costo_total_usd']:.4f}, errores {m['errores']}")
        print("  por tipo: " + ", ".join(f"{t} {v}" for t, v in m["exactitud_por_tipo"].items()))


def run_full(cases: list[dict], modes: list[str], args, out_dir: Path) -> dict:
    rows = []
    with open(out_dir / "responses.jsonl", "w", encoding="utf-8") as f:
        for mode in modes:
            for i, case in enumerate(cases, 1):
                res = answer(case["query"], mode, k=args.k, model=args.model, prompt_version=args.prompt)
                row = score_case(case, res)
                rows.append(row)
                f.write(json.dumps({"case_id": case["id"], "mode": mode, "response": res}, ensure_ascii=False) + "\n")
                mark = "OK " if row["exacto"] else "-- "
                print(f"  {mode:<12} {i:>2}/{len(cases)} {mark}{case['id']} {case['query'][:50]}"
                      + (f"  ERROR {res['_meta']['error']}" if res["_meta"]["error"] else ""))
    write_csv(out_dir / "rows.csv", rows)
    return {mode: summarize(mode_rows) for mode, mode_rows in by_mode(rows).items()}


def run_retrieval_only(cases: list[dict], modes: list[str], args, out_dir: Path) -> dict:
    """recall@k, MRR, abstención y corrección determinista sin llamar al LLM."""
    rows = []
    for mode in modes:
        for case in cases:
            t0 = time.perf_counter()
            ret = search(case["query"], mode, args.k)
            ids = [h.doc["id"] for h in ret.hits]
            corr = ret.correction
            rows.append({
                "case_id": case["id"], "tipo": case["tipo"], "mode": mode, "k": args.k,
                "has_gold": bool(case["gold_ids"]),
                "gold_rank": gold_rank(case, ids) if case["gold_ids"] else None,
                "no_evidence": ret.no_evidence,
                "abstencion_correcta": ret.no_evidence if case["debe_abstenerse"] else None,
                "falsa_abstencion": ret.no_evidence if not case["debe_abstenerse"] else None,
                "correccion_ok": (bool(corr) and key(corr["forma"]) in {key(c) for c in case["correccion"]})
                if case["correccion"] else None,
                "max_dense": ret.max_dense, "max_bm25": ret.max_bm25,
                "retrieval_ms": round((time.perf_counter() - t0) * 1000),
                "recuperados": "; ".join(ids),
            })
    write_csv(out_dir / "rows.csv", rows)

    def rate(rs, field):
        vals = [r[field] for r in rs if r[field] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    return {
        mode: {
            **retrieval_metrics(mode_rows),
            "abstencion_correcta": rate(mode_rows, "abstencion_correcta"),
            "falsa_abstencion": rate(mode_rows, "falsa_abstencion"),
            "correccion_ok": rate(mode_rows, "correccion_ok"),
            "latencia_media_ms": round(statistics.mean(r["retrieval_ms"] for r in mode_rows)),
        }
        for mode, mode_rows in by_mode(rows).items()
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="nombre de la iteración (carpeta en eval/results)")
    ap.add_argument("--modes", default="sin_contexto,lexico,hibrido")
    ap.add_argument("--model", default=config.MODEL)
    ap.add_argument("--prompt", default=config.PROMPT_VERSION)
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--dense-min", type=float, default=config.DENSE_MIN_SCORE)
    ap.add_argument("--no-short-circuit", action="store_true",
                    help="el LLM recibe las entradas aunque no haya evidencia (mide su abstención)")
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--limit", type=int, help="solo los primeros N casos")
    ap.add_argument("--nota", default="", help="qué cambió respecto a la iteración anterior")
    args = ap.parse_args()

    config.DENSE_MIN_SCORE = args.dense_min
    config.SHORT_CIRCUIT_NO_EVIDENCE = not args.no_short_circuit
    cases = load_cases()[: args.limit]
    modes = args.modes.split(",")
    if args.retrieval_only:
        modes = [m for m in modes if m != "sin_contexto"]
    out_dir = RESULTS_DIR / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cargando índice... ({len(cases)} casos x {len(modes)} modos)")
    warmup()
    if args.retrieval_only:
        modos = run_retrieval_only(cases, modes, args, out_dir)
    else:
        modos = run_full(cases, modes, args, out_dir)

    summary = {
        "tag": args.tag,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "solo_recuperacion": args.retrieval_only,
        "config": {
            "model": None if args.retrieval_only else args.model, "prompt": args.prompt, "k": args.k,
            "dense_min": args.dense_min, "short_circuit": not args.no_short_circuit,
            "casos": len(cases), "nota": args.nota,
        },
        "modos": modos,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.retrieval_only:
        for mode, m in modos.items():
            print(f"[{mode}] {m}")
    else:
        print_summary(summary)
        write_iterations_table()
    print(f"\nResultados en {out_dir}")


if __name__ == "__main__":
    main()
