"""Guarda las respuestas de las consultas demo para usarlas sin internet en la feria.

    python scripts/build_demo_cache.py                  # config.DEMO_QUERIES x 3 modos
    python scripts/build_demo_cache.py --modes hibrido

Escribe data/demo_cache.json con la forma {modo: {consulta normalizada: respuesta}}.
La app la usa solo cuando la API falla. Las respuestas con error no se guardan.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.kaq import config  # noqa: E402
from src.kaq.pipeline import answer, warmup  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="sin_contexto,lexico,hibrido")
    args = ap.parse_args()

    cache = {}
    if config.DEMO_CACHE.exists():
        cache = json.loads(config.DEMO_CACHE.read_text(encoding="utf-8"))

    print("Cargando índice y modelo de embeddings...")
    warmup()
    total, errors = 0.0, 0
    for mode in args.modes.split(","):
        saved = cache.setdefault(mode, {})
        for query in config.DEMO_QUERIES:
            res = answer(query, mode)
            total += res["_meta"]["cost_usd"]
            if res["_meta"]["error"]:
                errors += 1
                print(f"  {mode:<12} ERROR {query}: {res['_meta']['error']}")
                continue
            saved[" ".join(query.split()).lower()] = res
            print(f"  {mode:<12} encontrado={res['encontrado']!s:<5} {query}")

    config.DEMO_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nGuardado en {config.DEMO_CACHE} ({errors} errores, costo ${total:.4f})")


if __name__ == "__main__":
    main()
