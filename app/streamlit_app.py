"""Interfaz web del asistente de consulta kaqchikel.

    streamlit run app/streamlit_app.py

- Búsqueda con ejemplos clicables, corrección "¿Quisiste decir...?" y
  alternativas clicables (ik -> ik').
- Barra lateral: historial de la sesión y modo técnico (configuración, comparación
  de las 3 configuraciones y métricas de la última consulta).
- Respaldo sin internet: si la API falla, se usa data/demo_cache.json
  (python scripts/build_demo_cache.py).
- Feedback 👍/👎 por respuesta en logs/feedback.jsonl (evaluación humana extra).
"""
import html
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.kaq import config  # noqa: E402

# Con el modelo de embeddings ya descargado, se carga sin consultar Hugging Face:
# así la app arranca aunque no haya internet en la feria
_HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
if (_HF_HOME / "hub" / ("models--" + config.EMBEDDING_MODEL.replace("/", "--"))).exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import streamlit as st  # noqa: E402

from src.kaq.pipeline import NOT_FOUND, answer, warmup  # noqa: E402
from src.kaq.retrieval import MODES, get_index  # noqa: E402

MODE_LABELS = {"hibrido": "RAG híbrido", "lexico": "Solo léxico", "sin_contexto": "Sin contexto"}
MODE_CAPTIONS = {
    "hibrido": "Léxico + embeddings. Es el modo del producto.",
    "lexico": "Solo coincidencias de texto (baseline).",
    "sin_contexto": "El LLM responde de memoria (baseline).",
}
SIDEBAR_MODES = ["hibrido", "lexico", "sin_contexto"]
HISTORY_SIZE = 10


def _h(text) -> str:
    """Escapa texto para HTML. El $ se escapa porque Streamlit lo lee como LaTeX."""
    return html.escape(str(text)).replace("$", "&#36;")


def _key(query: str) -> str:
    return " ".join(query.split()).lower()


# ---------- Datos y llamadas ----------

class _Uncached(Exception):
    """Evita que st.cache_data guarde respuestas con error de la API."""

    def __init__(self, result: dict) -> None:
        super().__init__(result["_meta"]["error"])
        self.result = result


@st.cache_resource(show_spinner="Cargando los diccionarios (solo la primera vez)...")
def load() -> None:
    warmup()


@st.cache_data(show_spinner=False, max_entries=1000)
def _cached_answer(query: str, mode: str) -> dict:
    res = answer(query, mode)
    if res["_meta"]["error"]:
        raise _Uncached(res)
    return res


@st.cache_data(show_spinner=False)
def _demo_cache() -> dict:
    if not config.DEMO_CACHE.exists():
        return {}
    return json.loads(config.DEMO_CACHE.read_text(encoding="utf-8"))


def consult(query: str, mode: str) -> dict:
    """Respuesta de answer(); si la API falla, la guardada en demo_cache.json."""
    try:
        return _cached_answer(query, mode)
    except _Uncached as e:
        saved = _demo_cache().get(mode, {}).get(_key(query))
        return {**saved, "_respaldo": True} if saved else e.result


def _save_feedback(fb_key: str, query: str, mode: str, res: dict) -> None:
    value = st.session_state.get(fb_key)
    if value is None:
        return
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "query": query,
        "mode": mode,
        "util": bool(value),
        "encontrado": res["encontrado"],
        "entradas": [e.get("id") for e in res["entradas"]],
        "request_id": res["_meta"].get("request_id"),
        "respaldo": res.get("_respaldo", False),
    }
    with open(config.FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------- Callbacks (corren antes del rerun) ----------

def _set_query(query: str) -> None:
    st.session_state.q = query
    st.session_state.run = True


def _pick_example() -> None:
    if st.session_state.example:
        _set_query(st.session_state.example)
    st.session_state.example = None  # para poder elegir el mismo ejemplo otra vez


def _rerun_current() -> None:
    """Al cambiar de configuración se vuelve a responder la última consulta."""
    if st.session_state.current:
        _set_query(st.session_state.current["query"])


def run_query(query: str) -> None:
    modes = list(MODES) if st.session_state.compare else [st.session_state.mode]
    msg = "Consultando las 3 configuraciones..." if len(modes) > 1 else "Buscando en los diccionarios..."
    with st.spinner(msg):
        results = {m: consult(query, m) for m in modes}
    st.session_state.current = {"query": query, "results": results}
    history = st.session_state.history
    if query in history:
        history.remove(query)
    history.append(query)
    del history[:-HISTORY_SIZE]


# ---------- Presentación ----------

def _card(entry: dict, compact: bool) -> str:
    fuente = entry.get("fuente")
    if fuente:
        src = f'<span class="kq-src">{_h(fuente["diccionario"])} · pág. {_h(fuente["pagina"])}</span>'
    else:
        src = '<span class="kq-src kq-sin-fuente">Sin fuente · memoria del modelo</span>'
    ejemplos = "".join(
        f'<div class="kq-ej-kaq">{_h(ej["kaq"])}</div><div class="kq-ej-es">{_h(ej["es"])}</div>'
        for ej in entry["ejemplos"][: 1 if compact else 2]
    )
    return (
        f'<div class="kq-card{"" if fuente else " kq-memoria"}">'
        f'<div class="kq-head"><span class="kq-word">{_h(entry["palabra"])}</span>'
        f'<span class="kq-cat">{_h(entry.get("categoria") or "")}</span>{src}</div>'
        f'<div class="kq-trad">{_h(entry["traduccion"])}</div>'
        f'<p class="kq-sig">{_h(entry["significado"])}</p>'
        + (f'<div class="kq-ej">{ejemplos}</div>' if ejemplos else "")
        + "</div>"
    )


def render(res: dict, mode: str, query: str, compact: bool = False) -> None:
    meta = res["_meta"]
    if res.get("_respaldo"):
        st.caption(":material/cloud_off: Sin conexión con el modelo: se muestra una respuesta guardada.")
    elif meta["error"]:
        st.error(res["explicacion"], icon=":material/error:")
        return

    if res["advertencia"]:
        st.warning(res["advertencia"], icon=":material/shield:")

    corr = res["correccion_sugerida"]
    if corr:
        st.markdown(
            f'<div class="kq-corr">¿Quisiste decir <b>{_h(corr["forma"])}</b>? '
            f'<small>({_h(corr["motivo"])})</small></div>',
            unsafe_allow_html=True,
        )

    if not res["encontrado"]:
        detalle = "" if res["explicacion"] == NOT_FOUND else f"<p>{_h(res['explicacion'])}</p>"
        st.markdown(
            '<div class="kq-vacio"><h4>No aparece en los diccionarios</h4>'
            f"{detalle}<ul>"
            "<li>Revisa la escritura: el saltillo (') y la diéresis (ä, ë, ï, ö, ü) cambian la palabra.</li>"
            "<li>Prueba con una palabra suelta en lugar de una frase.</li>"
            "<li>Prueba a buscarla en español.</li></ul></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f'<p class="kq-expl">{_h(res["explicacion"])}</p>', unsafe_allow_html=True)
        for entry in res["entradas"]:
            st.markdown(_card(entry, compact), unsafe_allow_html=True)

    if res["alternativas"]:
        with st.container(horizontal=True, vertical_alignment="center"):
            st.caption("¿Buscabas también?")
            for alt in res["alternativas"]:
                st.button(alt, key=f"alt_{mode}_{alt}", on_click=_set_query, args=(alt,))

    if not meta["error"]:
        fb_key = f"fb_{mode}_{query}"
        with st.container(horizontal=True, vertical_alignment="center"):
            st.caption("¿Te sirvió esta respuesta?")
            st.feedback("thumbs", key=fb_key, on_change=_save_feedback, args=(fb_key, query, mode, res))


def _retrieved_rows(meta: dict) -> list[dict]:
    by_id = get_index().by_id
    rows = []
    for r in meta["retrieved"]:
        doc = by_id.get(r["id"], {})
        rows.append({
            "Entrada": doc.get("lema") or r["id"],
            "Fuente": f'{doc.get("diccionario", "")} p.{doc.get("pagina", "")}',
            "Método": r["metodo"],
            "Puntaje": r["score"],
        })
    return rows


def sidebar_metrics(current: dict) -> None:
    results = current["results"]
    if len(results) > 1:
        st.caption(f"Última consulta: «{current['query']}»")
        st.dataframe(
            [
                {
                    "Configuración": MODE_LABELS[m],
                    "Encontrado": "sí" if r["encontrado"] else "no",
                    "Latencia (s)": round(r["_meta"]["latency_ms"] / 1000, 1),
                    "Tokens": r["_meta"]["input_tokens"] + r["_meta"]["cache_read_tokens"] + r["_meta"]["output_tokens"],
                    "Costo (USD)": f'{r["_meta"]["cost_usd"]:.5f}',
                }
                for m, r in results.items()
            ],
            hide_index=True,
        )
        return

    mode, res = next(iter(results.items()))
    meta = res["_meta"]
    st.caption(f"Última consulta: «{current['query']}» · {MODE_LABELS[mode]}"
               + (" · respuesta guardada" if res.get("_respaldo") else ""))
    c1, c2 = st.columns(2)
    c1.metric("Latencia", f'{meta["latency_ms"] / 1000:.1f} s')
    c2.metric("Costo (USD)", f'{meta["cost_usd"]:.5f}')
    if meta["llm_called"]:
        st.caption(
            f'Tokens: {meta["input_tokens"]} de entrada + {meta["cache_read_tokens"]} en caché, '
            f'{meta["output_tokens"]} de salida · {meta["model"]}'
        )
    elif meta["no_evidence"]:
        st.caption("Sin evidencia en el índice: se respondió sin llamar al LLM.")
    if meta["citas_descartadas"]:
        st.caption("Citas descartadas (no recuperadas): " + ", ".join(meta["citas_descartadas"]))
    if meta["retrieved"]:
        st.caption(f'Entradas recuperadas (top-{meta["k"]})')
        st.dataframe(_retrieved_rows(meta), hide_index=True)


# ---------- Página ----------

for name, default in (("q", ""), ("history", []), ("current", None), ("mode", "hibrido"), ("compare", False)):
    st.session_state.setdefault(name, default)

st.set_page_config(
    page_title="Consulta de kaqchikel",
    page_icon=":material/menu_book:",
    layout="wide" if st.session_state.compare else "centered",
)
st.markdown(f"<style>{(ROOT / 'app' / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

st.markdown(
    '<p class="kq-title">Consulta de kaqchikel</p>'
    '<p class="kq-subtitle">Busca una palabra o frase en kaqchikel o en español. '
    "Cada respuesta cita el diccionario y la página de donde sale.</p>"
    '<div class="kq-band"></div>',
    unsafe_allow_html=True,
)

load()

with st.form("buscar", border=False):
    c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
    c1.text_input(
        "Consulta", key="q", max_chars=200, label_visibility="collapsed",
        placeholder="Ej.: awän, ¿cómo se dice luna?, ik'",
    )
    submitted = c2.form_submit_button("Buscar", type="primary", width="stretch")

st.pills(
    "Prueba con", config.DEMO_QUERIES[: config.N_EXAMPLES], key="example", on_change=_pick_example,
)

if submitted or st.session_state.pop("run", False):
    query = " ".join(st.session_state.q.split())
    if query:
        run_query(query)
    else:
        st.session_state.current = None
        st.info("Escribe una palabra o frase para buscar.", icon=":material/edit:")

current = st.session_state.current
if current:
    results = current["results"]
    if len(results) > 1:
        cols = st.columns(len(results), gap="large")
        for col, (mode, res) in zip(cols, results.items()):
            meta = res["_meta"]
            with col:
                st.markdown(
                    f'<div class="kq-modo">{MODE_LABELS[mode]}</div>'
                    f'<div class="kq-modo-meta">{meta["latency_ms"] / 1000:.1f} s · '
                    f'USD {meta["cost_usd"]:.5f}</div>',
                    unsafe_allow_html=True,
                )
                render(res, mode, current["query"], compact=True)
    else:
        mode, res = next(iter(results.items()))
        render(res, mode, current["query"])
else:
    st.caption(
        "No importa si no escribes el saltillo (') o la diéresis (ä): el asistente "
        "te sugiere la forma correcta del diccionario."
    )

with st.sidebar:
    st.markdown("#### Historial")
    if not st.session_state.history:
        st.caption("Aquí aparecerán tus consultas.")
    for i, past in enumerate(reversed(st.session_state.history)):
        st.button(past, key=f"hist_{i}", on_click=_set_query, args=(past,), type="tertiary")

    st.divider()
    st.markdown("#### Modo técnico")
    st.radio(
        "Configuración", SIDEBAR_MODES, key="mode", format_func=MODE_LABELS.get,
        captions=[MODE_CAPTIONS[m] for m in SIDEBAR_MODES],
        disabled=st.session_state.compare, on_change=_rerun_current,
    )
    st.toggle(
        "Comparar las 3 configuraciones", key="compare", on_change=_rerun_current,
        help="Responde la misma consulta sin contexto, con recuperación léxica y con RAG híbrido.",
    )
    if current:
        sidebar_metrics(current)

st.markdown(
    '<div class="kq-pie">Las respuestas las redacta un modelo de lenguaje (Claude) a partir del '
    "<i>Diccionario Bilingüe Kaqchikel</i> de OKMA y del material <i>Kaqchikel L2</i> de Dideduc. "
    "La fuente y la página salen del diccionario, no del modelo. Ante cualquier duda, revisa la "
    "página citada o consulta a una persona hablante. Las consultas se guardan sin datos personales "
    "para evaluar el sistema.</div>",
    unsafe_allow_html=True,
)
