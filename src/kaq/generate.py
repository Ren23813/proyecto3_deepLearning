"""Llamada a Claude con salida estructurada, medición de uso, costo y latencia.

La respuesta se valida contra el esquema Pydantic LLMResponse (structured
outputs garantiza JSON válido). La fuente (diccionario y página) no la escribe
el LLM: la agrega pipeline.py a partir del id, así no puede inventar páginas.
"""
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from . import config


class Ejemplo(BaseModel):
    kaq: str
    es: str


class Correccion(BaseModel):
    forma: str
    motivo: str


class EntradaLLM(BaseModel):
    id: str | None
    palabra: str
    categoria: str | None
    traduccion: str
    significado: str
    ejemplos: list[Ejemplo]


class LLMResponse(BaseModel):
    encontrado: bool
    idioma_consulta: Literal["kaq", "es", "mixto"]
    correccion_sugerida: Correccion | None
    entradas: list[EntradaLLM]
    explicacion: str
    advertencia: str | None


class GenerationError(Exception):
    """Error de generación con un mensaje apto para mostrar al usuario."""

    def __init__(self, user_message: str, detail: str = "") -> None:
        super().__init__(detail or user_message)
        self.user_message = user_message
        self.detail = detail or user_message


@dataclass
class Generation:
    response: LLMResponse
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    cost_usd: float
    stop_reason: str
    request_id: str | None


@lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    load_dotenv(config.ROOT / ".env")
    # El SDK reintenta solo 408/409/429/5xx y errores de conexión con backoff
    return anthropic.Anthropic(max_retries=3, timeout=60.0)


def cost(model: str, input_tokens: int, output_tokens: int, cache_write: int = 0, cache_read: int = 0) -> float:
    p = config.PRICES[model]
    total = (input_tokens * p["input"] + output_tokens * p["output"]
             + cache_write * p["cache_write"] + cache_read * p["cache_read"])
    return round(total / 1_000_000, 8)


def generate(system: str, user: str, model: str = config.MODEL) -> Generation:
    t0 = time.perf_counter()
    try:
        msg = get_client().messages.parse(
            model=model,
            max_tokens=config.MAX_TOKENS,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={"effort": config.EFFORT},
            output_format=LLMResponse,
        )
    except anthropic.AuthenticationError as e:
        raise GenerationError("Falta la API key de Claude o no es válida.", str(e)) from e
    except anthropic.RateLimitError as e:
        raise GenerationError("Hay muchas consultas en este momento. Intenta de nuevo en un minuto.", str(e)) from e
    except anthropic.APIConnectionError as e:
        raise GenerationError("No se pudo conectar con el servicio. Revisa la conexión a internet.", str(e)) from e
    except anthropic.APIStatusError as e:
        raise GenerationError("El servicio de Claude respondió con un error. Intenta de nuevo.",
                              f"{e.status_code}: {e.message}") from e
    except ValidationError as e:  # JSON truncado o fuera del esquema
        raise GenerationError("La respuesta del modelo no tuvo el formato esperado. Intenta de nuevo.", str(e)) from e
    latency_ms = round((time.perf_counter() - t0) * 1000)

    if msg.stop_reason == "refusal":
        raise GenerationError("El modelo se negó a responder esta consulta.", "stop_reason=refusal")
    if msg.stop_reason == "max_tokens" or msg.parsed_output is None:
        raise GenerationError("La respuesta quedó incompleta. Intenta con una consulta más corta.",
                              f"stop_reason={msg.stop_reason}")

    u = msg.usage
    cache_write = u.cache_creation_input_tokens or 0
    cache_read = u.cache_read_input_tokens or 0
    return Generation(
        response=msg.parsed_output,
        model=model,
        latency_ms=latency_ms,
        input_tokens=u.input_tokens,
        output_tokens=u.output_tokens,
        cache_write_tokens=cache_write,
        cache_read_tokens=cache_read,
        cost_usd=cost(model, u.input_tokens, u.output_tokens, cache_write, cache_read),
        stop_reason=msg.stop_reason,
        request_id=msg._request_id,
    )
