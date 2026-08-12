"""HTTP surface: translates requests into engine calls and back."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from app import metrics
from app.config import load_config
from app.dataset import load_dataset
from app.engine import Engine, QueueFullError


class GenerateRequest(BaseModel):
    prompt_id: int


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# Simulator-specific: a real inference server would not report per-request
# timings on the response body.
class TimingMs(BaseModel):
    queue_wait: float
    prefill: float
    ttft: float
    decode: float
    e2e: float


class GenerateResponse(BaseModel):
    prompt_id: int
    content: str
    usage: Usage
    timing_ms: TimingMs


class Model(BaseModel):
    id: str
    created: int
    object: str = "model"
    owned_by: str = "vllm-sim"
    gpu: str  # not an OpenAI field; SDK clients ignore extra keys


class ModelList(BaseModel):
    object: str = "list"
    data: list[Model]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    config = load_config()
    app.state.config = config
    app.state.dataset = load_dataset(config.dataset_path, config.chars_per_token)
    app.state.engine = Engine(config)
    app.state.created = int(time.time())
    metrics.bind_engine(app.state.engine, config)
    yield


app = FastAPI(title="vllm-sim", lifespan=lifespan)


# Rounded for reading only. The engine keeps seconds as unrounded floats, since
# that is what Prometheus base units and vLLM's *_seconds histograms want.
def _ms(seconds: float) -> float:
    return round(seconds * 1000, 2)


# async, not sync: a sync handler runs in a threadpool, off the event loop the
# simulation's contention depends on.
@app.post("/generate")
async def generate(request: GenerateRequest) -> GenerateResponse:
    record = app.state.dataset.get(request.prompt_id)
    if record is None:
        metrics.record_not_found()
        raise HTTPException(404, f"unknown prompt_id: {request.prompt_id}")
    try:
        result = await app.state.engine.generate(record)
    except QueueFullError as exc:
        metrics.record_rejected()
        raise HTTPException(503, str(exc)) from exc
    metrics.record_success(result)

    t = result.timings
    return GenerateResponse(
        prompt_id=record.id,
        content=result.content,
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
        timing_ms=TimingMs(queue_wait=_ms(t.queue_wait), prefill=_ms(t.prefill_time),
                           ttft=_ms(t.ttft), decode=_ms(t.decode_time), e2e=_ms(t.e2e)),
    )


# Liveness/readiness probe: status code is the whole payload.
@app.get("/health")
async def health() -> Response:
    return Response(status_code=200)


@app.get("/metrics")
async def scrape() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/models")
async def models() -> ModelList:
    cfg = app.state.config
    return ModelList(data=[Model(id=cfg.model_name, created=app.state.created,
                                 gpu=cfg.gpu)])
