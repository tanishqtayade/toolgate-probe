"""
ToolGate FastAPI Server
Run: uvicorn server:app --reload --port 8000
"""

import time
import json
from pathlib import Path
from contextlib import asynccontextmanager
from collections import deque

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from toolgate import ToolGateConfig, ToolProbe, ToolGate
from toolgate.data import load_json_dataset

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_PATH = "data/toy_when2tool.json"
TAU          = 0.5
MAX_TOKENS   = 80

# Cost estimates (USD per 1000 calls) — adjust to your actual API pricing
TOOL_CALL_COST   = 0.02   # e.g. calling a calculator / search API
LLM_DIRECT_COST  = 0.001  # answering from model memory

# ── In-memory stats store ─────────────────────────────────────────────────────
stats = {
    "total":        0,
    "tool_calls":   0,
    "direct":       0,
    "cost_with":    0.0,   # cost WITH ToolGate
    "cost_without": 0.0,   # cost WITHOUT (always call tool)
    "history":      deque(maxlen=50),   # last 50 queries
}

gate: ToolGate = None   # loaded at startup

# ── Startup / shutdown ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global gate
    print("Loading ToolGate...")
    cfg   = ToolGateConfig(model_name=MODEL_NAME, tau=TAU)
    probe = ToolProbe(cfg)
    prompts, labels = load_json_dataset(DATASET_PATH)
    probe.train(prompts, labels)
    probe.save()
    gate = ToolGate(probe)
    print("ToolGate ready.")
    yield
    print("Shutting down.")

app = FastAPI(title="ToolGate API", lifespan=lifespan)

# ── Request / response schemas ────────────────────────────────────────────────
class QueryRequest(BaseModel):
    prompt: str
    max_new_tokens: int = MAX_TOKENS

class QueryResponse(BaseModel):
    prompt:      str
    tool_needed: bool
    prob:        float
    response:    str
    latency_ms:  float

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    t0 = time.perf_counter()
    result = gate.generate(req.prompt, max_new_tokens=req.max_new_tokens)
    latency = (time.perf_counter() - t0) * 1000

    # Update stats
    stats["total"] += 1
    stats["cost_without"] += TOOL_CALL_COST   # baseline: always call tool

    if result["tool_needed"]:
        stats["tool_calls"] += 1
        stats["cost_with"]  += TOOL_CALL_COST
    else:
        stats["direct"]    += 1
        stats["cost_with"] += LLM_DIRECT_COST

    entry = {
        "prompt":      req.prompt,
        "tool_needed": result["tool_needed"],
        "prob":        round(result["prob"], 3),
        "response":    result["response"][:200],
        "latency_ms":  round(latency, 1),
    }
    stats["history"].appendleft(entry)

    return QueryResponse(**entry)


@app.get("/stats")
def get_stats():
    saved = stats["cost_without"] - stats["cost_with"]
    pct   = (saved / stats["cost_without"] * 100) if stats["cost_without"] > 0 else 0
    return {
        "total":           stats["total"],
        "tool_calls":      stats["tool_calls"],
        "direct_answers":  stats["direct"],
        "cost_with_gate":  round(stats["cost_with"],    4),
        "cost_without":    round(stats["cost_without"], 4),
        "saved_usd":       round(saved, 4),
        "saved_pct":       round(pct,   1),
        "history":         list(stats["history"]),
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "tau": TAU}


@app.get("/", response_class=HTMLResponse)
def dashboard():
    html = Path("templates/dashboard.html").read_text()
    return HTMLResponse(html)