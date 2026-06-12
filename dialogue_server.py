#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VerbalValue dialogue service (open-source reference implementation).

Implements the interactive response channel and the idle pitch channel
described in the paper's dual-channel architecture (the dual-channel architecture section):

  POST /chat       - interactive channel. Given a viewer comment, matches
                      it against the product knowledge base, generates and
                      reranks candidate responses, and returns the
                      structured four-field reply.
  GET  /idle_next   - idle channel. Returns the next segment of the idle
                      pitch-script corpus for continuous product narration
                      during viewer inactivity.

This server holds the model and product/script libraries in memory and is
intended to run alongside the media service (TTS + asset hosting) and the
browser frontend, which together implement the shared-audio-resource
arbitration between the two channels.

All numeric timing and history-length parameters are read from environment
variables with no hardcoded defaults, so this file discloses no tuned
configuration values.
"""

import os
import time
import threading
from collections import deque
from typing import List, Tuple, Dict, Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import infer_dialogue as core


# ---------------------------------------------------------------------------
# Configuration (all from environment, no hardcoded numeric defaults)
# ---------------------------------------------------------------------------

def _env_float(name: str) -> Optional[float]:
    v = os.environ.get(name)
    return float(v) if v is not None else None


def _env_int(name: str) -> Optional[int]:
    v = os.environ.get(name)
    return int(v) if v is not None else None


IDLE_MODE = os.environ.get("IDLE_MODE", "true").strip().lower() not in ("0", "false", "")
IDLE_AFTER = _env_float("IDLE_AFTER")            # seconds of inactivity before idle narration resumes
IDLE_INTERVAL = _env_float("IDLE_INTERVAL")      # seconds between idle narration segments
HISTORY_MAX_TURNS = _env_int("HISTORY_MAX_TURNS")  # max dialogue turns retained server-side

PRODUCT_LIBRARY_PATH = os.environ.get("PRODUCT_LIBRARY_PATH") or core.PRODUCT_LIBRARY_PATH
SCRIPTS_JSON = os.environ.get("SCRIPT_LIBRARY_PATH") or core.SCRIPT_LIBRARY_PATH


# ---------------------------------------------------------------------------
# Model and resource loading
# ---------------------------------------------------------------------------

model, tokenizer = core.load_model()

prod = core.load_product_library(PRODUCT_LIBRARY_PATH)
products = prod.get("products", []) or []
print(f">>> Loaded product library: {len(products)} products from {PRODUCT_LIBRARY_PATH}")

script_items = core.load_idle_scripts(SCRIPTS_JSON)
idle_player = core.IdleScriptPlayer(script_items) if IDLE_MODE else None
print(f">>> Loaded idle scripts: {len(script_items)} items from {SCRIPTS_JSON}")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

recent_spoken: List[str] = []
recent_sigs: List[str] = []
history: List[Tuple[str, str]] = []

last_danmu_ts = time.time()
next_idle_ts = last_danmu_ts + (IDLE_AFTER or 0)

cycle_count = 0

_REPEAT_BUFFER = deque(maxlen=core.CONFIG["diversity"]["recent_window"] or core.CONFIG["service"]["repeat_buffer_size"])


def _update_repeat_buffer(text: str):
    """Record recently produced spoken text for bookkeeping. Never raises."""
    try:
        t = (text or "").strip()
        if t:
            _REPEAT_BUFFER.append(t)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="VerbalValue Dialogue Service (idle + interactive)")

_CHAT_LOCK = threading.Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    danmu: str  # viewer comment


class ChatResponse(BaseModel):
    danmu: str
    spoken: str
    speak_lines: List[str]
    product_id: str = ""


class IdleResponse(BaseModel):
    product_id: str = ""
    content: str = ""
    ready: bool = False


# ---------------------------------------------------------------------------
# Interactive response channel
# ---------------------------------------------------------------------------

def _select_best_reply(danmu: str, triples, matched_product: Optional[Dict[str, Any]]) -> ChatResponse:
    candidates = []
    for raw, json_text, data in triples:
        if not data:
            continue
        spoken = core.build_spoken_reply(data)
        if not spoken:
            continue
        candidates.append({"data": data, "spoken": spoken, "raw": raw, "json_text": json_text})

    prev_user = history[-1][0] if history else ""
    all_product_names = [p.get("name", "") for p in products if isinstance(p, dict)]
    best = core.pick_best_candidate(
        candidates, danmu, prev_user,
        recent_spoken, recent_sigs,
        matched_product=matched_product,
        all_product_names=all_product_names,
    )
    spoken = best["spoken"] if best else ""

    speak_lines = [x.strip() for x in spoken.split("\u3002") if x.strip()] if spoken else []
    pid = (matched_product.get("product_id", "") if isinstance(matched_product, dict) else "")

    return ChatResponse(danmu=danmu, spoken=spoken, speak_lines=speak_lines, product_id=pid)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    global last_danmu_ts, next_idle_ts, history, recent_spoken, recent_sigs

    danmu = (req.danmu or "").strip()
    if not danmu:
        return ChatResponse(danmu="", spoken="", speak_lines=[], product_id="")

    last_danmu_ts = time.time()
    next_idle_ts = last_danmu_ts + (IDLE_AFTER or 0)

    with _CHAT_LOCK:
        try:
            matched_product = core.match_product(danmu, products)
            prompt = core.build_prompt(history, danmu, matched_product=matched_product)
            triples = core.generate_n(model, tokenizer, prompt)

            resp = _select_best_reply(danmu, triples, matched_product)

        except Exception as e:
            import traceback
            print("[/chat] ERROR:", repr(e))
            traceback.print_exc()

            fallback_spoken = (
                "Got it, I'm here. Are you more interested in cleansing, "
                "sun protection, or repair? Tell me and I'll go from there."
            )
            resp = ChatResponse(danmu=danmu, spoken=fallback_spoken, speak_lines=[fallback_spoken], product_id="")

    if resp.spoken:
        _update_repeat_buffer(resp.spoken)

    history.append((danmu, resp.spoken))
    if HISTORY_MAX_TURNS and len(history) > HISTORY_MAX_TURNS:
        history = history[-HISTORY_MAX_TURNS:]

    print("===== [HTTP] live-room dialogue =====")
    print(f"Viewer: {resp.danmu}")
    print(f"Host: {resp.spoken}" if resp.spoken else "Host: <empty response>")
    print(f"(product_id={resp.product_id})")

    return resp


# ---------------------------------------------------------------------------
# Idle pitch channel
# ---------------------------------------------------------------------------

@app.get("/idle_next", response_model=IdleResponse)
def idle_next(force: bool = False):
    global next_idle_ts, recent_spoken, recent_sigs, cycle_count

    if (not IDLE_MODE) or (idle_player is None):
        return IdleResponse(ready=False)

    now = time.time()

    if force:
        pid, text = idle_player.next_item()
        return {"ready": True, "product_id": pid, "content": text}

    if next_idle_ts and now < next_idle_ts:
        return IdleResponse(ready=False)

    pid, content = idle_player.next_item()

    if not content:
        try:
            idle_player.i = 0
        except Exception:
            pass
        pid, content = idle_player.next_item()

    if not content:
        next_idle_ts = now + (IDLE_INTERVAL or 0)
        return IdleResponse(ready=False)

    recent_spoken.append(content)
    recent_sigs.append(core.opening_signature(content))
    recent_window = core.CONFIG["diversity"]["recent_window"]
    sig_window = core.CONFIG["diversity"]["signature_window"]
    if recent_window and len(recent_spoken) > recent_window:
        recent_spoken = recent_spoken[-recent_window:]
    if sig_window and len(recent_sigs) > sig_window:
        recent_sigs = recent_sigs[-sig_window:]

    cycle_marker = core.CONFIG["service"]["idle_cycle_marker"]
    if pid and cycle_marker and pid.endswith(cycle_marker):
        cycle_count += 1
        print(f">>> [IDLE] cycle_count={cycle_count} (wrapped back to {pid})")

    next_idle_ts = now + (IDLE_INTERVAL or 0)
    return IdleResponse(product_id=pid, content=content, ready=True)


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = _env_int("PORT") or core.CONFIG["service"]["default_port"]
    uvicorn.run(app, host=host, port=port)
