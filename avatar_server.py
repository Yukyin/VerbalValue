#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VerbalValue media service (open-source reference implementation).

Implements the TTS synthesis and gesture-planning endpoint described as
part of the media service in the paper's architecture (Media Service, Sentence-level Streaming TTS). The frontend calls /speak once
per clause (see splitByPunctuation / speakSegments in the client runtime)
to achieve sentence-level streaming synthesis.

Also serves the static frontend and product image assets.
"""

import os
import base64
import io
import subprocess
import tempfile
from typing import List

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Paths (configurable via environment, no hardcoded values)
# ---------------------------------------------------------------------------

PIPER_BIN = os.environ.get("PIPER_BIN", "./piper/piper")
PIPER_MODEL = os.environ.get("PIPER_MODEL", "./piper/model.onnx")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.environ.get("AVATAR_WEB_DIR") or os.path.join(_THIS_DIR, "local_website")
PIC_DIR = os.environ.get("PRODUCT_PIC_DIR", "./data/pic")
DEFAULT_UI_PAGE = os.environ.get("DEFAULT_UI_PAGE", "avatar_client.html")


app = FastAPI(title="VerbalValue Media Service")

if os.path.isdir(WEB_DIR):
    app.mount("/ui", StaticFiles(directory=WEB_DIR, html=True), name="ui")

if os.path.isdir(PIC_DIR):
    app.mount("/data/pic", StaticFiles(directory=PIC_DIR), name="product_pics")


@app.get("/")
def root():
    return RedirectResponse(url=f"/ui/{DEFAULT_UI_PAGE}")


# Allow any origin to call /speak for local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SpeakRequest(BaseModel):
    text: str


class Gesture(BaseModel):
    type: str
    pos: float  # normalised position within the utterance


class SpeakResponse(BaseModel):
    audio_wav_base64: str
    sample_rate: int
    gestures: List[Gesture]


# ---------------------------------------------------------------------------
# Gesture planning
# ---------------------------------------------------------------------------
#
# Maps emphasis cues in the text to simple gesture events at illustrative
# positions within the utterance. The deployed system uses a tuned set of
# emphasis keywords and timing offsets; the values below are placeholders.

EMPHASIS_KEYWORDS = [
    "key point", "note", "so", "but", "in conclusion",
    "important", "recommend", "must",
]

GESTURE_OPEN_POS = None    # position for an opening "open hand" gesture
GESTURE_POINT_POS = None   # position for an emphasis "point" gesture
GESTURE_NOD_POS = None     # position for a closing "nod" gesture


def plan_gestures(text: str) -> List[Gesture]:
    gestures = []
    if GESTURE_OPEN_POS is not None:
        gestures.append(Gesture(type="OPEN_HAND", pos=GESTURE_OPEN_POS))
    if GESTURE_POINT_POS is not None and any(k in text.lower() for k in EMPHASIS_KEYWORDS):
        gestures.append(Gesture(type="POINT", pos=GESTURE_POINT_POS))
    if GESTURE_NOD_POS is not None:
        gestures.append(Gesture(type="NOD", pos=GESTURE_NOD_POS))
    for g in gestures:
        g.pos = max(0.0, min(1.0, g.pos))
    return gestures


# ---------------------------------------------------------------------------
# TTS synthesis
# ---------------------------------------------------------------------------

def tts_synthesize_wav(text: str) -> bytes:
    """Synthesize a single clause/sentence to WAV bytes using an ONNX TTS
    model via the configured TTS binary."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        subprocess.run(
            [PIPER_BIN, "--model", PIPER_MODEL, "--output_file", f.name],
            input=text.encode("utf-8"),
            check=True,
        )
        f.seek(0)
        return f.read()


def wav_sample_rate(wav_bytes: bytes) -> int:
    import wave
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.getframerate()


@app.post("/speak", response_model=SpeakResponse)
def speak(req: SpeakRequest):
    text = (req.text or "").strip()
    if not text:
        return SpeakResponse(audio_wav_base64="", sample_rate=0, gestures=[])

    wav = tts_synthesize_wav(text)
    sr = wav_sample_rate(wav)
    b64 = base64.b64encode(wav).decode("ascii")
    gestures = plan_gestures(text)

    return SpeakResponse(
        audio_wav_base64=b64,
        sample_rate=sr,
        gestures=gestures,
    )


@app.get("/health")
def health():
    return {"ok": True}
