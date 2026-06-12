# VerbalValue (open-source reference implementation)

This repository contains a sanitized, reference implementation of the
core architecture described in *VerbalValue: A Socially Intelligent
Virtual Host for Sales-Driven Live Commerce*. It demonstrates the
system's structure, prompting strategy, and reranking framework.

It is **not** the production deployment. All tuned scoring weights and
thresholds, the production product/training data, and deployment-specific
infrastructure paths have been removed or replaced with placeholders. See
"What is and isn't included" below.

## Architecture overview

The system follows the dual-channel architecture in Section 4.2 of the
paper:

- **`infer_dialogue.py`** -- core module. Builds prompts from a persona
  system prompt and a strict four-field JSON output schema
  (`speak_lines`, `caption`, `hook_question`, `cta`), matches viewer
  comments against a product knowledge base via keyword/category
  detection and coverage scoring (Section 4.1), generates multiple
  candidates, and reranks them with a configurable penalty/bonus scoring
  function (product alignment, repetition, compliance, topical
  relevance).

- **`dialogue_server.py`** -- FastAPI service exposing:
  - `POST /chat` -- interactive response channel (Q&A)
  - `GET /idle_next` -- idle pitch channel (broadcast narration)

  Both channels share a single conceptual audio resource, arbitrated by
  the client runtime (see below).

- **`avatar_server.py`** -- media service exposing `POST /speak`, which
  synthesises one clause/sentence of audio at a time via an ONNX TTS
  model, enabling sentence-level streaming playback.

- **`local_website/client_runtime.js`** -- client runtime implementing
  clause segmentation (`splitByPunctuation`), incremental streaming
  playback (`speakSegments`), and the idle/interactive channel
  arbitration logic (`pollIdleNext` / `sendComment`) that preempts idle
  narration on comment arrival and resumes it from the saved sentence
  boundary afterward.

- **`train_sft.sh`** -- LoRA fine-tuning script for the intent-conditioned
  dataset, using the hyperparameters reported in the paper (rank 8,
  alpha 32, 20 epochs, effective batch size 32, learning rate 1e-4,
  bfloat16, max sequence length 2048).

## Four-field output schema

Every generation produces exactly one JSON object:

```json
{
  "speak_lines": ["...", "..."],
  "caption": "...",
  "hook_question": "...",
  "cta": "..."
}
```

`speak_lines` is the spoken broadcast content (at most two short
sentences), `caption` is a short on-screen tagline, `hook_question` is a
follow-up question intended to draw the viewer into the next turn, and
`cta` is a light call-to-action (e.g. claim a coupon, check the product
card).

## Configuration

All numeric values used by generation, reranking, product matching, and
service defaults are loaded from `config.json` (not included) rather
than hardcoded, including reranking weights, slicing/truncation limits,
length thresholds, the internal product-ID pattern, and service defaults
such as the listen port and polling interval. See `config.example.json`
for the full schema. The decoding parameters reported in the paper as
public (temperature 0.9, top-p 0.92, repetition penalty 1.12, 6
candidates) and the LoRA hyperparameters (rank 8, alpha 32, 20 epochs,
effective batch size 32, learning rate 1e-4) may be used directly; all
other values are deployment-specific tuned configuration and are left as
placeholders.

```bash
cp config.example.json config.json
# edit config.json with your own values
export VERBALVALUE_CONFIG=$(pwd)/config.json
```

## Running

```bash
export BASE_MODEL=/path/to/Qwen2.5-32B-Instruct
export ADAPTER_DIR=/path/to/your/lora-checkpoint   # optional
export PRODUCT_LIBRARY_PATH=./data/product_library.example.json
export SCRIPT_LIBRARY_PATH=./data/livestream_scripts.example.json
export VERBALVALUE_CONFIG=./config.json

# interactive CLI demo
python infer_dialogue.py

# dialogue service (idle + interactive channels)
export IDLE_AFTER=3
export IDLE_INTERVAL=8
export HISTORY_MAX_TURNS=24
python dialogue_server.py

# media service (TTS + asset hosting)
export PIPER_BIN=./piper/piper
export PIPER_MODEL=./piper/model.onnx
python avatar_server.py
```

## What is and isn't included

**Included** (sanitized / illustrative):
- Core prompt construction, product matching, generation, and reranking
  logic, with all tuned numeric weights factored out to `config.json`
- FastAPI services for the dialogue and media layers
- Client-side streaming TTS and channel-arbitration logic
- One example product entry and one example idle script, matching the
  real schema but with placeholder content
- Training script with the public hyperparameters from the paper

**Not included** (proprietary / deployment-specific):
- `config.json` with tuned reranking weights and thresholds
- The production product knowledge base (12 skincare products) and
  ingredient glossary (23 ingredients)
- The intent-conditioned fine-tuning dataset (1,475 instances)
- LoRA checkpoint weights
- Full frontend UI (styling, layout, branding)
- Deployment infrastructure paths and model snapshot hashes

## Citation

If you use this code, please cite the VerbalValue paper.
