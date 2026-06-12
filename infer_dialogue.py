#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VerbalValue core inference module (open-source reference implementation).

This module implements the intent-conditioned generation and reranking
pipeline described in the VerbalValue paper:
  - Prompt construction with a persona system prompt and a strict
    four-field JSON output schema (speak_lines, caption, hook_question, cta)
  - Lightweight keyword/category-based product-knowledge-base matching
  - Multi-candidate generation followed by a rule-based reranker that
    penalises product misalignment, unsanctioned product mentions, and
    repetition, and rewards topical relevance and completeness

All tuned scoring weights and thresholds are loaded from an external
config file (config.json) rather than hardcoded, so this file contains
no proprietary tuning values. See config.example.json for the expected
schema; fill in your own values to reproduce a deployment.
"""

import json
import difflib
import os
import re
import sys
import time
import select
import argparse
from typing import Optional, Dict, Any, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from swift import Swift


# ---------------------------------------------------------------------------
# Paths (set via environment variables; no real paths are hardcoded here)
# ---------------------------------------------------------------------------

BASE_MODEL = os.environ.get("BASE_MODEL", "")            # e.g. local snapshot dir for Qwen2.5-32B-Instruct
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "")          # LoRA checkpoint directory

PRODUCT_LIBRARY_PATH = os.environ.get("PRODUCT_LIBRARY_PATH", "./data/product_library.example.json")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPT_LIBRARY_PATH = os.environ.get("SCRIPT_LIBRARY_PATH") or "./data/livestream_scripts.example.json"
_FALLBACK_SCRIPTS = os.path.join(_THIS_DIR, "data", "livestream_scripts.example.json")
if SCRIPT_LIBRARY_PATH and (not os.path.exists(SCRIPT_LIBRARY_PATH)) and os.path.exists(_FALLBACK_SCRIPTS):
    SCRIPT_LIBRARY_PATH = _FALLBACK_SCRIPTS

_FALLBACK_LIB = os.path.join(_THIS_DIR, "data", "product_library.example.json")
if PRODUCT_LIBRARY_PATH and (not os.path.exists(PRODUCT_LIBRARY_PATH)) and os.path.exists(_FALLBACK_LIB):
    PRODUCT_LIBRARY_PATH = _FALLBACK_LIB

CONFIG_PATH = os.environ.get("VERBALVALUE_CONFIG", os.path.join(_THIS_DIR, "config.json"))


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
#
# All numeric hyperparameters used by the generation and reranking pipeline
# live in this config object. None of the tuned values are checked into
# this repository; see config.example.json for the required keys and
# placeholder values, and supply your own config.json at deploy time.

def _default_config() -> Dict[str, Any]:
    """Fallback structure if no config.json is present.

    Values are intentionally left as None / empty so that the pipeline
    fails loudly rather than silently running with undisclosed tuned
    constants. Replace config.json with your own tuned values.
    """
    return {
        "generation": {
            "max_new_tokens": None,
            "temperature": None,
            "top_p": None,
            "repetition_penalty": None,
            "no_repeat_ngram_size": None,
            "num_candidates": None,
        },
        "dialogue": {
            "history_turns": None,
        },
        "diversity": {
            "recent_window": None,
            "signature_length": None,
            "signature_window": None,
            "signature_repeat_penalty": None,
            "near_duplicate_threshold": None,
            "near_duplicate_penalty": None,
            "near_duplicate_scale": None,
        },
        "reranking": {
            "relevance_weight": None,
            "banned_phrase_penalty": None,
            "placeholder_id_penalty": None,
            "placeholder_code_penalty": None,
            "other_product_name_penalty": None,
            "unmatched_ingredient_mention_penalty": None,
            "unmatched_ingredient_mention_cap": None,
            "product_name_match_bonus": None,
            "product_name_miss_penalty": None,
            "product_type_match_bonus": None,
            "usage_terms_missing_penalty": None,
            "short_response_threshold": None,
            "short_response_penalty": None,
        },
        "relevance": {
            "keyword_coverage_scale": None,
            "keyword_score_max": None,
            "tag_score_no_tags_default": None,
            "keyword_weight": None,
            "tag_weight": None,
            "no_hit_with_tags_cap": None,
            "score_min": None,
            "score_max": None,
        },
        "completeness": {
            "two_plus_lines_bonus": None,
            "one_line_bonus": None,
            "ideal_length_min": None,
            "ideal_length_max": None,
            "ideal_length_bonus": None,
            "too_short_length": None,
            "too_short_penalty": None,
        },
        "product_matching": {
            "min_score": None,
            "category_weight": None,
            "coverage_weight": None,
            "name_keyword_bonus": None,
            "name_keyword_topn": None,
        },
        "text": {
            "keyword_min_length": None,
            "short_comment_length": None,
            "context_skin_types_topn": None,
            "context_concerns_topn": None,
            "context_ingredients_topn": None,
            "context_talking_points_topn": None,
            "speak_lines_full_bonus_count": None,
            "speak_lines_partial_bonus_count": None,
        },
        "format": {
            "internal_id_pattern": None,
        },
        "service": {
            "idle_poll_seconds": None,
            "repeat_buffer_size": None,
            "default_port": None,
            "idle_cycle_marker": None,
            "error_exit_code": None,
        },
        "banned_phrases": [],
    }


def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    cfg = _default_config()
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            for section, vals in user_cfg.items():
                if isinstance(vals, dict) and section in cfg and isinstance(cfg[section], dict):
                    cfg[section].update(vals)
                else:
                    cfg[section] = vals
        except Exception as e:
            print(f"[WARN] Failed to load config from {path}: {e}")
    return cfg


CONFIG = load_config()


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
#
# These templates define the persona and the four-field structured output
# schema described in the paper (spoken broadcast line, display slogan,
# hook question, call-to-action). The wording below is illustrative; in
# the deployed system the persona prompt is tuned further.

SYSTEM_STYLE = (
    "You are a socially intelligent live-commerce host. "
    "Always address the viewer's comment directly before introducing any "
    "product information; do not change the subject.\n"
    "Style: speak the way a real host would on stream, casual and specific, "
    "with a light personal touch, but avoid repeating fixed catchphrases.\n"
    "Length: a couple of short sentences, concrete but not overloaded with claims.\n"
    "Compliance: do not exaggerate or make medical claims; do not output "
    "the names of real people (influencers, doctors, customers, etc.).\n"
    "Product library rule: if you reference a product from the knowledge "
    "base, you must use its real product name. Never output an internal "
    "product ID or a 'category + internal code' placeholder, and never "
    "mention 'the product library' or 'product codes' to the viewer.\n"
    "If no product in the library matches this turn, do not invent "
    "ingredient names or mechanism-level claims; give general selection "
    "and usage advice only, and avoid contradicting established facts.\n\n"
)

JSON_SPEC = (
    "You must output exactly one JSON object and nothing else (no "
    "explanations, no extra text). The JSON schema is strictly:\n"
    "{\n"
    "  \"speak_lines\": [\"...\", \"...\"],\n"
    "  \"caption\": \"may be empty\",\n"
    "  \"hook_question\": \"may be empty\",\n"
    "  \"cta\": \"may be empty\"\n"
    "}\n\n"
    "speak_lines must be coherent and on-topic (it may be split into "
    "multiple short sentences as an array); do not output unrelated "
    "content.\n\n"
)

PROMPT_SUFFIX = "\n[JSON]"


# Topic tag patterns used for lightweight relevance scoring between the
# viewer comment and the candidate response. The patterns below are a
# small illustrative subset; the deployed system uses a larger, tuned
# pattern set specific to the product vertical.
TAG_PATTERNS = [
    ("price_concern", [r"expensive", r"price", r"cost", r"discount"]),
    ("efficacy_timeline", [r"how long", r"when.*work", r"results"]),
    ("product_request", [r"recommend", r"which", r"what.*use", r"suggest"]),
    ("skin_type", [r"dry", r"oily", r"sensitive", r"combination"]),
]


# ---------------------------------------------------------------------------
# Basic text utilities
# ---------------------------------------------------------------------------

def die(msg: str, code: Optional[int] = None):
    print(f"[ERROR] {msg}")
    raise SystemExit(code if code is not None else CONFIG["service"].get("error_exit_code"))


def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = " ".join(s.replace("\u3000", " ").split())
    return s


def ensure_local_path(path_or_repo: str) -> str:
    if os.path.isabs(path_or_repo) and (not os.path.isdir(path_or_repo)):
        die(
            f"BASE_MODEL path does not exist: {path_or_repo}\n"
            f"Set the BASE_MODEL environment variable to a local snapshot "
            f"directory for the base model (e.g. Qwen2.5-32B-Instruct)."
        )
    return path_or_repo


def extract_first_json(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1].strip()
    return None


def safe_parse_json(json_text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(json_text)
    except Exception:
        return None


def build_spoken_reply(data: Dict[str, Any]) -> str:
    speak_lines = data.get("speak_lines") or []
    if not isinstance(speak_lines, list):
        return ""
    lines = [normalize_text(x) for x in speak_lines if normalize_text(x)]
    spoken = normalize_text(" ".join(lines))
    return spoken


# ---------------------------------------------------------------------------
# Reranking components
# ---------------------------------------------------------------------------
#
# Each function below returns a penalty (higher = worse) or a bonus
# (negative penalty), combined in pick_best_candidate(). All numeric
# weights and thresholds come from CONFIG so the exact tuned reranker
# behaviour is not disclosed in this reference implementation.

def completeness_score(data: Dict[str, Any]) -> float:
    """Reward responses that populate the structured schema and fall
    within a reasonable spoken-line length."""
    c = CONFIG["completeness"]
    s = 0.0
    speak_lines = data.get("speak_lines")
    full_n = CONFIG["text"]["speak_lines_full_bonus_count"]
    partial_n = CONFIG["text"]["speak_lines_partial_bonus_count"]
    if isinstance(speak_lines, list):
        nonempty = [x for x in speak_lines if normalize_text(x)]
        if full_n is not None and len(nonempty) >= full_n:
            s += c["two_plus_lines_bonus"] or 0
        elif partial_n is not None and len(nonempty) == partial_n:
            s += c["one_line_bonus"] or 0
    spoken = build_spoken_reply(data)
    lo, hi = c["ideal_length_min"], c["ideal_length_max"]
    if lo is not None and hi is not None and lo <= len(spoken) <= hi:
        s += c["ideal_length_bonus"] or 0
    too_short = c["too_short_length"]
    if too_short is not None and len(spoken) < too_short:
        s -= c["too_short_penalty"] or 0
    return s


def similarity(a: str, b: str) -> float:
    a = normalize_text(a)
    b = normalize_text(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def diversity_penalty(candidate_spoken: str, recent_spoken: List[str]) -> float:
    """Penalise near-duplicates of recently produced responses."""
    if not recent_spoken:
        return 0.0
    sims = [similarity(candidate_spoken, x) for x in recent_spoken if x]
    if not sims:
        return 0.0
    max_sim = max(sims)
    d = CONFIG["diversity"]
    threshold = d["near_duplicate_threshold"]
    if threshold is not None and max_sim > threshold:
        return d["near_duplicate_penalty"] or 0
    scale = d["near_duplicate_scale"] or 0
    return (max_sim ** 2) * scale


def opening_signature(spoken: str) -> str:
    """Short signature of a response's opening, used to detect repeated
    openers across consecutive turns."""
    s = normalize_text(spoken)
    if not s:
        return ""
    sig_len = CONFIG["diversity"]["signature_length"] or len(s)
    for sep in ["\u3002", "\uff01", "\uff1f", ".", "!", "?", "\uff0c", ","]:
        if sep in s:
            s = s.split(sep, 1)[0]
            break
    return s[:sig_len]


def signature_penalty(sig: str, recent_sigs: List[str]) -> float:
    if not sig or not recent_sigs:
        return 0.0
    cnt = sum(1 for x in recent_sigs if x == sig)
    return cnt * (CONFIG["diversity"]["signature_repeat_penalty"] or 0)


def banned_phrase_penalty(text: str) -> float:
    """Penalise reuse of phrases on a configurable denylist (e.g. stock
    phrases the model has overfit to)."""
    t = normalize_text(text)
    p = 0.0
    penalty = CONFIG["reranking"]["banned_phrase_penalty"] or 0
    for ph in CONFIG.get("banned_phrases", []):
        if ph in t:
            p += penalty
    return p


def placeholder_penalty(spoken: str) -> float:
    """Penalise leakage of internal product identifiers into spoken
    output (e.g. raw product IDs or 'category + internal code' phrasing)."""
    s = normalize_text(spoken)
    r = CONFIG["reranking"]
    p = 0.0
    id_pattern = CONFIG["format"]["internal_id_pattern"]
    if id_pattern and re.search(id_pattern, s):
        p += r["placeholder_id_penalty"] or 0
    if re.search(r"internal\s*code|catalog\s*id", s, flags=re.IGNORECASE):
        p += r["placeholder_code_penalty"] or 0
    return p


def ingredient_mention_penalty_when_unmatched(spoken: str) -> float:
    """When no product was matched, penalise the model for inventing
    specific ingredient or mechanism-level claims rather than giving
    general advice."""
    s = normalize_text(spoken)
    if not s:
        return 0.0
    # Illustrative subset of ingredient/mechanism terms; the deployed
    # system uses a larger glossary-derived pattern list.
    pats = [
        r"niacinamide", r"salicylic acid", r"\bBHA\b", r"\bAHA\b", r"\bPHA\b",
        r"retinol", r"vitamin c", r"ceramide", r"hyaluronic acid",
        r"ingredient|formula|mechanism|inhibit|promote",
    ]
    hit = sum(1 for pat in pats if re.search(pat, s, flags=re.IGNORECASE))
    r = CONFIG["reranking"]
    cap = r["unmatched_ingredient_mention_cap"]
    per_hit = r["unmatched_ingredient_mention_penalty"] or 0
    val = hit * per_hit
    return min(cap, val) if cap is not None else val


def other_product_name_penalty(
    spoken: str,
    matched_product: Optional[Dict[str, Any]],
    all_product_names: List[str],
) -> float:
    """Penalise mentioning a different catalogue product than the one
    that was matched for this turn."""
    if not matched_product:
        return 0.0
    s = normalize_text(spoken)
    if not s:
        return 0.0
    keep = normalize_text(str(matched_product.get("name", "")))
    p = 0.0
    penalty = CONFIG["reranking"]["other_product_name_penalty"] or 0
    for n in all_product_names:
        nn = normalize_text(n)
        if not nn or (keep and nn == keep):
            continue
        if nn in s:
            p += penalty
    return p


# ---------------------------------------------------------------------------
# Keyword / topic utilities
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "and", "or", "but",
    "so", "to", "of", "in", "on", "for", "with", "this", "that", "it",
}


def _keywords(text: str) -> List[str]:
    t = normalize_text(text).lower()
    toks = re.findall(r"[a-z0-9]+", t)
    min_len = CONFIG["text"]["keyword_min_length"] or 0
    out = [x for x in toks if x not in _STOP_WORDS and len(x) >= min_len]
    seen = set()
    merged = []
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        merged.append(x)
    return merged


def topic_tags(text: str) -> List[str]:
    t = normalize_text(text).lower()
    tags = []
    for tag, pats in TAG_PATTERNS:
        for pat in pats:
            if re.search(pat, t, flags=re.IGNORECASE):
                tags.append(tag)
                break
    return tags


def relevance_score(danmu: str, spoken: str, prev_user: str = "") -> float:
    """Score how on-topic a candidate response is (config-defined range,
    higher is more on-topic), combining keyword coverage and topic-tag
    agreement. Short comments borrow context from the previous turn."""
    spoken_n = normalize_text(spoken)
    if not spoken_n:
        return 0.0

    rc = CONFIG["relevance"]
    score_min = rc["score_min"] or 0.0
    score_max = rc["score_max"] or 0.0

    base_q = danmu
    short_len = CONFIG["text"]["short_comment_length"]
    if short_len is not None and len(re.sub(r"\s+", "", danmu)) <= short_len and prev_user:
        base_q = prev_user + " " + danmu

    kws = _keywords(base_q)
    hit = sum(1 for k in kws if k in spoken_n.lower()) if kws else 0
    cov = hit / max(1, len(kws)) if kws else (rc["tag_score_no_tags_default"] or 0.0) / max(score_max, 1.0)
    coverage_scale = rc["keyword_coverage_scale"] or 0.0
    score_kw_max = rc["keyword_score_max"] or 0.0
    score_kw = score_kw_max * min(1.0, cov * coverage_scale)

    q_tags = topic_tags(base_q)
    a_tags = topic_tags(spoken_n)
    if q_tags:
        tag_hit = sum(1 for t in q_tags if t in a_tags)
        score_tag = score_max * (tag_hit / len(q_tags))
    else:
        score_tag = rc["tag_score_no_tags_default"] or 0.0

    kw_w = rc["keyword_weight"] or 0.0
    tag_w = rc["tag_weight"] or 0.0
    score = kw_w * score_kw + tag_w * score_tag

    no_hit_cap = rc["no_hit_with_tags_cap"]
    if hit == 0 and q_tags and no_hit_cap is not None:
        score = min(score, no_hit_cap)
    return max(score_min, min(score_max, score))


# ---------------------------------------------------------------------------
# Product knowledge base
# ---------------------------------------------------------------------------

# Category synonym patterns used for lightweight product-category matching.
# The deployed system uses a vertical-specific synonym table; the entries
# below are illustrative placeholders for the skincare vertical described
# in the paper.
CATEGORY_SYNONYMS = {
    "cleanser": [r"cleanser", r"face wash"],
    "sunscreen": [r"sunscreen", r"SPF", r"PA\+{2,}"],
    "moisturizer": [r"moisturi[sz]er", r"cream"],
    "serum": [r"serum", r"ampoule"],
}

PRODUCT_QUERY_HINTS = [
    r"recommend", r"which one", r"what should i use", r"suggest",
    r"is .* good", r"any .* for",
]


def _product_search_text(p: Dict[str, Any]) -> str:
    parts = []
    for k in ("product_id", "name", "type", "texture"):
        v = p.get(k)
        if v:
            parts.append(str(v))

    sf = p.get("suitable_for") or {}
    if isinstance(sf, dict):
        for kk in ("skin_types", "concerns", "age_groups"):
            vv = sf.get(kk)
            if isinstance(vv, list):
                parts.extend([str(x) for x in vv if x])

    kis = p.get("key_ingredients") or []
    if isinstance(kis, list):
        for it in kis:
            if isinstance(it, dict):
                for kk in ("name", "role", "notes"):
                    vv = it.get(kk)
                    if vv:
                        parts.append(str(vv))

    tps = p.get("live_talking_points") or []
    if isinstance(tps, list):
        parts.extend([str(x) for x in tps if x])

    return normalize_text(" ".join(parts))


def load_product_library(path: str) -> Dict[str, Any]:
    out = {"raw": None, "products": []}
    if not path or (not os.path.exists(path)):
        return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        products = raw.get("products", []) if isinstance(raw, dict) else []
        if not isinstance(products, list):
            products = []
        for p in products:
            if isinstance(p, dict):
                p["_search_text"] = _product_search_text(p)
        out["raw"] = raw
        out["products"] = [p for p in products if isinstance(p, dict)]
        return out
    except Exception:
        return out


def load_idle_scripts(path: str) -> List[Dict[str, str]]:
    if not path or (not os.path.exists(path)):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            return []

        def _pid_key(x):
            pid = str(x.get("product_id", ""))
            mm = re.search(r"(\d+)", pid)
            return int(mm.group(1)) if mm else 10 ** 9

        raw_sorted = sorted([x for x in raw if isinstance(x, dict)], key=_pid_key)
        items: List[Dict[str, str]] = []
        for it in raw_sorted:
            pid = normalize_text(str(it.get("product_id", "")))
            c = normalize_text(str(it.get("content", "")))
            if pid and c:
                items.append({"product_id": pid, "content": c})
        return items
    except Exception:
        return []


class IdleScriptPlayer:
    """Cycles through the idle pitch-script corpus for the broadcast
    (idle) channel described in the paper's dual-channel architecture."""

    def __init__(self, script_items: List[Dict[str, str]]):
        items = script_items or []
        cleaned: List[Dict[str, str]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            pid = normalize_text(str(it.get("product_id", "")))
            c = normalize_text(str(it.get("content", "")))
            if pid and c:
                cleaned.append({"product_id": pid, "content": c})
        self.items = cleaned
        self.i = 0

    def next_item(self) -> Tuple[str, str]:
        if not self.items:
            return ("", "")
        it = self.items[self.i % len(self.items)]
        self.i += 1
        return (it.get("product_id", ""), it.get("content", ""))


def _append_text_line(path: str, text: str) -> None:
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(text.strip() + "\n")
    except Exception:
        pass


def _detect_query_categories(text: str) -> List[str]:
    t = normalize_text(text)
    cats = []
    for cat, pats in CATEGORY_SYNONYMS.items():
        for pat in pats:
            if re.search(pat, t, flags=re.IGNORECASE):
                cats.append(cat)
                break
    return cats


def is_product_query(danmu: str) -> bool:
    t = normalize_text(danmu)
    if any(re.search(p, t, flags=re.IGNORECASE) for p in PRODUCT_QUERY_HINTS):
        return True
    if _detect_query_categories(t):
        return True
    return False


def match_product(danmu: str, products: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Match a viewer comment against the product knowledge base via
    category detection plus keyword coverage scoring, as described in
    the paper's intent-conditioned fine-tuning section."""
    if not products:
        return None

    q = normalize_text(danmu)
    if not is_product_query(q):
        return None

    kws = _keywords(q)
    q_cats = _detect_query_categories(q)
    pm = CONFIG["product_matching"]

    cand_products: List[Dict[str, Any]] = []
    if q_cats:
        for p in products:
            p_type = normalize_text(str(p.get("type", "")))
            p_name = normalize_text(str(p.get("name", "")))
            for c in q_cats:
                if c and (c in p_type or c in p_name):
                    cand_products.append(p)
                    break

        if not cand_products:
            for p in products:
                p_type = normalize_text(str(p.get("type", "")))
                p_name = normalize_text(str(p.get("name", "")))
                ok = False
                for c in q_cats:
                    pats = CATEGORY_SYNONYMS.get(c, [])
                    for pat in pats:
                        if re.search(pat, p_type, flags=re.IGNORECASE) or re.search(pat, p_name, flags=re.IGNORECASE):
                            ok = True
                            break
                    if ok:
                        break
                if ok:
                    cand_products.append(p)

        if cand_products:
            best = None
            best_score = -1.0
            name_bonus_step = pm["name_keyword_bonus"] or 0
            topn = pm["name_keyword_topn"]
            for p in cand_products:
                txt = p.get("_search_text", "") or ""
                hit = sum(1 for k in kws if k in txt) if kws else 0
                cov = hit / max(1, len(kws)) if kws else 0.0
                p_name = normalize_text(str(p.get("name", "")))
                name_bonus = 0.0
                for k in (kws[:topn] if topn else kws):
                    if k and k in p_name:
                        name_bonus += name_bonus_step
                score = cov + name_bonus
                if score > best_score:
                    best_score = score
                    best = p
            if best:
                best = dict(best)
                best["_match_score"] = float(best_score)
                best["_match_cats"] = q_cats
                return best

    best = None
    best_score = 0.0
    cat_w = pm["category_weight"] or 0
    cov_w = pm["coverage_weight"] or 0
    name_bonus_step = pm["name_keyword_bonus"] or 0
    topn = pm["name_keyword_topn"]
    for p in products:
        txt = p.get("_search_text", "")
        if not txt:
            continue
        hit = sum(1 for k in kws if k in txt) if kws else 0
        cov = hit / max(1, len(kws)) if kws else 0.0

        p_type = normalize_text(str(p.get("type", "")))
        p_name = normalize_text(str(p.get("name", "")))
        cat_hit = 0
        if q_cats:
            for c in q_cats:
                if c in p_type or c in p_name:
                    cat_hit += 1
        cat_score = (cat_hit / len(q_cats)) if q_cats else 0.0

        fuzzy_bonus = 0.0
        for k in (kws[:topn] if topn else kws):
            if k and (k in p_name):
                fuzzy_bonus += name_bonus_step

        score = cov_w * cov + cat_w * cat_score + fuzzy_bonus
        if score > best_score:
            best_score = score
            best = p

    min_score = pm["min_score"]
    if best is not None and min_score is not None and best_score >= min_score:
        best = dict(best)
        best["_match_score"] = best_score
        best["_match_cats"] = q_cats
        return best
    return None


def format_product_context(p: Dict[str, Any]) -> str:
    """Serialise a matched catalogue entry into the system prompt,
    grounding generation in catalogue-verified content as described in
    the paper."""
    name = p.get("name", "")
    ptype = p.get("type", "")
    texture = p.get("texture", "")
    t = CONFIG["text"]
    sf = p.get("suitable_for") or {}
    skin_n = t["context_skin_types_topn"]
    concern_n = t["context_concerns_topn"]
    skin_types = ", ".join((sf.get("skin_types") or [])[:skin_n] if skin_n else (sf.get("skin_types") or [])) if isinstance(sf, dict) else ""
    concerns = ", ".join((sf.get("concerns") or [])[:concern_n] if concern_n else (sf.get("concerns") or [])) if isinstance(sf, dict) else ""

    kis = p.get("key_ingredients") or []
    ing_n = t["context_ingredients_topn"]
    ing_lines = []
    if isinstance(kis, list):
        for it in (kis[:ing_n] if ing_n else kis):
            if isinstance(it, dict):
                ing_lines.append(f"- {it.get('name','')}: {it.get('role','')}")

    usage = p.get("usage") or {}
    how_to = usage.get("how_to", "") if isinstance(usage, dict) else ""
    freq = usage.get("frequency", "") if isinstance(usage, dict) else ""

    tps = p.get("live_talking_points") or []
    tps = [x for x in tps if isinstance(x, str)]
    tp_n = t["context_talking_points_topn"]
    tp_lines = "\n".join([f"- {x}" for x in (tps[:tp_n] if tp_n else tps)])

    return (
        f"Product: {name}\n"
        f"Category: {ptype}; Texture: {texture}\n"
        f"Suitable for: {skin_types}\n"
        f"Targets: {concerns}\n"
        f"Key ingredients:\n" + "\n".join(ing_lines) + "\n"
        f"Usage: {how_to}; Frequency: {freq}\n"
        f"Live talking points:\n{tp_lines}\n"
        f"Compliance disclaimer: {p.get('compliance_disclaimer','')}\n"
    )


def product_alignment_penalty(spoken: str, matched_product: Optional[Dict[str, Any]]) -> float:
    """If a product was matched for this turn, require that the response
    name it explicitly and reference its usage; penalise responses that
    omit this grounding."""
    if not matched_product:
        return 0.0
    s = normalize_text(spoken)
    pname = normalize_text(str(matched_product.get("name", "")))
    ptype = normalize_text(str(matched_product.get("type", "")))
    r = CONFIG["reranking"]
    pen = 0.0
    if pname:
        if pname in s:
            pen -= r["product_name_match_bonus"] or 0
        else:
            pen += r["product_name_miss_penalty"] or 0
    if ptype and ptype in s:
        pen -= r["product_type_match_bonus"] or 0
    if not re.search(r"suitable|usage|how to use|frequency|morning|evening|layer|pair with", s, flags=re.IGNORECASE):
        pen += r["usage_terms_missing_penalty"] or 0
    return pen


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(
    history: List[Tuple[str, str]],
    danmu: str,
    matched_product: Optional[Dict[str, Any]] = None,
) -> str:
    """Construct the full prompt for one generation step, combining the
    persona system prompt, the four-field JSON schema, optional matched
    product context, recent dialogue history, and the current viewer
    comment."""
    parts = [SYSTEM_STYLE, JSON_SPEC]

    if matched_product:
        parts.append("[PRODUCT MATCH]\n")
        parts.append(
            "A matching product was found in the knowledge base. You must:\n"
            "- Directly answer the viewer's comment first;\n"
            "- Refer to this product only by its real catalogue name "
            "(never output an internal product identifier or a "
            "'category plus internal code' placeholder, and do not "
            "mention 'the product library' or 'product codes');\n"
            "- Reference several live talking points plus a usage detail "
            "and a suitable skin type or concern, all consistent with "
            "this product;\n"
            "- You may add a first-person anecdote, but do not name any "
            "real person;\n"
            "- Keep the overall response short, conversational, and "
            "persuasive, without repeating fixed catchphrases;\n"
            "- Do not recommend or name any other product (if a "
            "comparison is needed, refer only to general directions such "
            "as 'a lighter or richer option in the same category').\n\n"
        )
        parts.append(format_product_context(matched_product))
        parts.append("\n")
    else:
        parts.append("[NO PRODUCT MATCH]\n")
        parts.append(
            "No specific catalogue entry matches this turn. Respond as a "
            "knowledgeable live-commerce host improvising:\n"
            "- Directly answer the viewer's comment first;\n"
            "- Give a few general selection or usage tips, described "
            "only in terms of texture, skin type, scenario, or routine "
            "steps;\n"
            "- You may offer a couple of general directions and a "
            "follow-up question to narrow it down;\n"
            "- You may add a first-person anecdote, but do not say "
            "'we don't have that' or name any real person;\n"
            "- Do not name any brand or specific product, and do not "
            "invent ingredient names, concentrations, mechanisms, prices, "
            "or medical claims;\n"
            "- Keep the overall response short, conversational, and "
            "persuasive.\n\n"
        )

    if history:
        parts.append("[DIALOGUE HISTORY]\n")
        history_turns = CONFIG["dialogue"]["history_turns"]
        recent = history[-history_turns:] if history_turns else history
        for u, a in recent:
            parts.append(f"Viewer: {u}\n")
            parts.append(f"Host: {a}\n")
        parts.append("\n")

    parts.append(f"[VIEWER COMMENT] {danmu}")
    parts.append(PROMPT_SUFFIX)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Model loading and generation
# ---------------------------------------------------------------------------

def load_model():
    base = ensure_local_path(BASE_MODEL)
    print(">>> Loading base model:", base)
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    )
    if ADAPTER_DIR:
        print(">>> Loading LoRA adapter from:", ADAPTER_DIR)
        model = Swift.from_pretrained(model, ADAPTER_DIR)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_n(model, tokenizer, prompt: str, n: Optional[int] = None) -> List[Tuple[str, Optional[str], Optional[Dict[str, Any]]]]:
    """Generate n candidate responses in a single batched call."""
    g = CONFIG["generation"]
    n = n or g["num_candidates"]

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    out = model.generate(
        **inputs,
        max_new_tokens=g["max_new_tokens"],
        do_sample=True,
        temperature=g["temperature"],
        top_p=g["top_p"],
        repetition_penalty=g["repetition_penalty"],
        no_repeat_ngram_size=g["no_repeat_ngram_size"],
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=n,
    )

    results = []
    for i in range(out.shape[0]):
        gen_ids = out[i, input_len:]
        raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        json_text = extract_first_json(raw_text)
        data = safe_parse_json(json_text) if json_text else None
        results.append((raw_text, json_text, data))
    return results


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------

def pick_best_candidate(
    candidates: List[Dict[str, Any]],
    danmu: str,
    prev_user: str,
    recent_spoken: List[str],
    recent_sigs: List[str],
    matched_product: Optional[Dict[str, Any]] = None,
    all_product_names: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Score each candidate by combining repetition, compliance, and
    product-alignment penalties with completeness and topical-relevance
    bonuses, then return the best-scoring candidate.

    Internally this is implemented as a penalty score where lower is
    better (penalties summed, bonuses subtracted); equivalently this is
    the negative of the "higher is better" reranker score described in
    the paper.
    """
    if not candidates:
        return None

    best = None
    best_score = float("inf")
    r = CONFIG["reranking"]

    for c in candidates:
        data = c["data"]
        spoken = c["spoken"]
        if not spoken:
            continue

        comp = completeness_score(data)
        pen = diversity_penalty(spoken, recent_spoken)
        sig_pen = signature_penalty(opening_signature(spoken), recent_sigs)
        ban_pen = banned_phrase_penalty(spoken)
        prod_pen = product_alignment_penalty(spoken, matched_product)
        place_pen = placeholder_penalty(spoken)
        other_name_pen = other_product_name_penalty(spoken, matched_product, all_product_names or [])
        ing_pen = ingredient_mention_penalty_when_unmatched(spoken) if (not matched_product) else 0.0

        rel = relevance_score(danmu, spoken, prev_user=prev_user)  # range from CONFIG["relevance"]
        relevance_weight = r["relevance_weight"] or 0

        score = (pen + sig_pen + ban_pen + prod_pen + place_pen + other_name_pen + ing_pen) \
            - comp - (rel * relevance_weight)

        short_threshold = r["short_response_threshold"]
        if short_threshold is not None and len(spoken) < short_threshold:
            score += r["short_response_penalty"] or 0

        if score < best_score:
            best_score = score
            best = c

    return best


# ---------------------------------------------------------------------------
# CLI entry point (interactive demo)
# ---------------------------------------------------------------------------

def render_dialogue(danmu: str, spoken: Optional[str]):
    print("\n===== Live-room dialogue =====")
    print(f"Viewer: {danmu}")
    if not spoken:
        print("Host: (didn't catch that, try again)")
    else:
        print(f"Host: {spoken}")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Viewer comment -> host reply (interactive demo; supports an idle pitch-script loop)"
    )
    ap.add_argument("--product_library", type=str, default=PRODUCT_LIBRARY_PATH, help="Path to product library JSON")
    ap.add_argument("--scripts_json", type=str, default=SCRIPT_LIBRARY_PATH, help="Path to idle pitch-script JSON (array of {product_id, content})")
    ap.add_argument("--idle_mode", action="store_true", help="Enable idle pitch loop when no comments arrive")
    ap.add_argument("--idle_after", type=float, default=None, help="Seconds of inactivity before idle narration resumes")
    ap.add_argument("--idle_interval", type=float, default=None, help="Seconds between idle narration segments")
    ap.add_argument("--idle_log", type=str, default="", help="Optional file to log idle narration text")
    ap.add_argument("--idle_poll", type=float, default=CONFIG["service"]["idle_poll_seconds"], help="stdin polling interval in seconds")
    return ap.parse_args()


def main(args):
    model, tokenizer = load_model()
    print("=== Viewer comment -> host reply (interactive demo). Ctrl+C to exit. ===")

    recent_spoken: List[str] = []
    recent_sigs: List[str] = []
    history: List[Tuple[str, str]] = []

    prod = load_product_library(args.product_library)
    products = prod.get("products", []) or []
    if products:
        print(f">>> Loaded product library: {len(products)} products from {args.product_library}")
    else:
        print(f">>> Product library not found / empty: {args.product_library} (skipping)")

    script_items = load_idle_scripts(args.scripts_json)
    if script_items:
        print(f">>> Loaded idle scripts: {len(script_items)} items from {args.scripts_json}")
    else:
        print(f">>> Idle scripts not found / empty: {args.scripts_json} (idle channel will be silent)")

    idle_player = IdleScriptPlayer(script_items) if args.idle_mode else None
    idle_after = args.idle_after if args.idle_after is not None else 0.0
    idle_interval = args.idle_interval if args.idle_interval is not None else 0.0
    last_danmu_ts = time.time()
    next_idle_ts = last_danmu_ts + idle_after

    recent_window = CONFIG["diversity"]["recent_window"]
    sig_window = CONFIG["diversity"]["signature_window"]
    history_turns = CONFIG["dialogue"]["history_turns"]

    def _handle_one_danmu(danmu: str) -> None:
        nonlocal recent_spoken, recent_sigs, history, last_danmu_ts, next_idle_ts
        danmu = (danmu or "").strip()
        if not danmu:
            return

        matched_product = match_product(danmu, products)
        prompt = build_prompt(history, danmu, matched_product=matched_product)
        triples = generate_n(model, tokenizer, prompt)

        candidates = []
        for raw, json_text, data in triples:
            if not data:
                continue
            spoken = build_spoken_reply(data)
            if not spoken:
                continue
            candidates.append({"data": data, "spoken": spoken, "raw": raw, "json_text": json_text})

        prev_user = history[-1][0] if history else ""
        all_product_names = [p.get("name", "") for p in products if isinstance(p, dict)]
        best = pick_best_candidate(
            candidates, danmu, prev_user,
            recent_spoken, recent_sigs,
            matched_product=matched_product,
            all_product_names=all_product_names,
        )
        spoken = best["spoken"] if best else None

        render_dialogue(danmu, spoken)

        if spoken:
            recent_spoken.append(spoken)
            recent_sigs.append(opening_signature(spoken))
            if recent_window and len(recent_spoken) > recent_window:
                recent_spoken = recent_spoken[-recent_window:]
            if sig_window and len(recent_sigs) > sig_window:
                recent_sigs = recent_sigs[-sig_window:]

            history.append((danmu, spoken))
            if history_turns:
                history = history[-history_turns:]

        last_danmu_ts = time.time()
        next_idle_ts = last_danmu_ts + idle_after

    try:
        if not args.idle_mode:
            while True:
                try:
                    danmu = input("\n[comment] ").strip()
                except KeyboardInterrupt:
                    print("\nExiting.")
                    break
                if not danmu:
                    continue
                _handle_one_danmu(danmu)
        else:
            print("=== Idle mode ON: idle pitch script plays when no comments arrive ===")
            print("Note: idle text is not printed by default; use --idle_log to write it to a file.")

            while True:
                r, _, _ = select.select([sys.stdin], [], [], float(args.idle_poll))
                if r:
                    line = sys.stdin.readline()
                    if line == "":  # EOF
                        print("\nstdin EOF, exiting.")
                        break
                    danmu = line.strip()
                    if not danmu:
                        continue
                    _handle_one_danmu(danmu)
                    continue

                now = time.time()
                if idle_player and now >= next_idle_ts:
                    idle_pid, idle_text = idle_player.next_item()
                    if idle_text:
                        _append_text_line(args.idle_log, idle_text)

                        recent_spoken.append(idle_text)
                        recent_sigs.append(opening_signature(idle_text))
                        if recent_window and len(recent_spoken) > recent_window:
                            recent_spoken = recent_spoken[-recent_window:]
                        if sig_window and len(recent_sigs) > sig_window:
                            recent_sigs = recent_sigs[-sig_window:]

                    next_idle_ts = now + idle_interval

    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
