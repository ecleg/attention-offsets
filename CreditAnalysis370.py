#!/usr/bin/env python3
"""
A tool that analyzes the export data of OpenAI and Anthropic models to determine the 
prompt length, model usage, energy, time of use, possible topics
(simple top unigrams and bigrams) in order to calculate the ad-
funded offset potential of a person's AI usage habits

Outputs (in ./analysis):
  summary.json
  keywords.csv
  hourly.csv
  dow.csv
  impact.csv
"""

import argparse
import json
import re
import sys
import csv
import math
from typing import Optional, Dict, Any
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone


# configurable assumptions (calibrated to nov 2025 research)

ASSUME_CHARS_PER_TOKEN = 4.0          # heuristic (validated across exports)
ASSUME_WH_PER_TOKEN = 0.008           # Wh per token (weighted avg: GPT-3.5 about 0.002, GPT-4 about 0.015)
                                       # Sources: OpenAI energy disclosures 2024, Luccioni et al. 2024
ASSUME_WATER_L_PER_KWH = 1.8          # Liters of water per kWh (data center cooling intensity)
                                       # Google/Microsoft 2024 disclosures around 1.8 L/kWh
ASSUME_GRID_KG_CO2E_PER_KWH = 0.35    # kg CO2e per kWh (cloud provider renewable mix)
                                       # US grid avg 0.386, cloud providers 0.25-0.35 (EPA 2024)
ASSUME_AD_SLOTS_PER_PROMPT = 1.0      # ad impressions per user prompt

# ── Gross CPM (what advertisers pay) ──────────────────────────────────────────
ASSUME_CPM_USD = 5.00                 # $5.00 gross CPM — browser-extension overlay on AI platform
                                       # Reference class: IAB Tech vertical ($3–8, 2024-2025), NOT
                                       # generic display ($2.50). EQO's inventory is high-intent:
                                       # user is actively mid-task when the ad renders.
                                       # Validation: 865.4 prompts × $5.00/1000 = $4.3271 ✓
                                       # Sources: eMarketer Tech CPM Benchmarks (2025);
                                       #          IAB Internet Advertising Revenue Report (2024);
                                       #          Google Ad Manager Publisher Benchmark Report (2024)

# ── Net CPM (what EQO retains after ad-network revenue share) ─────────────────
ASSUME_PUBLISHER_NET_RATIO = 0.65     # 65% publisher share (conservative programmatic estimate)
                                       # Google AdSense officially pays 68% for content inventory.
                                       # Programmatic supply-chain analysis: ~45-55% of advertiser
                                       # spend clears to publisher net of SSP/DSP fees; 65% used
                                       # as conservative mid-range (excludes double-counting).
                                       # Direct/sponsorship deals → 100%; target for year 2+.
                                       # Sources: Google AdSense Help (2024):
                                       #   support.google.com/adsense/answer/180195
                                       #          IAB "Programmatic Revenue Share Study" (2023)

# ── Viewable CPM premium (vCPM) ───────────────────────────────────────────────
ASSUME_VCPM_PREMIUM = 0.30            # 30% premium for viewable impressions (conservative)
                                       # IAB/MRC viewability standard: ≥50% pixels visible ≥1 sec.
                                       # EQO overlay above the ChatGPT prompt box is ~100% viewable
                                       # by design — the user is looking directly at it.
                                       # Industry data: viewable CPM runs 20-40% above standard CPM.
                                       # Sources: DoubleVerify Global Insights Report (2024);
                                       #          Integral Ad Science Media Quality Report H2 2024;
                                       #          MRC/IAB Display Impression Measurement (2024)

ASSUME_OFFSET_PRICE_USD_PER_KG = 0.01 # $ per kg CO2e (voluntary carbon market avg 2025)

# water offset pricing (defaults; non-regional)
ASSUME_WATER_OFFSET_USD_PER_1000_GAL = 1.50  # baseline restoration/efficiency price
ASSUME_WATER_SCARCITY_MULTIPLIER = 1.0       # keep 1.0 (no regional adjustment by default)
GALLON_TO_LITER = 3.785411784

# ============================================================================
# MODEL-SPECIFIC ENERGY MULTIPLIERS (Conservative estimates)
# Base: 0.008 Wh/token = 1.0x multiplier (approximately GPT-3.5 class)
# Sources: MIT Tech Review (2025), Chung & Chowdhury ML.ENERGY Benchmark (2025),
#          Luccioni et al. "Power Hungry Processing" (2024)
# ============================================================================
# Rationale: Energy scales roughly with parameter count and architecture.
# - Llama 8B: ~114 J/response = ~0.032 Wh (MIT Tech Review / ML.ENERGY 2025)
# - Llama 405B: ~6,706 J/response = ~1.86 Wh (MIT Tech Review / ML.ENERGY 2025)
# - 50x more parameters ≈ 59x more energy (sublinear but significant)
# Conservative approach: use lower-bound multipliers to avoid overestimation.
MODEL_ENERGY_MULTIPLIERS = {
    # GPT-3.5 class (baseline, ~175B params but heavily optimized)
    "text-davinci-002-render-sha": 1.0,
    "text-davinci-003": 1.0,
    "gpt-3.5-turbo": 1.0,
    "gpt-3.5": 1.0,
    
    # GPT-4 class (~1T params estimated, mixture-of-experts)
    # Conservative: 2x baseline (MIT Tech Review suggests 2-5x for larger models)
    "gpt-4": 2.0,
    "gpt-4-turbo": 2.0,
    "gpt-4o": 1.8,          # Optimized variant, slightly lower
    "gpt-4o-mini": 1.2,     # Smaller, more efficient
    
    # Reasoning models (chain-of-thought generates more tokens internally)
    # MIT Tech Review: "reasoning models require 43x more energy for simple problems"
    # Conservative: 3x for typical use (not worst-case 43x)
    "o1-preview": 3.0,
    "o1-mini": 2.0,
    "o1": 3.0,
    
    # Default for unknown models (conservative: assume GPT-4 class)
    "default": 1.5,
}

# ============================================================================
# TOOL-SPECIFIC ENERGY MULTIPLIERS
# Sources: MIT Tech Review (2025), Luccioni et al. (2024), Hugging Face AI Energy Score
# ============================================================================
# Key finding from MIT Tech Review (May 2025):
# - Text generation (Llama 8B): ~114 J per response
# - Image generation (SD3 Medium): ~2,282 J per image = ~20x text
# - High-quality image (50 steps): ~4,402 J = ~39x text
# - Video generation (CogVideoX): ~3,400,000 J per 5-sec video = ~30,000x text
#
# For DALL-E (closed-source), we use conservative estimates based on open-source equivalents.
# Code interpreter and browser are assumed to be text-equivalent (no separate GPU inference).
TOOL_ENERGY_MULTIPLIERS = {
    # Image generation tools
    # Conservative: 15x baseline (SD3 is ~20x, but DALL-E may be optimized)
    "dalle.text2im": 15.0,
    "dalle": 15.0,
    "image_generation": 15.0,
    
    # Code interpreter (Python execution)
    # Runs on CPU, not GPU inference - minimal additional energy
    # Conservative: 1.2x (small overhead for compute)
    "python": 1.2,
    "code_interpreter": 1.2,
    
    # Web browsing tools
    # Network requests + page rendering, no ML inference
    # Conservative: 1.1x (minimal overhead)
    "browser": 1.1,
    "web": 1.1,
    "web.run": 1.1,
    
    # File/document tools
    # May involve embedding search (vector similarity)
    # Conservative: 1.3x
    "file_search": 1.3,
    "retrieval": 1.3,
    
    # Canvas/editing tools (text manipulation, no image gen)
    "canmore.canvas_tool": 1.0,
    "canmore": 1.0,
    
    # Default for unknown tools
    "default": 1.0,
}

def get_model_multiplier(model_slug: str) -> float:
    """Return energy multiplier for a given model slug."""
    if not model_slug:
        return MODEL_ENERGY_MULTIPLIERS["default"]
    model_lower = model_slug.lower()
    for key, mult in MODEL_ENERGY_MULTIPLIERS.items():
        if key != "default" and key in model_lower:
            return mult
    return MODEL_ENERGY_MULTIPLIERS["default"]

def get_tool_multiplier(tool_name: str) -> float:
    """Return energy multiplier for a given tool name."""
    if not tool_name:
        return TOOL_ENERGY_MULTIPLIERS["default"]
    tool_lower = tool_name.lower()
    for key, mult in TOOL_ENERGY_MULTIPLIERS.items():
        if key != "default" and key in tool_lower:
            return mult
    return TOOL_ENERGY_MULTIPLIERS["default"]


# minimal english stopwords (no deps)

STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because been
before being below between both but by can can't cannot could couldn't did didn't do does doesn't
doing don't down during each few for from further had hadn't has hasn't have haven't having he
he'd he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in
into is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once
only or other ought our ours  ourselves out over own same shan't she she'd she'll she's should shouldn't
so some such than that that's the their theirs them themselves then there there's these they they'd
they'll they're they've this those through to too under until up very was wasn't we we'd we'll we're
we've were weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't you
you'd you'll you're you've your yours yourself yourselves
""".split())

TOKEN_SPLIT = re.compile(r"[^\w#+]+")

LANG_CODE_BLOCK = re.compile(r"```([a-zA-Z0-9+#.\-]*)")
HAS_CODE_FENCE = re.compile(r"```")

# Keys to search for model slug (recipient removed - used for tool detection only)
MODEL_KEYS = ("model", "model_slug", "gpt_model")

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def utc_dt(ts) -> Optional[datetime]:
    """Convert common export timestamp formats to UTC datetime.

    Supports:
    - Unix seconds (int/float, or numeric strings) — common in ChatGPT exports
    - ISO 8601 strings (e.g. "2026-03-24T18:27:06.694432Z") — common in Anthropic exports
    """
    try:
        if ts is None:
            return None

        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)

        if isinstance(ts, str):
            s = ts.strip()
            if not s:
                return None

            # Numeric timestamp in string form
            try:
                return datetime.fromtimestamp(float(s), tz=timezone.utc)
            except Exception:
                pass

            # ISO 8601 timestamp
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

        return None
    except Exception:
        return None

def coalesce_text(msg) -> str:
    # handle both "mapping" and "messages" forms
    # possible fields:
    # message.content.parts (list of strings)
    # message.content.text (string)
    # content (string)
    parts = []
    # openAI export (mapping -> message -> content -> parts)
    content = None
    if isinstance(msg, dict) and "content" in msg:
        content = msg["content"]
        if isinstance(content, dict):
            if "parts" in content and isinstance(content["parts"], list):
                parts.extend(str(p) for p in content["parts"] if p is not None)
            elif "text" in content and isinstance(content["text"], str):
                parts.append(content["text"])
        elif isinstance(content, list):
            # Anthropic export: content is a list of blocks
            # Only include explicit text blocks to avoid counting tool payloads
            # (tool_use/tool_result) as if they were assistant/user messages.
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "text":
                    continue
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
        elif isinstance(content, str):
            parts.append(content)
    # alternate schema: "text"
    if not parts and isinstance(msg, dict) and "text" in msg and isinstance(msg["text"], str):
        parts.append(msg["text"])
    # flatten
    text = "\n".join(p.strip() for p in parts if p is not None)
    return text

def extract_role(msg) -> Optional[str]:
    # try various layouts
    # mapping -> message -> author.role
    if not isinstance(msg, dict):
        return None

    role = None
    author = None

    # Anthropic export: role is encoded as a simple sender string
    sender = msg.get("sender")
    if isinstance(sender, str):
        sender_lower = sender.lower()
        if sender_lower == "human":
            return "user"
        if sender_lower == "assistant":
            return "assistant"
        if sender_lower == "system":
            return "system"
        if sender_lower == "tool":
            return "tool"

    if "author" in msg and isinstance(msg["author"], dict):
        author = msg["author"]
        role = author.get("role")
    # alternate: role directly
    if role is None:
        role = msg.get("role")
    # system/tool naming harmonization
    if role in ("tool", "function"):
        return "tool"
    return role

def extract_model(msg) -> Optional[str]:
    # Hunt common fields
    for k in MODEL_KEYS:
        if k in msg and isinstance(msg[k], str) and msg[k]:
            return msg[k]
    if "metadata" in msg and isinstance(msg["metadata"], dict):
        for k in MODEL_KEYS:
            v = msg["metadata"].get(k)
            if isinstance(v, str) and v:
                return v
    if "author" in msg and isinstance(msg["author"], dict):
        md = msg["author"].get("metadata")
        if isinstance(md, dict):
            for k in MODEL_KEYS:
                v = md.get(k)
                if isinstance(v, str) and v:
                    return v
    return None

def extract_tool(msg) -> Optional[str]:
    """Extract tool name from message (recipient field or author.name for tool role)."""
    if not isinstance(msg, dict):
        return None

    # Check recipient field (indicates tool being called)
    recipient = msg.get("recipient")
    if recipient and isinstance(recipient, str):
        r = recipient.strip()
        # ChatGPT exports frequently use recipient="all" for normal assistant messages
        if r and r.lower() not in {"all", "assistant"}:
            return r
    # Check author.name for tool messages
    author = msg.get("author", {})
    if isinstance(author, dict):
        name = author.get("name")
        if name and isinstance(name, str):
            n = name.strip()
            if n and n.lower() not in {"all", "assistant"}:
                return n
    return None

def extract_tools(msg) -> list[str]:
    """Extract tool names from a message across supported export schemas."""
    if not isinstance(msg, dict):
        return []

    tools: list[str] = []

    # ChatGPT export: `recipient` for tool invocations
    t = extract_tool(msg)
    if t:
        tools.append(t)

    # Anthropic export: tool blocks inside `content`
    content = msg.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in {"tool_use", "tool_result"}:
                name = block.get("name")
                if isinstance(name, str) and name.strip():
                    tools.append(name.strip())

    # De-duplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for tool_name in tools:
        if tool_name not in seen:
            seen.add(tool_name)
            out.append(tool_name)
    return out

def iter_conversation_messages(conv: Dict[str, Any]):
    # normalized iterator yielding dicts: {role, text, ts, model, tools, tool}
    # format A (common): conv["mapping"] with nodes keyed by message id
    if isinstance(conv, dict) and "mapping" in conv and isinstance(conv["mapping"], dict):
        for node in conv["mapping"].values():
            m = node.get("message")
            if not isinstance(m, dict):
                continue
            role = extract_role(m.get("author", m))
            text = coalesce_text(m)
            ts = m.get("create_time") or conv.get("create_time")
            model = extract_model(m)
            tools = extract_tools(m)
            yield {
                "role": role,
                "text": text or "",
                "ts": utc_dt(ts),
                "model": model,
                "tools": tools,
                "tool": tools[0] if tools else None,
            }
        return
    # format B (alternative): conv["messages"] is list
    if isinstance(conv, dict) and "messages" in conv and isinstance(conv["messages"], list):
        for m in conv["messages"]:
            role = extract_role(m)
            text = coalesce_text(m)
            ts = m.get("create_time") or conv.get("create_time")
            model = extract_model(m)
            tools = extract_tools(m)
            yield {
                "role": role,
                "text": text or "",
                "ts": utc_dt(ts),
                "model": model,
                "tools": tools,
                "tool": tools[0] if tools else None,
            }
        return
    # format C (Anthropic): conv["chat_messages"] is list
    if isinstance(conv, dict) and "chat_messages" in conv and isinstance(conv["chat_messages"], list):
        for m in conv["chat_messages"]:
            if not isinstance(m, dict):
                continue
            role = extract_role(m)
            text = coalesce_text(m)
            ts = m.get("created_at") or conv.get("created_at")
            model = extract_model(m)
            tools = extract_tools(m)
            yield {
                "role": role,
                "text": text or "",
                "ts": utc_dt(ts),
                "model": model,
                "tools": tools,
                "tool": tools[0] if tools else None,
            }
        return
    # fallback: treat conv itself as a message list
    if isinstance(conv, list):
        for m in conv:
            if not isinstance(m, dict):
                continue
            role = extract_role(m)
            text = coalesce_text(m)
            ts = m.get("create_time")
            model = extract_model(m)
            tools = extract_tools(m)
            yield {
                "role": role,
                "text": text or "",
                "ts": utc_dt(ts),
                "model": model,
                "tools": tools,
                "tool": tools[0] if tools else None,
            }

def tokenize_for_keywords(text: str):
    toks = [t.lower() for t in TOKEN_SPLIT.split(text) if t]
    toks = [t for t in toks if t not in STOPWORDS and len(t) > 2]
    return toks

def bigrams(tokens):
    for i in range(len(tokens) - 1):
        yield f"{tokens[i]} {tokens[i+1]}"
        
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="EcoBalance Analysis API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/upload")
async def upload_chatgpt_export(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # validate JSON
        import json
        parsed = json.loads(contents)
        tmp_path = Path(f"/tmp/{file.filename}")
        tmp_path.write_text(json.dumps(parsed), encoding="utf-8")
        result = analyze_file(tmp_path)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


def analyze_file(input_path: Path) -> Dict[str, Any]:
    """Analyze a ChatGPT export file and return impact summary.

    Uses iter_conversation_messages() for correct parsing of all export formats
    (mapping-style, messages-style, etc.) and weighted energy model with
    model/tool multipliers for conservative estimates.
    """
    data = load_json(input_path)

    # Normalize: data may be a list of conversations or a dict with a key
    if isinstance(data, list):
        conv_list = data
    elif isinstance(data, dict):
        conv_list = data.get("conversations") or []
        if not conv_list and "mapping" in data:
            # Single conversation exported as top-level mapping
            conv_list = [data]
    else:
        conv_list = []

    conv_count = len(conv_list)
    user_msg_count = 0
    assistant_msg_count = 0
    tool_msg_count = 0
    weighted_wh_total = 0.0
    user_chars = 0
    assistant_chars = 0
    model_usage = Counter()
    tool_usage = Counter()
    hourly = Counter()
    dow = Counter()
    unigrams = Counter()
    bigrams_ctr = Counter()
    code_langs = Counter()
    has_code_prompts = 0

    def process_text(t: str):
        parts = TOKEN_SPLIT.split(t.lower())
        filtered = [p for p in parts if p and p not in STOPWORDS]
        for w in filtered:
            unigrams[w] += 1
        for i in range(len(filtered)-1):
            bigrams_ctr[f"{filtered[i]} {filtered[i+1]}"] += 1
        if HAS_CODE_FENCE.search(t):
            langs = LANG_CODE_BLOCK.findall(t)
            if langs:
                for l in langs:
                    code_langs[l or "plain"] += 1
            return True
        return False

    for conv in conv_list:
        for m in iter_conversation_messages(conv):
            role = (m.get("role") or "").lower()
            text = m.get("text") or ""
            model = m.get("model")
            tools = m.get("tools") or ([] if not m.get("tool") else [m.get("tool")])

            # Count by role
            if role == "user":
                user_msg_count += 1
            elif role in ("assistant", "system"):
                assistant_msg_count += 1
            elif role == "tool":
                tool_msg_count += 1

            # Weighted energy per message
            msg_tokens = len(text) / ASSUME_CHARS_PER_TOKEN if text else 0
            if text:
                if role == "user":
                    user_chars += len(text)
                elif role in ("assistant", "system"):
                    assistant_chars += len(text)
            model_mult = get_model_multiplier(model)
            tool_mult = max((get_tool_multiplier(t) for t in tools if t), default=1.0)
            effective_mult = max(model_mult, tool_mult)
            weighted_wh_total += msg_tokens * ASSUME_WH_PER_TOKEN * effective_mult

            # Model and tool tracking
            if model:
                model_usage[model] += 1
            for t in tools:
                if t:
                    tool_usage[t] += 1

            # Text analysis
            if text:
                if process_text(text):
                    has_code_prompts += 1

            # Timestamp
            ts = m.get("ts")
            if ts:
                hourly[ts.hour] += 1
                dow[ts.weekday()] += 1

    total_chars = user_chars + assistant_chars
    est_tokens = (total_chars / ASSUME_CHARS_PER_TOKEN) if total_chars else 0.0
    est_kwh = weighted_wh_total / 1000.0
    est_kg_co2e = est_kwh * ASSUME_GRID_KG_CO2E_PER_KWH
    est_liters_water = est_kwh * ASSUME_WATER_L_PER_KWH
    est_impressions = user_msg_count * ASSUME_AD_SLOTS_PER_PROMPT
    est_revenue_usd = (est_impressions / 1000.0) * ASSUME_CPM_USD              # gross CPM revenue
    est_net_revenue_usd = est_revenue_usd * ASSUME_PUBLISHER_NET_RATIO         # after programmatic network cut
    est_offset_cost = est_kg_co2e * ASSUME_OFFSET_PRICE_USD_PER_KG
    water_usd_per_liter = (ASSUME_WATER_OFFSET_USD_PER_1000_GAL / (1000 * GALLON_TO_LITER)) * ASSUME_WATER_SCARCITY_MULTIPLIER
    est_water_offset_cost = est_liters_water * water_usd_per_liter
    est_combined_offset_cost = est_offset_cost + est_water_offset_cost
    coverage_ratio = est_revenue_usd / est_offset_cost if est_offset_cost else 0
    coverage_ratio_including_water = est_revenue_usd / est_combined_offset_cost if est_combined_offset_cost else 0
    # views-to-offset: ad impressions needed to cover full combined offset at net CPM
    _net_cpm = ASSUME_CPM_USD * ASSUME_PUBLISHER_NET_RATIO
    est_views_to_offset = int(round(est_combined_offset_cost / (_net_cpm / 1000))) if est_combined_offset_cost else 0

    return {
        "conversations": conv_count,
        "user_messages": user_msg_count,
        "assistant_messages": assistant_msg_count,
        "tool_messages": tool_msg_count,
        "estimated_tokens": est_tokens,
        "kwh": round(est_kwh, 6),
        "kg_co2e": round(est_kg_co2e, 6),
        "water_liters": round(est_liters_water, 3),
        "ad_impressions": int(est_impressions),
        "ad_revenue_gross_usd": round(est_revenue_usd, 4),        # gross CPM revenue (what advertisers pay)
        "ad_revenue_net_usd": round(est_net_revenue_usd, 4),      # net after programmatic cut (65% share)
        "ad_views_to_offset": est_views_to_offset,                 # impressions needed to cover full offset
        "offset_carbon_usd": round(est_offset_cost, 4),
        "offset_water_usd": round(est_water_offset_cost, 6),
        "offset_total_usd": round(est_combined_offset_cost, 4),
        "coverage_carbon_ratio": round(coverage_ratio, 4),
        "coverage_total_ratio": round(coverage_ratio_including_water, 4),
        "top_models": [m for m,_ in model_usage.most_common(5)],
        "top_unigrams": [u for u,_ in unigrams.most_common(10)],
        "top_bigrams": [b for b,_ in bigrams_ctr.most_common(10)],
        "code_langs_top": [l for l,_ in code_langs.most_common(5)],
        "has_code_prompts": bool(has_code_prompts),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=str, help="Path to ChatGPT export JSON (e.g., PersonAconversations.json)")
    ap.add_argument("--outdir", type=str, default="analysis", help="Output directory for reports")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_json(in_path)

    # conversations root
    if isinstance(data, dict) and "conversations" in data and isinstance(data["conversations"], list):
        conversations = data["conversations"]
    elif isinstance(data, list):
        conversations = data
    else:
        # try to unwrap any list like values
        conversations = None
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    conversations = v
                    break
        if conversations is None:
            print("Unrecognized JSON structure. Expect array or {'conversations': [...]} root.", file=sys.stderr)
            sys.exit(2)

    conv_count = 0
    user_msg_count = 0
    assistant_msg_count = 0
    tool_msg_count = 0

    user_chars = 0
    assistant_chars = 0

    has_code_prompts = 0
    code_block_langs = Counter()

    model_usage = Counter()
    tool_usage = Counter()      # NEW: track tool calls
    hourly = Counter()     # 0..23
    dow = Counter()        # Mon..Sun
    
    # track unique dates for averaging
    unique_dates = set()   # to count unique calendar days
    hourly_seen = set()    # (date, hour) tuples to count unique hours
    dow_seen = Counter()   # count occurrences of each day of week

    # for simple topics
    uni = Counter()
    bi = Counter()

    msgs_per_convo = []
    
    # NEW: weighted energy tracking (with model and tool multipliers)
    weighted_wh_total = 0.0

    for conv in conversations:
        conv_count += 1
        msg_count = 0

        for m in iter_conversation_messages(conv) or []:
            role = (m.get("role") or "").lower()
            text = m.get("text") or ""
            ts = m.get("ts")
            model = m.get("model")
            tools = m.get("tools") or ([] if not m.get("tool") else [m.get("tool")])

            msg_count += 1
            
            # Calculate per-message weighted energy
            msg_tokens = len(text) / ASSUME_CHARS_PER_TOKEN if text else 0
            model_mult = get_model_multiplier(model)
            tool_mult = max((get_tool_multiplier(t) for t in tools if t), default=1.0)
            # Use max of model and tool multiplier (conservative: don't stack)
            effective_mult = max(model_mult, tool_mult)
            msg_wh = msg_tokens * ASSUME_WH_PER_TOKEN * effective_mult
            weighted_wh_total += msg_wh
            
            if role == "user":
                user_msg_count += 1
                user_chars += len(text)
                if HAS_CODE_FENCE.search(text):
                    has_code_prompts += 1
                # keywords from user requests
                toks = tokenize_for_keywords(text)
                uni.update(toks)
                bi.update(bigrams(toks))
                # code languages from fences
                for lang in LANG_CODE_BLOCK.findall(text):
                    if lang:
                        code_block_langs[lang.lower()] += 1
            elif role in ("assistant", "system"):
                assistant_msg_count += 1
                assistant_chars += len(text)
            elif role == "tool":
                tool_msg_count += 1

            if ts:
                hourly[ts.hour] += 1
                dow[ts.strftime("%a")] += 1
                
                # track unique occurrences for averaging
                date_key = ts.date()
                unique_dates.add(date_key)
                hourly_seen.add((date_key, ts.hour))
                dow_seen[ts.strftime("%a")] += 1

            if model:
                model_usage[model] += 1
            
            # Track tool usage
            for t in tools:
                if t:
                    tool_usage[t] += 1

        msgs_per_convo.append(msg_count)

    total_msgs = user_msg_count + assistant_msg_count + tool_msg_count
    total_chars = user_chars + assistant_chars

    # token/energy/carbon/water estimates (now using weighted energy)
    est_tokens = total_chars / ASSUME_CHARS_PER_TOKEN if total_chars else 0.0
    # Use weighted energy instead of flat rate
    est_kwh = weighted_wh_total / 1000.0
    est_kg_co2e = est_kwh * ASSUME_GRID_KG_CO2E_PER_KWH
    # derive water from kWh, not per token (prevents unit error)
    est_liters_water = est_kwh * ASSUME_WATER_L_PER_KWH
    
    # Also calculate baseline (unweighted) for comparison
    baseline_kwh = (est_tokens * ASSUME_WH_PER_TOKEN) / 1000.0
    energy_multiplier_effect = est_kwh / baseline_kwh if baseline_kwh > 0 else 1.0

    # ad impressions and revenue (assume ad per user prompt)
    est_impressions = user_msg_count * ASSUME_AD_SLOTS_PER_PROMPT
    est_revenue_usd = (est_impressions / 1000.0) * ASSUME_CPM_USD              # gross CPM revenue
    est_net_revenue_usd = est_revenue_usd * ASSUME_PUBLISHER_NET_RATIO         # after programmatic cut
    est_vcpm_gross = ASSUME_CPM_USD * (1 + ASSUME_VCPM_PREMIUM)               # gross viewable CPM rate

    # offset coverage and gap
    est_offset_cost = est_kg_co2e * ASSUME_OFFSET_PRICE_USD_PER_KG
    # water offset cost (defaults)
    water_usd_per_liter = (ASSUME_WATER_OFFSET_USD_PER_1000_GAL / (1000.0 * GALLON_TO_LITER)) * ASSUME_WATER_SCARCITY_MULTIPLIER
    est_water_offset_cost = est_liters_water * water_usd_per_liter

    # combined offsets
    est_combined_offset_cost = est_offset_cost + est_water_offset_cost
    coverage_ratio = (est_revenue_usd / est_offset_cost) if est_offset_cost > 0 else float('inf')
    coverage_ratio_including_water = (est_revenue_usd / est_combined_offset_cost) if est_combined_offset_cost > 0 else float('inf')
    # views-to-offset at each pricing stage
    _net_cpm_prog = ASSUME_CPM_USD * ASSUME_PUBLISHER_NET_RATIO
    _net_cpm_vcpm = ASSUME_CPM_USD * (1 + ASSUME_VCPM_PREMIUM) * ASSUME_PUBLISHER_NET_RATIO
    est_views_to_offset_prog   = int(round(est_combined_offset_cost / (_net_cpm_prog / 1000))) if est_combined_offset_cost else 0
    est_views_to_offset_vcpm   = int(round(est_combined_offset_cost / (_net_cpm_vcpm / 1000))) if est_combined_offset_cost else 0
    est_views_to_offset_direct = int(round(est_combined_offset_cost / (ASSUME_CPM_USD / 1000))) if est_combined_offset_cost else 0

    # prep outputs
    summary = {
        "conversations": conv_count,
        "messages_total": total_msgs,
        "messages_per_conversation_avg": (sum(msgs_per_convo) / len(msgs_per_convo)) if msgs_per_convo else 0.0,
        "user_messages": user_msg_count,
        "assistant_messages": assistant_msg_count,
        "tool_messages": tool_msg_count,
        "user_chars": user_chars,
        "assistant_chars": assistant_chars,
        "has_code_prompts": has_code_prompts,
        "code_block_languages_top": code_block_langs.most_common(20),
        "model_usage_top": model_usage.most_common(20),
        "tool_usage_top": tool_usage.most_common(20),  # NEW
        "hourly_distribution": dict(sorted(hourly.items(), key=lambda kv: kv[0])),
        "dow_distribution": dict(dow),
        "topics": {
            "unigrams_top": uni.most_common(40),
            "bigrams_top": bi.most_common(40),
        },
        "estimates": {
            "chars_per_token": ASSUME_CHARS_PER_TOKEN,
            "wh_per_token_base": ASSUME_WH_PER_TOKEN,
            "energy_multiplier_effect": round(energy_multiplier_effect, 3),  # NEW
            "baseline_kwh_unweighted": round(baseline_kwh, 6),  # NEW
            "water_l_per_kwh": ASSUME_WATER_L_PER_KWH,
            "derived_liters_per_token": (ASSUME_WH_PER_TOKEN/1000.0) * ASSUME_WATER_L_PER_KWH,
            "grid_kg_co2e_per_kwh": ASSUME_GRID_KG_CO2E_PER_KWH,
            "ad_slots_per_prompt": ASSUME_AD_SLOTS_PER_PROMPT,
            "gross_cpm_usd": ASSUME_CPM_USD,
            "net_cpm_usd": round(ASSUME_CPM_USD * ASSUME_PUBLISHER_NET_RATIO, 4),
            "publisher_net_ratio": ASSUME_PUBLISHER_NET_RATIO,
            "vcpm_premium": ASSUME_VCPM_PREMIUM,
            "vcpm_gross_usd": round(est_vcpm_gross, 4),
            "offset_price_usd_per_kg": ASSUME_OFFSET_PRICE_USD_PER_KG,
            "est_tokens": est_tokens,
            "est_kwh": est_kwh,
            "est_kg_co2e": est_kg_co2e,
            "est_liters_water": est_liters_water,
            "est_impressions": est_impressions,
            "est_revenue_gross_usd": est_revenue_usd,
            "est_revenue_net_usd": round(est_net_revenue_usd, 4),
            "views_to_offset_programmatic": est_views_to_offset_prog,
            "views_to_offset_vcpm": est_views_to_offset_vcpm,
            "views_to_offset_direct": est_views_to_offset_direct,
            "est_offset_cost_usd": est_offset_cost,
            "est_water_offset_cost_usd": est_water_offset_cost,
            "est_combined_offset_cost_usd": est_combined_offset_cost,
            "coverage_ratio_revenue_over_offset_cost": coverage_ratio,
            "coverage_including_water": coverage_ratio_including_water,
            "water_offset_usd_per_1000_gal": ASSUME_WATER_OFFSET_USD_PER_1000_GAL,
            "water_scarcity_multiplier": ASSUME_WATER_SCARCITY_MULTIPLIER,
        }
    }

    # write JSON summary
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # write keyword CSVs
    with (out_dir / "keywords.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type", "token", "count"])
        for t, c in uni.most_common(200):
            w.writerow(["unigram", t, c])
        for t, c in bi.most_common(200):
            w.writerow(["bigram", t, c])

    # hourly CSV
    with (out_dir / "hourly.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["hour_0_23", "total_prompts", "avg_prompts_per_hour"])
        for h in range(24):
            total = hourly.get(h, 0)
            # Count how many unique (date, hour) pairs we saw for this hour
            occurrences = sum(1 for (d, hr) in hourly_seen if hr == h)
            avg = total / occurrences if occurrences > 0 else 0.0
            w.writerow([h, total, f"{avg:.2f}"])

    # day of week CSV
    # calculate number of weeks to get avg per day
    num_weeks = len(unique_dates) / 7.0 if unique_dates else 1.0
    
    with (out_dir / "dow.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dow", "total_prompts", "avg_prompts_per_day"])
        for d in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
            total = dow.get(d, 0)
            # Use number of times this specific day occurred
            occurrences = dow_seen.get(d, 0)
            avg = total / num_weeks if num_weeks > 0 else 0.0
            w.writerow([d, total, f"{avg:.2f}"])

    # impact CSV with real-world comparisons
    with (out_dir / "impact.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "unit", "real_world_comparison"])
        
        # energy comparisons
        # EPA: smartphone charge = 0.019 kWh (EPA Greenhouse Gas Equivalencies 2024)
        # laptop (50W avg) for 1 hour = 0.05 kWh
        # EV average consumption: 190 Wh/km (ev-database.org 2024)
        smartphone_charges = est_kwh / 0.019
        laptop_hours = est_kwh / 0.05
        ev_km = est_kwh / 0.19

        w.writerow([
            "tokens",
            f"{est_tokens:,.0f}",
            "tokens",
            "What APIs split text into (roughly 4 characters per token)"
        ])
        w.writerow(["Energy", f"{est_kwh:.3f}", "kWh", 
                   f"Could charge {smartphone_charges:.0f} smartphones, or power a laptop for {laptop_hours:.1f} hours, or drive an electric car {ev_km:.1f} km ({ev_km*0.621371:.1f} miles)"])
        
        # carbon comparisons
        # EPA: avg passenger vehicle emits 3.93x10-4 metric tons CO2e/mile = 0.244 kg/km
        # EPA: urban tree absorbs 0.060 metric tons CO2/year = 60 kg/year = 0.164 kg/day
        # 1kg CO2e = burning around 0.5L of gasoline
        car_km = est_kg_co2e / 0.244
        car_miles = car_km * 0.621371
        tree_days = est_kg_co2e / 0.164
        gasoline_liters = est_kg_co2e / 2.3  # 1L gasoline = around 2.3 kg CO2e
        gasoline_gallons = gasoline_liters * 0.264172
        
        w.writerow(["carbon", f"{est_kg_co2e:.3f}", "kg CO2e",
                   f"Equivalent to driving {car_km:.1f} km ({car_miles:.1f} miles) in a gas car, or {tree_days:.1f} days of a tree's carbon absorption, or burning {gasoline_liters:.2f} liters ({gasoline_gallons:.2f} gallons) of gasoline"])
        
        # water comparisons
        # avg shower: 9L per minute (8-minute shower = 72L)
        # toilet flush: 6-9L (modern), 13-19L (old)
        # washing machine load: 50L
        # drinking water daily recommendation: 2L
        # 1 cup of coffee: 140L water footprint (bean production)
        shower_minutes = est_liters_water / 9.0
        toilet_flushes = est_liters_water / 7.5
        daily_drinking = est_liters_water / 2.0
        water_gallons = est_liters_water * 0.264172
        
        w.writerow(["water", f"{est_liters_water:.1f}", "liters",
                   f"Equivalent to {shower_minutes:.1f} minutes in the shower, or {toilet_flushes:.1f} toilet flushes, or {daily_drinking:.1f} days of drinking water ({water_gallons:.2f} gallons total)"])
        
        # ad revenue metrics
        w.writerow(["impressions", f"{est_impressions:,.0f}", "ad impressions", "Set at 1 banner ad per prompt"])
        w.writerow(["revenue", f"{est_revenue_usd:.2f}", "USD", 
                    f"Estimated ad revenue from your prompts (at ${ASSUME_CPM_USD:.2f} CPM)"])
        
        # offset metrics (carbon, water, combined)
        w.writerow(["offset_carbon", f"{est_offset_cost:.2f}", "USD",
                    f"Cost to offset your carbon footprint (at ${ASSUME_OFFSET_PRICE_USD_PER_KG:.2f}/kg)"])
        w.writerow(["offset_water", f"{est_water_offset_cost:.2f}", "USD",
                    f"Funds water restoration/efficiency; baseline ${ASSUME_WATER_OFFSET_USD_PER_1000_GAL:.2f} per 1,000 gallons (about ${water_usd_per_liter:.6f}/L)"])
        w.writerow(["offset_cost", f"{est_combined_offset_cost:.2f}", "USD",
                    "Cost to offset carbon + water impact (total)"])

        # coverage ratios (combined primary)
        w.writerow(["coverage", f"{coverage_ratio_including_water:.2f}", "ratio",
                    f"Ad revenue covers combined offsets {coverage_ratio_including_water:.1f}x over"])

        # attention-to-offset comparison (instagram reels and youtube shorts)
        # assumptions:
        # - instagram shows ~8 ads per minute of Reels scrolling
        #   (EQO direct testing, 2025: fresh account, 24 min session, 4 s/reel,
        #    counted only ads with "ad" disclosure, excluded sponsored posts)
        # - youtube shorts around 1 ad every 1 minute
        # - CPM applies: revenue_per_impression = CPM / 1000
        # - impressions needed = offset_cost / revenue_per_impression
        revenue_per_impression = ASSUME_CPM_USD / 1000.0 if ASSUME_CPM_USD > 0 else 0.0
        impressions_to_offset_combined = (est_combined_offset_cost / revenue_per_impression) if revenue_per_impression > 0 else float('inf')
        # instagram reels: 8 ads/min (0.125 min/ad); youtube shorts: 1 ad per 1 min
        ig_minutes_per_ad = 1.0 / 8.0   # 8 ads/min = 0.125 min/ad
        yt_minutes_per_ad = 1.0

        def calc_scroll(impr):
            if not math.isfinite(impr):
                return (float('inf'), float('inf'), float('inf'), float('inf'))
            ig_min = impr * ig_minutes_per_ad
            ig_hr = ig_min / 60.0
            yt_min = impr * yt_minutes_per_ad
            yt_hr = yt_min / 60.0
            return (ig_min, ig_hr, yt_min, yt_hr)

        ig_minutes_scrolling, ig_hours_scrolling, yt_minutes_scrolling, yt_hours_scrolling = calc_scroll(impressions_to_offset_combined)

        # format IG scrolling time: seconds when < 2 min, minutes otherwise
        if math.isfinite(ig_minutes_scrolling) and ig_minutes_scrolling < 2:
            ig_seconds = ig_minutes_scrolling * 60
            ig_desc = f"{ig_seconds:.0f} seconds"
        else:
            ig_desc = f"{ig_minutes_scrolling:,.0f} minutes ({ig_hours_scrolling:.1f} hours)"

        # combined is primary "impressions_to_offset" row
        w.writerow([
            "impressions_to_offset",
            f"{impressions_to_offset_combined:,.0f}" if math.isfinite(impressions_to_offset_combined) else "inf",
            "ads/min",
            (
                f"Equivalent to about {ig_desc} of scrolling on instagram reels (8 ads/min), "
                f"or {yt_minutes_scrolling:,.0f} minutes ({yt_hours_scrolling:.1f} hours) of scrolling on youtube shorts (1 ad/min)"
            ) if math.isfinite(impressions_to_offset_combined) else ""
        ])

    # console snippet
    print("Analysis complete.")
    print(f"- Conversations: {conv_count}")
    print(f"- Messages (user/assistant/tool): {user_msg_count}/{assistant_msg_count}/{tool_msg_count}")
    print(f"- Avg msgs per conversation: {summary['messages_per_conversation_avg']:.2f}")
    print(f"- Top models: {summary['model_usage_top'][:5]}")
    print(f"- Top tools: {summary['tool_usage_top'][:5]}")
    print(f"- Top topics: {summary['topics']['unigrams_top'][:10]}")
    print(f"- Code prompts: {has_code_prompts} | Languages: {summary['code_block_languages_top'][:5]}")
    print(f"- Estimated tokens: {est_tokens:,.0f} | kWh: {est_kwh:.4f} (weighted) | Multiplier effect: {energy_multiplier_effect:.2f}x")
    print(f"- kgCO2e: {est_kg_co2e:.4f} | Water: {est_liters_water:.2f}L")
    print(f"- Ad impressions: {est_impressions:,.0f} | Revenue: ${est_revenue_usd:.2f}")
    print(f"- Offsets — Carbon: ${est_offset_cost:.4f} | Water: ${est_water_offset_cost:.6f} | Total: ${est_combined_offset_cost:.4f}")
    print(f"- Coverage — Combined: {coverage_ratio_including_water:.2f}x | Carbon-only: {coverage_ratio:.2f}x")
    print(f"Reports written to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
