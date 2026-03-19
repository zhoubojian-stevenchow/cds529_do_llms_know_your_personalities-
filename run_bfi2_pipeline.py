#!/usr/bin/env python3
"""
LLM Personality Recognition Pipeline: BFI-2 Assessment
=======================================================
CDS529 Research Project — "Diagnosing the Judge: Benchmarking LLMs' Personality Recognition"

This script:
1. Feeds character profiles (name, Big Five traits, wiki description) from a CSV to multiple LLMs
2. Asks each LLM to impersonate the character and complete the BFI-2 (60-item) questionnaire
3. Scores the BFI-2 responses to derive Big Five personality dimensions
4. Compares LLM-derived scores against the CSV baseline (ground truth)
5. Saves all results locally

Execution: PARALLEL across LLMs (ThreadPoolExecutor)

Usage:
    1. pip install anthropic openai google-generativeai zhipuai pandas tqdm matplotlib python-dotenv
    2. Create a .env file with your API keys (see below)
    3. python run_bfi2_pipeline.py --csv diverse_big5_profiles.csv --output ./results

.env file format:
    ANTHROPIC_API_KEY=sk-ant-...
    OPENAI_API_KEY=sk-...
    GOOGLE_API_KEY=AIza...
    ZHIPUAI_API_KEY=...
    DEEPSEEK_API_KEY=sk-...
"""

import os
import sys
import json
import time
import datetime
import re
import traceback
import threading
import argparse
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# ============================================================
# Load .env file (if python-dotenv is installed)
# ============================================================
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[OK] Loaded .env file")
except ImportError:
    print("[INFO] python-dotenv not installed, reading keys from environment variables directly")

# ============================================================
# LLM SDKs — import with graceful fallback
# ============================================================
SDK_AVAILABLE = {}

try:
    import anthropic
    SDK_AVAILABLE['claude'] = True
except ImportError:
    SDK_AVAILABLE['claude'] = False
    print("[WARN] anthropic SDK not installed — Claude disabled. pip install anthropic")

try:
    import openai
    SDK_AVAILABLE['chatgpt'] = True
    SDK_AVAILABLE['deepseek'] = True
except ImportError:
    SDK_AVAILABLE['chatgpt'] = False
    SDK_AVAILABLE['deepseek'] = False
    print("[WARN] openai SDK not installed — ChatGPT & DeepSeek disabled. pip install openai")

try:
    from google import genai as genai_new
    SDK_AVAILABLE['gemini'] = True
    GEMINI_SDK = 'new'  # google-genai
except ImportError:
    try:
        import google.generativeai as genai_old
        SDK_AVAILABLE['gemini'] = True
        GEMINI_SDK = 'old'  # google-generativeai (deprecated)
        print("[WARN] Using deprecated google-generativeai. Consider: pip install google-genai")
    except ImportError:
        SDK_AVAILABLE['gemini'] = False
        GEMINI_SDK = None
        print("[WARN] No Gemini SDK installed — Gemini disabled. pip install google-genai")

try:
    from zhipuai import ZhipuAI
    SDK_AVAILABLE['characterglm'] = True
except ImportError:
    SDK_AVAILABLE['characterglm'] = False
    print("[WARN] zhipuai SDK not installed — CharacterGLM disabled. pip install zhipuai")


# ╔══════════════════════════════════════════════════════════════╗
# ║  1. CONFIGURATION                                           ║
# ╚══════════════════════════════════════════════════════════════╝

def load_api_keys() -> dict:
    """Load API keys from environment variables (or .env file)."""
    return {
        'ANTHROPIC_API_KEY': os.environ.get('ANTHROPIC_API_KEY', ''),
        'OPENAI_API_KEY':    os.environ.get('OPENAI_API_KEY', ''),
        'GOOGLE_API_KEY':    os.environ.get('GOOGLE_API_KEY', ''),
        'ZHIPUAI_API_KEY':   os.environ.get('ZHIPUAI_API_KEY', ''),
        'DEEPSEEK_API_KEY':  os.environ.get('DEEPSEEK_API_KEY', ''),
    }


def build_model_config(api_keys: dict) -> dict:
    """Build model configuration based on available keys and SDKs."""
    return {
        "claude": {
            "enabled": bool(api_keys['ANTHROPIC_API_KEY']) and SDK_AVAILABLE.get('claude', False),
            "model_name": "claude-sonnet-4-20250514",
            "display_name": "Claude (Anthropic)",
        },
        "chatgpt": {
            "enabled": bool(api_keys['OPENAI_API_KEY']) and SDK_AVAILABLE.get('chatgpt', False),
            "model_name": "gpt-4o",
            "display_name": "ChatGPT (OpenAI)",
        },
        "gemini": {
            "enabled": bool(api_keys['GOOGLE_API_KEY']) and SDK_AVAILABLE.get('gemini', False),
            "model_name": "gemini-2.0-flash",
            "display_name": "Gemini (Google)",
        },
        "characterglm": {
            "enabled": bool(api_keys['ZHIPUAI_API_KEY']) and SDK_AVAILABLE.get('characterglm', False),
            "model_name": "charglm-4",
            "display_name": "CharacterGLM (ZhipuAI)",
        },
        "deepseek": {
            "enabled": bool(api_keys['DEEPSEEK_API_KEY']) and SDK_AVAILABLE.get('deepseek', False),
            "model_name": "deepseek-chat",
            "display_name": "DeepSeek",
        },
    }


# ╔══════════════════════════════════════════════════════════════╗
# ║  2. BFI-2 QUESTIONNAIRE & SCORING                           ║
# ╚══════════════════════════════════════════════════════════════╝

BFI2_ITEMS = {
    1:  "Is outgoing, sociable.",
    2:  "Is compassionate, has a soft heart.",
    3:  "Tends to be disorganized.",
    4:  "Is relaxed, handles stress well.",
    5:  "Has few artistic interests.",
    6:  "Has an assertive personality.",
    7:  "Is respectful, treats others with respect.",
    8:  "Tends to be lazy.",
    9:  "Stays optimistic after experiencing a setback.",
    10: "Is curious about many different things.",
    11: "Rarely feels excited or eager.",
    12: "Tends to find fault with others.",
    13: "Is dependable, steady.",
    14: "Is moody, has up and down mood swings.",
    15: "Is inventive, finds clever ways to do things.",
    16: "Tends to be quiet.",
    17: "Feels little sympathy for others.",
    18: "Is systematic, likes to keep things in order.",
    19: "Can be tense.",
    20: "Is fascinated by art, music, or literature.",
    21: "Is dominant, acts as a leader.",
    22: "Starts arguments with others.",
    23: "Has difficulty getting started on tasks.",
    24: "Feels secure, comfortable with self.",
    25: "Avoids intellectual, philosophical discussions.",
    26: "Is less active than other people.",
    27: "Has a forgiving nature.",
    28: "Can be somewhat careless.",
    29: "Is emotionally stable, not easily upset.",
    30: "Has little creativity.",
    31: "Is sometimes shy, introverted.",
    32: "Is helpful and unselfish with others.",
    33: "Keeps things neat and tidy.",
    34: "Worries a lot.",
    35: "Values art and beauty.",
    36: "Finds it hard to influence people.",
    37: "Is sometimes rude to others.",
    38: "Is efficient, gets things done.",
    39: "Often feels sad.",
    40: "Is complex, a deep thinker.",
    41: "Is full of energy.",
    42: "Is suspicious of others' intentions.",
    43: "Is reliable, can always be counted on.",
    44: "Keeps their emotions under control.",
    45: "Has difficulty imagining things.",
    46: "Is talkative.",
    47: "Can be cold and uncaring.",
    48: "Leaves a mess, doesn't clean up.",
    49: "Rarely feels anxious or afraid.",
    50: "Thinks poetry and plays are boring.",
    51: "Prefers to have others take charge.",
    52: "Is polite, courteous to others.",
    53: "Is persistent, works until the task is finished.",
    54: "Tends to feel depressed, blue.",
    55: "Has little interest in abstract ideas.",
    56: "Shows a lot of enthusiasm.",
    57: "Assumes the best about people.",
    58: "Sometimes behaves irresponsibly.",
    59: "Is temperamental, gets emotional easily.",
    60: "Is original, comes up with new ideas.",
}

# BFI-2 Scoring Key — (item_number, is_reverse_scored)
# Reference: Soto & John (2017), Table 2
BFI2_SCORING = {
    "Extraversion": [
        (1, False), (6, False), (11, True), (16, True),
        (21, False), (26, True), (31, True), (36, True),
        (41, False), (46, False), (51, True), (56, False),
    ],
    "Agreeableness": [
        (2, False), (7, False), (12, True), (17, True),
        (22, True), (27, False), (32, False), (37, True),
        (42, True), (47, True), (52, False), (57, False),
    ],
    "Conscientiousness": [
        (3, True), (8, True), (13, False), (18, False),
        (23, True), (28, True), (33, False), (38, False),
        (43, False), (48, True), (53, False), (58, True),
    ],
    "Neuroticism": [
        (4, True), (9, True), (14, False), (19, False),
        (24, True), (29, True), (34, False), (39, False),
        (44, True), (49, True), (54, False), (59, False),
    ],
    "Openness": [
        (5, True), (10, False), (15, False), (20, False),
        (25, True), (30, True), (35, False), (40, False),
        (45, True), (50, True), (55, True), (60, False),
    ],
}


def score_bfi2(responses: Dict[int, int]) -> Dict[str, float]:
    """Score a complete set of BFI-2 responses (1-5 Likert)."""
    scores = {}
    for domain, items in BFI2_SCORING.items():
        domain_scores = []
        for item_num, is_reverse in items:
            raw = responses.get(item_num)
            if raw is None or raw not in [1, 2, 3, 4, 5]:
                continue
            score = (6 - raw) if is_reverse else raw
            domain_scores.append(score)
        scores[domain] = round(np.mean(domain_scores), 4) if domain_scores else None
    return scores


def bfi2_mean_to_percentile(mean_score: float) -> float:
    """Convert BFI-2 domain mean (1-5) to percentile (0-100). Linear mapping."""
    return round(max(0, min(100, (mean_score - 1) / 4 * 100)), 2)


# ╔══════════════════════════════════════════════════════════════╗
# ║  3. CSV LOADER                                               ║
# ╚══════════════════════════════════════════════════════════════╝

def parse_pct(val) -> Optional[int]:
    """Extract numeric percentage from strings like 'Extraversion 75%'."""
    if pd.isna(val):
        return None
    val = str(val).strip()
    match = re.search(r'(\d+)\s*%', val)
    if match:
        return int(match.group(1))
    try:
        return int(float(val))
    except ValueError:
        return None


def load_character_data(csv_path: str) -> pd.DataFrame:
    """Load and parse the character CSV file."""
    # Try utf-8 first, fall back to latin-1 (handles non-UTF-8 bytes in wiki descriptions)
    try:
        df_raw = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        print("[INFO] UTF-8 failed, reading CSV with latin-1 encoding")
        df_raw = pd.read_csv(csv_path, encoding='latin-1')

    df = pd.DataFrame()
    df['id'] = df_raw['id']
    df['name'] = df_raw['name']
    df['category'] = df_raw.get('category', '')
    df['subcategory'] = df_raw.get('subcategory', '')
    df['wiki_description'] = df_raw['wiki_description'].fillna('')

    # Ground truth Big Five
    df['baseline_extraversion']      = df_raw['big5_extraversion'].apply(parse_pct)
    df['baseline_neuroticism']       = df_raw['big5_neuroticism'].apply(parse_pct)
    df['baseline_conscientiousness'] = df_raw['big5_conscientiousness'].apply(parse_pct)
    df['baseline_agreeableness']     = df_raw['big5_agreeableness'].apply(parse_pct)
    df['baseline_openness']          = df_raw['big5_openness'].apply(parse_pct)

    return df


# ╔══════════════════════════════════════════════════════════════╗
# ║  4. PROMPT ENGINEERING                                       ║
# ╚══════════════════════════════════════════════════════════════╝

def build_system_prompt(name: str, big5: dict, description: str) -> str:
    """Build the system prompt to make the LLM impersonate the character."""
    return f"""You are now fully embodying the character described below. You must think, feel, \
and respond EXACTLY as this character would — not as an AI assistant. Stay in character at all times.

=== CHARACTER PROFILE ===
Name: {name}

Personality Traits (Big Five, percentile scale 0-100):
  - Extraversion:      {big5.get('extraversion', 'N/A')}%
  - Neuroticism:       {big5.get('neuroticism', 'N/A')}%
  - Conscientiousness:  {big5.get('conscientiousness', 'N/A')}%
  - Agreeableness:     {big5.get('agreeableness', 'N/A')}%
  - Openness:          {big5.get('openness', 'N/A')}%

Background Description:
{description}
=== END OF PROFILE ===

IMPORTANT INSTRUCTIONS:
- You ARE this character. Answer all questions from their perspective.
- Use your knowledge of this character's personality, experiences, and worldview.
- When answering personality questionnaires, respond honestly AS the character, \
reflecting their actual tendencies, not idealized versions.
- Do NOT break character or add disclaimers about being an AI."""


def build_bfi2_user_prompt() -> str:
    """Build the user prompt with all 60 BFI-2 items."""
    items_text = "\n".join(
        f"  {num}. {text}" for num, text in sorted(BFI2_ITEMS.items())
    )
    return f"""Now, please complete the Big Five Inventory-2 (BFI-2) personality questionnaire AS the character you are portraying.

For each of the 60 statements below, rate how much you agree or disagree that the statement describes you (the character), using this scale:
  1 = Disagree strongly
  2 = Disagree a little
  3 = Neutral; no opinion
  4 = Agree a little
  5 = Agree strongly

"I am someone who..."
{items_text}

RESPOND WITH ONLY A JSON OBJECT mapping item numbers (as strings) to your ratings (as integers 1-5).
Example format: {{"1": 4, "2": 2, "3": 5, ...}}

Do NOT include any text, explanation, or commentary — ONLY the JSON object."""


# ╔══════════════════════════════════════════════════════════════╗
# ║  5. LLM API CALLERS (all thread-safe)                        ║
# ╚══════════════════════════════════════════════════════════════╝

def retry_api_call(func, max_retries=3, base_delay=5):
    """Wrapper that retries on transient errors with exponential backoff."""
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                err_str = str(e).lower()
                if any(kw in err_str for kw in ['rate', '429', '500', '502', '503', 'timeout', 'overloaded']):
                    wait = base_delay * (2 ** attempt)
                    print(f"    [Retry] {func.__name__} in {wait}s (attempt {attempt+1}/{max_retries}): {e}")
                    time.sleep(wait)
                else:
                    raise
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper


def make_api_callers(api_keys: dict, model_config: dict) -> dict:
    """Create API caller functions for each enabled model."""
    callers = {}

    # Claude
    if model_config['claude']['enabled']:
        def call_claude(system_prompt: str, user_prompt: str) -> str:
            client = anthropic.Anthropic(api_key=api_keys['ANTHROPIC_API_KEY'])
            response = client.messages.create(
                model=model_config['claude']['model_name'],
                max_tokens=2000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.0,
            )
            return response.content[0].text
        callers['claude'] = retry_api_call(call_claude)

    # ChatGPT
    if model_config['chatgpt']['enabled']:
        def call_chatgpt(system_prompt: str, user_prompt: str) -> str:
            client = openai.OpenAI(api_key=api_keys['OPENAI_API_KEY'])
            response = client.chat.completions.create(
                model=model_config['chatgpt']['model_name'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=2000,
            )
            return response.choices[0].message.content
        callers['chatgpt'] = retry_api_call(call_chatgpt)

    # Gemini
    if model_config['gemini']['enabled']:
        if GEMINI_SDK == 'new':
            def call_gemini(system_prompt: str, user_prompt: str) -> str:
                client = genai_new.Client(api_key=api_keys['GOOGLE_API_KEY'])
                response = client.models.generate_content(
                    model=model_config['gemini']['model_name'],
                    contents=user_prompt,
                    config=genai_new.types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.0,
                        max_output_tokens=2000,
                    ),
                )
                return response.text
        else:
            def call_gemini(system_prompt: str, user_prompt: str) -> str:
                genai_old.configure(api_key=api_keys['GOOGLE_API_KEY'])
                model = genai_old.GenerativeModel(
                    model_name=model_config['gemini']['model_name'],
                    system_instruction=system_prompt,
                )
                response = model.generate_content(
                    user_prompt,
                    generation_config=genai_old.types.GenerationConfig(
                        temperature=0.0,
                        max_output_tokens=2000,
                    ),
                )
                return response.text
        callers['gemini'] = retry_api_call(call_gemini)

    # CharacterGLM
    if model_config['characterglm']['enabled']:
        def call_characterglm(system_prompt: str, user_prompt: str) -> str:
            client = ZhipuAI(api_key=api_keys['ZHIPUAI_API_KEY'])
            response = client.chat.completions.create(
                model=model_config['characterglm']['model_name'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                meta={
                    "user_info": "A psychology researcher conducting a personality assessment.",
                    "bot_info": system_prompt[:1500],
                    "bot_name": "Character",
                    "user_name": "Researcher",
                },
                temperature=0.1,
                max_tokens=2000,
            )
            return response.choices[0].message.content
        callers['characterglm'] = retry_api_call(call_characterglm)

    # DeepSeek
    if model_config['deepseek']['enabled']:
        def call_deepseek(system_prompt: str, user_prompt: str) -> str:
            client = openai.OpenAI(
                api_key=api_keys['DEEPSEEK_API_KEY'],
                base_url="https://api.deepseek.com",
            )
            response = client.chat.completions.create(
                model=model_config['deepseek']['model_name'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=2000,
            )
            return response.choices[0].message.content
        callers['deepseek'] = retry_api_call(call_deepseek)

    return callers


# ╔══════════════════════════════════════════════════════════════╗
# ║  6. RESPONSE PARSER                                          ║
# ╚══════════════════════════════════════════════════════════════╝

def parse_bfi2_response(raw_text: str) -> Tuple[Optional[Dict[int, int]], str]:
    """Parse the LLM's BFI-2 JSON response into {item_number: rating}."""
    if not raw_text or not raw_text.strip():
        return None, "Empty response"

    text = raw_text.strip()

    # Extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*\n?({.*?})\s*\n?```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        brace_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if brace_match:
            text = brace_match.group(0)

    # Parse JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        text_clean = re.sub(r',\s*}', '}', text)
        text_clean = re.sub(r'\n', ' ', text_clean)
        try:
            data = json.loads(text_clean)
        except json.JSONDecodeError as e:
            return None, f"JSON parse error: {e}"

    # Convert to {int: int}
    responses = {}
    for k, v in data.items():
        try:
            item_num = int(k)
            rating = int(v)
            if 1 <= item_num <= 60 and 1 <= rating <= 5:
                responses[item_num] = rating
        except (ValueError, TypeError):
            continue

    if len(responses) < 50:
        return responses if responses else None, (
            f"Only {len(responses)}/60 valid items parsed. "
            f"Missing: {sorted(set(range(1,61)) - set(responses.keys()))}"
        )

    missing = sorted(set(range(1, 61)) - set(responses.keys()))
    warn = f"Missing items: {missing}" if missing else ""
    return responses, warn


# ╔══════════════════════════════════════════════════════════════╗
# ║  7. CHECKPOINT & LOGGING (thread-safe)                       ║
# ╚══════════════════════════════════════════════════════════════╝

_file_lock = threading.Lock()


def load_checkpoint(path: str) -> dict:
    """Load checkpoint: {model_key -> set of completed character IDs}."""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {k: set(v) for k, v in data.items()}
    return {}


def save_checkpoint(path: str, completed: dict):
    """Save checkpoint (thread-safe)."""
    with _file_lock:
        serializable = {k: list(v) for k, v in completed.items()}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f)


def append_raw_log(path: str, entry: dict):
    """Append one entry to JSONL log (thread-safe)."""
    with _file_lock:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


# ╔══════════════════════════════════════════════════════════════╗
# ║  8. WORKER FUNCTION (runs inside thread)                     ║
# ╚══════════════════════════════════════════════════════════════╝

def process_single_llm_task(
    model_key: str,
    caller,
    model_config: dict,
    system_prompt: str,
    user_prompt: str,
    row: pd.Series,
) -> dict:
    """
    Process one (character, model) pair:
      1. Call LLM API → impersonate character
      2. Parse BFI-2 JSON response
      3. Score BFI-2 domains
      4. Compute gap vs ground truth
    """
    char_id = int(row['id'])
    char_name = row['name']

    result_entry = {
        'character_id': char_id,
        'character_name': char_name,
        'model_key': model_key,
        'model_name': model_config[model_key]['display_name'],
        'timestamp': datetime.datetime.now().isoformat(),
        'baseline_extraversion':      row['baseline_extraversion'],
        'baseline_neuroticism':       row['baseline_neuroticism'],
        'baseline_conscientiousness': row['baseline_conscientiousness'],
        'baseline_agreeableness':     row['baseline_agreeableness'],
        'baseline_openness':          row['baseline_openness'],
    }

    try:
        raw_response = caller(system_prompt, user_prompt)
        result_entry['raw_response'] = raw_response
        result_entry['api_error'] = None

        parsed, parse_warning = parse_bfi2_response(raw_response)
        result_entry['parse_warning'] = parse_warning
        result_entry['items_parsed'] = len(parsed) if parsed else 0

        if parsed:
            result_entry['bfi2_item_responses'] = parsed

            domain_means = score_bfi2(parsed)
            domain_pcts = {
                k: bfi2_mean_to_percentile(v)
                for k, v in domain_means.items() if v is not None
            }

            result_entry['scored_extraversion_mean']      = domain_means.get('Extraversion')
            result_entry['scored_agreeableness_mean']     = domain_means.get('Agreeableness')
            result_entry['scored_conscientiousness_mean'] = domain_means.get('Conscientiousness')
            result_entry['scored_neuroticism_mean']       = domain_means.get('Neuroticism')
            result_entry['scored_openness_mean']          = domain_means.get('Openness')

            result_entry['scored_extraversion_pct']      = domain_pcts.get('Extraversion')
            result_entry['scored_agreeableness_pct']     = domain_pcts.get('Agreeableness')
            result_entry['scored_conscientiousness_pct'] = domain_pcts.get('Conscientiousness')
            result_entry['scored_neuroticism_pct']       = domain_pcts.get('Neuroticism')
            result_entry['scored_openness_pct']          = domain_pcts.get('Openness')

            for dim in ['extraversion', 'neuroticism', 'conscientiousness', 'agreeableness', 'openness']:
                baseline = row[f'baseline_{dim}']
                scored = domain_pcts.get(dim.capitalize())
                if baseline is not None and scored is not None:
                    result_entry[f'gap_{dim}'] = round(scored - baseline, 2)
                else:
                    result_entry[f'gap_{dim}'] = None

            result_entry['status'] = 'OK'
        else:
            result_entry['status'] = 'PARSE_FAILED'

    except Exception as e:
        result_entry['raw_response'] = None
        result_entry['api_error'] = f"{type(e).__name__}: {e}"
        result_entry['status'] = 'API_ERROR'

    return result_entry


# ╔══════════════════════════════════════════════════════════════╗
# ║  9. VISUALIZATION                                            ║
# ╚══════════════════════════════════════════════════════════════╝

def generate_charts(df_valid: pd.DataFrame, model_config: dict, output_dir: str, timestamp_str: str):
    """Generate and save analysis charts."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for scripts
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed — skipping charts. pip install matplotlib")
        return

    dims = ['extraversion', 'neuroticism', 'conscientiousness', 'agreeableness', 'openness']
    gap_cols = [f'gap_{d}' for d in dims]
    models_list = sorted(df_valid['model_key'].unique())

    # Chart 1: MAE per model per dimension
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    x = np.arange(len(dims))
    width = 0.8 / len(models_list)

    for i, mk in enumerate(models_list):
        sub = df_valid[df_valid['model_key'] == mk]
        maes = [sub[f'gap_{d}'].abs().mean() for d in dims]
        ax.bar(x + i * width, maes, width, label=model_config[mk]['display_name'])

    ax.set_xticks(x + width * (len(models_list) - 1) / 2)
    ax.set_xticklabels([d.capitalize() for d in dims])
    ax.set_ylabel('Mean Absolute Error (pp)')
    ax.set_title('BFI-2 Personality Impersonation Accuracy: MAE by Dimension')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path1 = os.path.join(output_dir, f"chart_mae_by_dimension_{timestamp_str}.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"  Chart saved: {path1}")

    # Chart 2: Gap heatmap
    fig, axes = plt.subplots(1, len(models_list), figsize=(5 * len(models_list), 6), sharey=True)
    if len(models_list) == 1:
        axes = [axes]

    for ax, mk in zip(axes, models_list):
        sub = df_valid[df_valid['model_key'] == mk].copy()
        gap_matrix = sub.set_index('character_name')[[f'gap_{d}' for d in dims]]
        gap_matrix.columns = [d.capitalize() for d in dims]
        im = ax.imshow(gap_matrix.values, cmap='RdBu_r', vmin=-50, vmax=50, aspect='auto')
        ax.set_xticks(range(len(dims)))
        ax.set_xticklabels([d.capitalize() for d in dims], rotation=45, ha='right')
        ax.set_yticks(range(len(gap_matrix)))
        ax.set_yticklabels(gap_matrix.index, fontsize=8)
        ax.set_title(model_config[mk]['display_name'])

    fig.colorbar(im, ax=axes, label='Gap: LLM Scored - Ground Truth (pp)', shrink=0.8)
    fig.suptitle('Impersonation Gap Heatmap (per character x per LLM)', fontsize=13)
    plt.tight_layout()
    path2 = os.path.join(output_dir, f"chart_gap_heatmap_{timestamp_str}.png")
    fig.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Chart saved: {path2}")

    # Chart 3: Overall ranking
    fig, ax = plt.subplots(figsize=(8, 4))
    overall_mae = df_valid.groupby('model_key')[gap_cols].apply(
        lambda x: x.abs().mean().mean()
    ).sort_values()
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(overall_mae)))
    bars = ax.barh(
        [model_config[mk]['display_name'] for mk in overall_mae.index],
        overall_mae.values,
        color=colors,
    )
    ax.set_xlabel('Overall MAE (pp) — lower = better impersonation')
    ax.set_title('LLM Personality Impersonation Ranking')
    for bar, val in zip(bars, overall_mae.values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}', va='center', fontsize=10)
    plt.tight_layout()
    path3 = os.path.join(output_dir, f"chart_llm_ranking_{timestamp_str}.png")
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    print(f"  Chart saved: {path3}")


# ╔══════════════════════════════════════════════════════════════╗
# ║  10. MAIN PIPELINE                                           ║
# ╚══════════════════════════════════════════════════════════════╝

def run_pipeline(csv_path: str, output_dir: str):
    """Run the full parallel BFI-2 experimental pipeline."""

    # ── Setup ──
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "checkpoint.json")
    raw_log_path = os.path.join(output_dir, "raw_responses_log.jsonl")

    api_keys = load_api_keys()
    model_config = build_model_config(api_keys)
    enabled_models = [k for k, v in model_config.items() if v['enabled']]

    if not enabled_models:
        print("\n[ERROR] No models enabled. Check your .env file or environment variables.")
        print("Required format in .env:")
        print("  ANTHROPIC_API_KEY=sk-ant-...")
        print("  OPENAI_API_KEY=sk-...")
        print("  GOOGLE_API_KEY=AIza...")
        print("  ZHIPUAI_API_KEY=...")
        print("  DEEPSEEK_API_KEY=sk-...")
        sys.exit(1)

    callers = make_api_callers(api_keys, model_config)

    print(f"\n{'='*60}")
    print(f" BFI-2 LLM Personality Impersonation Pipeline")
    print(f"{'='*60}")
    print(f" Enabled models:  {', '.join(model_config[mk]['display_name'] for mk in enabled_models)}")
    print(f" CSV:             {csv_path}")
    print(f" Output:          {output_dir}")
    print(f" Execution:       PARALLEL ({len(enabled_models)} LLMs per character)")
    print(f"{'='*60}\n")

    # ── Load data ──
    df = load_character_data(csv_path)
    print(f"Loaded {len(df)} character profiles.\n")

    # ── Build BFI-2 prompt (same for all) ──
    bfi2_user_prompt = build_bfi2_user_prompt()

    # ── Load checkpoint ──
    completed = load_checkpoint(checkpoint_path)
    for mk in enabled_models:
        if mk not in completed:
            completed[mk] = set()

    total_tasks = len(df) * len(enabled_models)
    already_done = sum(
        1 for _, row in df.iterrows()
        for mk in enabled_models
        if int(row['id']) in completed.get(mk, set())
    )
    remaining = total_tasks - already_done

    print(f"Total tasks:   {total_tasks} ({len(df)} characters x {len(enabled_models)} models)")
    print(f"Already done:  {already_done}")
    print(f"Remaining:     {remaining}\n")

    if remaining == 0:
        print("All tasks already completed! Delete checkpoint.json to re-run.")
        return

    # ── Run pipeline ──
    all_results = []
    pbar = tqdm(total=remaining, desc="Pipeline")

    for idx, row in df.iterrows():
        char_id = int(row['id'])
        char_name = row['name']

        models_todo = [
            mk for mk in enabled_models
            if char_id not in completed.get(mk, set())
        ]

        if not models_todo:
            continue

        # Build system prompt for this character
        big5_input = {
            'extraversion':      row['baseline_extraversion'],
            'neuroticism':       row['baseline_neuroticism'],
            'conscientiousness': row['baseline_conscientiousness'],
            'agreeableness':     row['baseline_agreeableness'],
            'openness':          row['baseline_openness'],
        }
        description = str(row['wiki_description'])[:2000]
        system_prompt = build_system_prompt(char_name, big5_input, description)

        print(f"\n[{idx+1}/{len(df)}] {char_name} (ID={char_id}) -> {models_todo}")

        # PARALLEL: fire all LLM calls simultaneously
        with ThreadPoolExecutor(max_workers=len(models_todo)) as executor:
            future_to_model = {
                executor.submit(
                    process_single_llm_task,
                    model_key=mk,
                    caller=callers[mk],
                    model_config=model_config,
                    system_prompt=system_prompt,
                    user_prompt=bfi2_user_prompt,
                    row=row,
                ): mk
                for mk in models_todo
            }

            for future in as_completed(future_to_model):
                mk = future_to_model[future]
                try:
                    result = future.result()
                    status = result.get('status', '?')
                    items = result.get('items_parsed', 0)
                    print(f"    OK {mk:15s}  {items}/60 items  [{status}]")
                except Exception as e:
                    result = {
                        'character_id': char_id,
                        'character_name': char_name,
                        'model_key': mk,
                        'model_name': model_config[mk]['display_name'],
                        'timestamp': datetime.datetime.now().isoformat(),
                        'api_error': f"Thread exception: {e}",
                        'status': 'THREAD_ERROR',
                    }
                    print(f"    FAIL {mk:15s}  {e}")

                all_results.append(result)
                append_raw_log(raw_log_path, result)
                completed[mk].add(char_id)
                pbar.update(1)

        save_checkpoint(checkpoint_path, completed)

    pbar.close()
    print(f"\n{'='*60}")
    print(f" Pipeline complete! {len(all_results)} results collected.")
    print(f"{'='*60}\n")

    # ── Save results ──
    timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    df_results = pd.DataFrame(all_results)

    summary_cols = [
        'character_id', 'character_name', 'model_key', 'model_name',
        'status', 'items_parsed',
        'baseline_extraversion', 'baseline_neuroticism',
        'baseline_conscientiousness', 'baseline_agreeableness', 'baseline_openness',
        'scored_extraversion_pct', 'scored_neuroticism_pct',
        'scored_conscientiousness_pct', 'scored_agreeableness_pct', 'scored_openness_pct',
        'gap_extraversion', 'gap_neuroticism',
        'gap_conscientiousness', 'gap_agreeableness', 'gap_openness',
        'parse_warning', 'api_error', 'timestamp',
    ]
    existing_cols = [c for c in summary_cols if c in df_results.columns]
    df_summary = df_results[existing_cols].copy()

    # Summary CSV
    csv_out = os.path.join(output_dir, f"bfi2_results_summary_{timestamp_str}.csv")
    df_summary.to_csv(csv_out, index=False)
    print(f"Summary CSV:  {csv_out}")

    # Full JSON
    json_out = os.path.join(output_dir, f"bfi2_results_full_{timestamp_str}.json")
    results_ser = []
    for r in all_results:
        entry = r.copy()
        if 'bfi2_item_responses' in entry and entry['bfi2_item_responses']:
            entry['bfi2_item_responses'] = {str(k): v for k, v in entry['bfi2_item_responses'].items()}
        results_ser.append(entry)
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(results_ser, f, ensure_ascii=False, indent=2)
    print(f"Full JSON:    {json_out}")

    # ── Analysis ──
    gap_cols = ['gap_extraversion', 'gap_neuroticism', 'gap_conscientiousness',
                'gap_agreeableness', 'gap_openness']
    df_valid = df_summary.dropna(subset=gap_cols)

    if len(df_valid) > 0:
        print(f"\n{'='*60}")
        print(f" GAP ANALYSIS ({len(df_valid)} valid results)")
        print(f"{'='*60}")

        # Status counts
        if 'status' in df_results.columns:
            print(f"\nStatus counts:\n{df_results['status'].value_counts().to_string()}")

        # MAE
        mae_per_model = df_valid.groupby('model_key')[gap_cols].apply(
            lambda x: x.abs().mean()
        ).round(2)
        mae_per_model.columns = [c.replace('gap_', 'MAE_') for c in mae_per_model.columns]
        mae_per_model['MAE_overall'] = mae_per_model.mean(axis=1).round(2)
        print(f"\nMAE per model (lower = better impersonation):\n{mae_per_model.to_string()}")

        mae_out = os.path.join(output_dir, f"mae_summary_{timestamp_str}.csv")
        mae_per_model.to_csv(mae_out)
        print(f"\nMAE saved:    {mae_out}")

        # Bias
        bias_per_model = df_valid.groupby('model_key')[gap_cols].mean().round(2)
        bias_per_model.columns = [c.replace('gap_', 'bias_') for c in bias_per_model.columns]
        print(f"\nBias (+ = overshoot, - = undershoot):\n{bias_per_model.to_string()}")

        # Correlation
        print(f"\nPearson r (ground truth vs LLM scored):")
        for mk in sorted(df_valid['model_key'].unique()):
            sub = df_valid[df_valid['model_key'] == mk]
            corrs = {}
            for dim in ['extraversion', 'neuroticism', 'conscientiousness', 'agreeableness', 'openness']:
                b = sub[f'baseline_{dim}'].astype(float)
                s = sub[f'scored_{dim}_pct'].astype(float)
                if b.std() > 0 and s.std() > 0:
                    corrs[dim] = round(b.corr(s), 3)
                else:
                    corrs[dim] = None
            print(f"  {model_config[mk]['display_name']:25s}: {corrs}")

        # Per-character report
        print(f"\n{'='*60}")
        print(f" PER-CHARACTER REPORT")
        print(f"{'='*60}")
        for char_name in df_valid['character_name'].unique():
            char_data = df_valid[df_valid['character_name'] == char_name]
            row0 = char_data.iloc[0]
            print(f"\n  {char_name}")
            print(f"  GROUND TRUTH:  E={row0['baseline_extraversion']:>3}%  "
                  f"N={row0['baseline_neuroticism']:>3}%  "
                  f"C={row0['baseline_conscientiousness']:>3}%  "
                  f"A={row0['baseline_agreeableness']:>3}%  "
                  f"O={row0['baseline_openness']:>3}%")
            for _, r in char_data.iterrows():
                mk = r['model_key']
                print(f"  {model_config[mk]['display_name']:22s}: "
                      f"E={r.get('scored_extraversion_pct','?'):>6} ({r.get('gap_extraversion','?'):>+6})  "
                      f"N={r.get('scored_neuroticism_pct','?'):>6} ({r.get('gap_neuroticism','?'):>+6})  "
                      f"C={r.get('scored_conscientiousness_pct','?'):>6} ({r.get('gap_conscientiousness','?'):>+6})  "
                      f"A={r.get('scored_agreeableness_pct','?'):>6} ({r.get('gap_agreeableness','?'):>+6})  "
                      f"O={r.get('scored_openness_pct','?'):>6} ({r.get('gap_openness','?'):>+6})")

        # Charts
        print(f"\nGenerating charts...")
        generate_charts(df_valid, model_config, output_dir, timestamp_str)
    else:
        print("\nNo valid gap data for analysis.")

    print(f"\n{'='*60}")
    print(f" ALL DONE. Results in: {output_dir}")
    print(f"{'='*60}")


# ╔══════════════════════════════════════════════════════════════╗
# ║  ENTRY POINT                                                 ║
# ╚══════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BFI-2 LLM Personality Impersonation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_bfi2_pipeline.py --csv diverse_big5_profiles.csv
  python run_bfi2_pipeline.py --csv data/profiles.csv --output ./my_results

Environment variables (or .env file):
  ANTHROPIC_API_KEY=sk-ant-...
  OPENAI_API_KEY=sk-...
  GOOGLE_API_KEY=AIza...
  ZHIPUAI_API_KEY=...
  DEEPSEEK_API_KEY=sk-...
        """,
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to the character profiles CSV file",
    )
    parser.add_argument(
        "--output", default="./bfi2_results",
        help="Output directory for results (default: ./bfi2_results)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV file not found: {args.csv}")
        sys.exit(1)

    run_pipeline(csv_path=args.csv, output_dir=args.output)
