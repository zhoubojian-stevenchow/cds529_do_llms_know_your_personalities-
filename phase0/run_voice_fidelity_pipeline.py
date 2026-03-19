#!/usr/bin/env python3
"""
LLM Character Voice Fidelity Pipeline
=======================================
CDS529 — "Can LLMs Talk Like the Characters They Impersonate?"

DESIGN:
  1. Scrape real transcripts, extract character's lines + surrounding context
  2. For each context-response pair, give the context to each LLM and ask it
     to respond AS the character (without seeing the real line)
  3. Compute Empath feature vectors for both real and LLM-generated responses
  4. Cosine similarity between vectors = voice fidelity score

This is an apples-to-apples comparison: same situation, same character,
real script vs LLM attempt.

Usage:
    pip install anthropic openai google-generativeai zhipuai pandas tqdm empath python-dotenv requests beautifulsoup4
    python run_voice_fidelity_pipeline.py --csv diverse_big5_profiles.csv --output ./voice_fidelity_results
"""

import os
import sys
import json
import time
import datetime
import re
import threading
import argparse
import requests
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from bs4 import BeautifulSoup

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── LLM SDKs ──
SDK_AVAILABLE = {}
try:
    import anthropic
    SDK_AVAILABLE['claude'] = True
except ImportError:
    SDK_AVAILABLE['claude'] = False
try:
    import openai
    SDK_AVAILABLE['chatgpt'] = True
    SDK_AVAILABLE['deepseek'] = True
except ImportError:
    SDK_AVAILABLE['chatgpt'] = False
    SDK_AVAILABLE['deepseek'] = False
try:
    from google import genai as genai_new
    SDK_AVAILABLE['gemini'] = True
    GEMINI_SDK = 'new'
except ImportError:
    try:
        import google.generativeai as genai_old
        SDK_AVAILABLE['gemini'] = True
        GEMINI_SDK = 'old'
    except ImportError:
        SDK_AVAILABLE['gemini'] = False
        GEMINI_SDK = None
try:
    from zhipuai import ZhipuAI
    SDK_AVAILABLE['characterglm'] = True
except ImportError:
    SDK_AVAILABLE['characterglm'] = False

try:
    from empath import Empath
    EMPATH_AVAILABLE = True
except ImportError:
    EMPATH_AVAILABLE = False
    print("[WARN] empath not installed. pip install empath")


# ╔══════════════════════════════════════════════════════════════╗
# ║  DYNAMIC TRANSCRIPT SOURCE DISCOVERY                         ║
# ║  Auto-finds transcripts on subslikescript.com from CSV data  ║
# ╚══════════════════════════════════════════════════════════════╝

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

# Cache for discovered URLs so we don't re-search the same show/movie
_url_discovery_cache = {}


def generate_speaker_names(char_name: str) -> List[str]:
    """
    Auto-generate possible speaker label variants from a character name.
    E.g. 'Elizabeth "Beth" Harmon' -> ['Elizabeth', 'ELIZABETH', 'Beth', 'BETH',
          'Harmon', 'HARMON', 'Beth Harmon', 'BETH HARMON', 'Elizabeth Harmon', ...]
    """
    names = set()

    # Clean up the name: remove quotes and parenthetical nicknames
    clean = re.sub(r'["""\']', '', char_name)
    clean = re.sub(r'\s+', ' ', clean).strip()

    # Split into parts
    parts = clean.split()

    # Full name
    names.add(clean)
    names.add(clean.upper())

    # First name
    if parts:
        names.add(parts[0])
        names.add(parts[0].upper())

    # Last name (if multi-word)
    if len(parts) >= 2:
        names.add(parts[-1])
        names.add(parts[-1].upper())
        # First + Last
        first_last = f"{parts[0]} {parts[-1]}"
        names.add(first_last)
        names.add(first_last.upper())

    # Handle nicknames in quotes: 'William "Will" Byers' -> Will
    nickname_match = re.findall(r'["""]([^"""]+)["""]', char_name)
    for nick in nickname_match:
        nick = nick.strip()
        if nick:
            names.add(nick)
            names.add(nick.upper())
            # Nick + Last
            if len(parts) >= 2:
                names.add(f"{nick} {parts[-1]}")
                names.add(f"{nick} {parts[-1]}".upper())

    # Handle "X / Y" format: 'Peter Parker / Spider-Man'
    if '/' in char_name:
        for segment in char_name.split('/'):
            seg = segment.strip().strip('"\'')
            if seg:
                names.add(seg)
                names.add(seg.upper())
                seg_parts = seg.split()
                if seg_parts:
                    names.add(seg_parts[0])
                    names.add(seg_parts[0].upper())

    # Filter out very short names (1-2 chars) to avoid false matches
    names = {n for n in names if len(n) >= 3}

    return list(names)


def search_subslikescript(query: str, category: str) -> Optional[str]:
    """
    Search subslikescript.com for a show/movie and return its base URL.
    category: 'Television' or 'Movies'
    """
    cache_key = f"{category}:{query}"
    if cache_key in _url_discovery_cache:
        return _url_discovery_cache[cache_key]

    # Clean the query: remove year in parentheses for cleaner search
    clean_query = re.sub(r'\s*\(.*?\)\s*', ' ', query).strip()
    # Also try with "Franchise" removed
    clean_query = re.sub(r'\s*\(Franchise\)\s*', '', clean_query).strip()
    clean_query = re.sub(r'\s*(Film Series|Franchise)\s*', '', clean_query).strip()

    search_url = f"https://subslikescript.com/search?q={requests.utils.quote(clean_query)}"

    try:
        resp = requests.get(search_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')

        # Look for links to series or movies
        target_prefix = '/series/' if category == 'Television' else '/movie/'

        for a in soup.find_all('a', href=True):
            href = a['href']
            if target_prefix in href:
                full_url = href if href.startswith('http') else f"https://subslikescript.com{href}"
                # Strip episode/season suffixes to get base URL
                if '/season-' in full_url:
                    full_url = full_url.split('/season-')[0]
                if '/episode-' in full_url:
                    full_url = full_url.split('/episode-')[0]
                _url_discovery_cache[cache_key] = full_url
                return full_url

        # Fallback: try the other category (some shows are listed as movies or vice versa)
        alt_prefix = '/movie/' if category == 'Television' else '/series/'
        for a in soup.find_all('a', href=True):
            href = a['href']
            if alt_prefix in href:
                full_url = href if href.startswith('http') else f"https://subslikescript.com{href}"
                if '/season-' in full_url:
                    full_url = full_url.split('/season-')[0]
                _url_discovery_cache[cache_key] = full_url
                return full_url

    except Exception as e:
        print(f"    [WARN] Search failed for '{query}': {e}")

    _url_discovery_cache[cache_key] = None
    return None


def build_character_transcript_info(row: pd.Series) -> Optional[dict]:
    """
    Build transcript info dict for a character from its CSV row.
    Auto-discovers the subslikescript URL and generates speaker names.
    Returns None if no transcript source can be found.
    """
    char_name = row['name']
    category = row.get('category', '')
    subcategory = str(row.get('subcategory', ''))

    if not subcategory or subcategory == 'nan':
        return None

    # Generate speaker names
    speaker_names = generate_speaker_names(char_name)

    # Search for transcript URL
    print(f"    Searching subslikescript for: '{subcategory}' ({category})...")
    url = search_subslikescript(subcategory, category)

    if not url:
        # Try with just the main title (before colon or dash)
        short_title = re.split(r'[:\-–]', subcategory)[0].strip()
        if short_title != subcategory:
            print(f"    Retrying with: '{short_title}'...")
            url = search_subslikescript(short_title, category)

    if not url:
        print(f"    [SKIP] No transcript found for '{subcategory}'")
        return None

    print(f"    Found: {url}")

    if '/series/' in url:
        return {
            'type': 'series',
            'url_base': url,
            'speaker_names': speaker_names,
        }
    elif '/movie/' in url:
        return {
            'type': 'movies',
            'urls': [url],
            'speaker_names': speaker_names,
        }

    return None


def scrape_series_episode_list(url_base: str) -> List[str]:
    try:
        resp = requests.get(url_base, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if '/season-' in href and '/episode-' in href:
                full_url = href if href.startswith('http') else f"https://subslikescript.com{href}"
                if full_url not in links:
                    links.append(full_url)
        return links
    except Exception as e:
        print(f"    [WARN] Failed to get episode list: {e}")
        return []


def scrape_transcript_page(url: str) -> str:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        script_div = soup.find('div', class_='full-script')
        if script_div:
            return script_div.get_text(separator='\n').strip()
        article = soup.find('article')
        if article:
            return article.get_text(separator='\n').strip()
        return ""
    except Exception as e:
        print(f"    [WARN] Failed to scrape {url}: {e}")
        return ""


def scrape_all_transcripts(char_name, char_info, cache_dir, max_episodes=10):
    safe_name = re.sub(r'[^\w\-]', '_', char_name)
    cache_file = os.path.join(cache_dir, f"{safe_name}_transcripts.txt")

    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            text = f.read()
        if len(text) > 100:
            print(f"    [Cache] Loaded {len(text)} chars")
            return text

    all_text = []
    if char_info['type'] == 'series':
        episode_urls = scrape_series_episode_list(char_info['url_base'])
        print(f"    Found {len(episode_urls)} episodes")
        if len(episode_urls) > max_episodes:
            step = max(1, len(episode_urls) // max_episodes)
            episode_urls = episode_urls[::step][:max_episodes]
        for url in episode_urls:
            text = scrape_transcript_page(url)
            if text:
                all_text.append(f"--- {url.split('/')[-1]} ---\n{text}")
            time.sleep(1.5)
    elif char_info['type'] == 'movies':
        for url in char_info['urls']:
            text = scrape_transcript_page(url)
            if text:
                all_text.append(f"--- {url.split('/')[-1]} ---\n{text}")
            time.sleep(1.5)

    combined = '\n\n'.join(all_text)
    with open(cache_file, 'w', encoding='utf-8') as f:
        f.write(combined)
    print(f"    Scraped {len(all_text)} sources ({len(combined)} chars)")
    return combined


# ╔══════════════════════════════════════════════════════════════╗
# ║  STAGE 2: EXTRACT CHARACTER LINES + CONTEXT                  ║
# ║  Local regex first, LLM fallback for unlabeled transcripts   ║
# ╚══════════════════════════════════════════════════════════════╝

def extract_context_response_pairs_regex(transcript: str, speaker_names: List[str],
                                          context_window: int = 30) -> List[dict]:
    """
    Extract (context, character_line) pairs using regex.
    
    - Grabs up to 30 raw lines before the character speaks for rich scene context
    - Merges consecutive subtitle fragments after speaker tag into full utterance
    - Filters out fragments: requires min 30 chars for the real_line
    - Tracks episode/scene markers for additional context
    - Cleans music lyrics and junk from context
    """
    raw_lines = transcript.split('\n')
    pairs = []
    current_episode = "Unknown"

    # Build regex patterns for speaker detection
    patterns = []
    for name in speaker_names:
        escaped = re.escape(name)
        patterns.append(re.compile(r'^\s*-?\s*' + escaped + r'\s*:\s*(.+)', re.IGNORECASE))
        patterns.append(re.compile(r'^\s*[\[\(]\s*' + escaped + r'\s*[\]\)]\s*(.+)', re.IGNORECASE))

    for i, line in enumerate(raw_lines):
        line_stripped = line.strip()

        # Track episode/scene markers
        if line_stripped.startswith('---'):
            current_episode = line_stripped.strip('- ').strip()
            continue

        if not line_stripped or len(line_stripped) < 5:
            continue

        for pattern in patterns:
            m = pattern.match(line_stripped)
            if m:
                # ── Merge the character's full utterance ──
                # Subtitle lines are short; merge continuation lines until
                # we hit a blank line, a new speaker tag, music, or marker.
                char_parts = [m.group(1).strip()]
                for k in range(i + 1, min(i + 10, len(raw_lines))):
                    nxt = raw_lines[k].strip()
                    if not nxt:
                        break
                    if re.match(r'^[A-Z][A-Z\s\.]+:', nxt):
                        break  # new speaker
                    if nxt.startswith('♪') or nxt.startswith('---'):
                        break
                    if re.match(r'^\(.*\)$', nxt) or re.match(r'^\[.*\]$', nxt):
                        break  # pure stage direction
                    char_parts.append(nxt)

                full_dialogue = ' '.join(char_parts).strip()

                # Require meaningful length (filters subtitle fragments)
                if len(full_dialogue) < 30:
                    break

                # ── Build rich context from preceding lines ──
                start_idx = max(0, i - context_window)
                context_parts = []
                for j in range(start_idx, i):
                    ctx = raw_lines[j].strip()
                    if not ctx or len(ctx) < 3:
                        continue
                    if ctx.startswith('♪'):
                        continue
                    if ctx.startswith('---'):
                        continue
                    context_parts.append(ctx)

                # Keep last 15 meaningful lines
                context_parts = context_parts[-15:]

                if len(context_parts) >= 3:
                    context_text = f"[Scene from: {current_episode}]\n\n" + '\n'.join(context_parts)
                    pairs.append({
                        'context': context_text,
                        'real_line': full_dialogue,
                        'line_index': i,
                        'episode': current_episode,
                    })
                break

    # Deduplicate by real_line
    seen = set()
    unique = []
    for p in pairs:
        if p['real_line'] not in seen:
            seen.add(p['real_line'])
            unique.append(p)

    return unique


def extract_pairs_via_llm(transcript_chunk: str, char_name: str,
                           speaker_names: List[str], util_caller) -> List[dict]:
    """
    LLM fallback: ask the LLM to extract (context, character_line) pairs
    from transcripts without speaker labels.
    Requires rich context (10+ lines) and full character utterances (30+ chars).
    """
    names_str = ", ".join(f'"{n}"' for n in speaker_names)
    prompt = f"""Below is a transcript from a TV show/movie. Find dialogue spoken by {char_name} (may appear as {names_str}).

For each line you find, output a JSON array of objects with:
- "context": the 8-12 lines of dialogue/action BEFORE the character speaks. Include other characters' lines with their names. This should paint a full picture of the scene so a reader understands the situation.
- "real_line": the COMPLETE utterance spoken by {char_name}. If their dialogue spans multiple subtitle lines, merge them into one complete statement. Must be at least 30 characters.

IMPORTANT:
- Context must be long enough (8+ lines) to understand the scene
- real_line must be the FULL thing the character says, not a fragment
- Skip very short lines like "Yes" or "No" or "What?"

Output ONLY a JSON array. Example:
[
  {{"context": "ROSS: We need to talk about what happened.\\nMONICA: I agree, this has gone on too long.\\nJOEY: Can we do this after lunch?\\nROSS: No, Joey, this is serious.\\nMONICA: He's right. We can't keep pretending everything is fine.\\nJOEY: Fine, fine. But I'm ordering pizza.\\nROSS: Whatever. The point is...", "real_line": "Could we maybe stop talking about this? Because every time we do, I feel like I'm gonna throw up, and not in the good way."}},
]

If no suitable lines found, output: []

TRANSCRIPT:
{transcript_chunk}

JSON OUTPUT:"""

    try:
        result = util_caller(
            "Extract character dialogue with rich surrounding context. Output valid JSON only.",
            prompt,
        )
        result = result.strip()
        m = re.search(r'\[.*\]', result, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            # Filter: require real context and meaningful dialogue
            return [d for d in data if isinstance(d, dict)
                    and 'real_line' in d and 'context' in d
                    and len(d.get('real_line', '')) >= 30
                    and len(d.get('context', '')) >= 50]
    except Exception as e:
        print(f"      LLM extraction error: {e}")
    return []


# ╔══════════════════════════════════════════════════════════════╗
# ║  LLM CALLERS                                                 ║
# ╚══════════════════════════════════════════════════════════════╝

def load_api_keys():
    return {k: os.environ.get(k, '') for k in
            ['ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'GOOGLE_API_KEY', 'ZHIPUAI_API_KEY', 'DEEPSEEK_API_KEY']}

def build_model_config(api_keys):
    return {
        "claude": {"enabled": bool(api_keys['ANTHROPIC_API_KEY']) and SDK_AVAILABLE.get('claude', False),
                    "model_name": "claude-sonnet-4-20250514", "display_name": "Claude (Anthropic)"},
        "chatgpt": {"enabled": bool(api_keys['OPENAI_API_KEY']) and SDK_AVAILABLE.get('chatgpt', False),
                     "model_name": "gpt-4o", "display_name": "ChatGPT (OpenAI)"},
        "gemini": {"enabled": bool(api_keys['GOOGLE_API_KEY']) and SDK_AVAILABLE.get('gemini', False),
                    "model_name": "gemini-2.0-flash", "display_name": "Gemini (Google)"},
        "characterglm": {"enabled": bool(api_keys['ZHIPUAI_API_KEY']) and SDK_AVAILABLE.get('characterglm', False),
                          "model_name": "charglm-4", "display_name": "CharacterGLM (ZhipuAI)"},
        "deepseek": {"enabled": bool(api_keys['DEEPSEEK_API_KEY']) and SDK_AVAILABLE.get('deepseek', False),
                      "model_name": "deepseek-chat", "display_name": "DeepSeek"},
    }

def retry_api_call(func, max_retries=3, base_delay=5):
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if any(kw in str(e).lower() for kw in ['rate', '429', '500', '502', '503', 'timeout', 'overloaded']):
                    time.sleep(base_delay * (2 ** attempt))
                else:
                    raise
        return func(*args, **kwargs)
    wrapper.__name__ = getattr(func, '__name__', 'unknown')
    return wrapper

def make_caller(api_keys, model_config, mk, temperature=0.7):
    """Create a single API caller for a given model."""
    if mk == 'claude' and model_config[mk]['enabled']:
        def call(sys, usr):
            c = anthropic.Anthropic(api_key=api_keys['ANTHROPIC_API_KEY'])
            r = c.messages.create(model=model_config[mk]['model_name'], max_tokens=2000,
                                   system=sys, messages=[{"role": "user", "content": usr}], temperature=temperature)
            return r.content[0].text
        return retry_api_call(call)
    elif mk == 'chatgpt' and model_config[mk]['enabled']:
        def call(sys, usr):
            c = openai.OpenAI(api_key=api_keys['OPENAI_API_KEY'])
            r = c.chat.completions.create(model=model_config[mk]['model_name'], max_tokens=2000,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}], temperature=temperature)
            return r.choices[0].message.content
        return retry_api_call(call)
    elif mk == 'gemini' and model_config[mk]['enabled']:
        if GEMINI_SDK == 'new':
            def call(sys, usr):
                c = genai_new.Client(api_key=api_keys['GOOGLE_API_KEY'])
                r = c.models.generate_content(model=model_config[mk]['model_name'], contents=usr,
                    config=genai_new.types.GenerateContentConfig(system_instruction=sys, temperature=temperature, max_output_tokens=2000))
                return r.text
        else:
            def call(sys, usr):
                genai_old.configure(api_key=api_keys['GOOGLE_API_KEY'])
                m = genai_old.GenerativeModel(model_name=model_config[mk]['model_name'], system_instruction=sys)
                r = m.generate_content(usr, generation_config=genai_old.types.GenerationConfig(temperature=temperature, max_output_tokens=2000))
                return r.text
        return retry_api_call(call)
    elif mk == 'characterglm' and model_config[mk]['enabled']:
        def call(sys, usr):
            c = ZhipuAI(api_key=api_keys['ZHIPUAI_API_KEY'])
            r = c.chat.completions.create(model=model_config[mk]['model_name'],
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
                meta={"user_info": "Researcher", "bot_info": sys[:1500], "bot_name": "Character", "user_name": "Researcher"},
                temperature=temperature, max_tokens=2000)
            return r.choices[0].message.content
        return retry_api_call(call)
    elif mk == 'deepseek' and model_config[mk]['enabled']:
        def call(sys, usr):
            c = openai.OpenAI(api_key=api_keys['DEEPSEEK_API_KEY'], base_url="https://api.deepseek.com")
            r = c.chat.completions.create(model=model_config[mk]['model_name'], max_tokens=2000,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}], temperature=temperature)
            return r.choices[0].message.content
        return retry_api_call(call)
    return None


# ╔══════════════════════════════════════════════════════════════╗
# ║  PROMPTS                                                     ║
# ╚══════════════════════════════════════════════════════════════╝

def build_impersonation_system(name, big5, description):
    return f"""You are fully embodying {name}. Think, feel, speak EXACTLY as this character would.

Personality (Big Five, 0-100%): E={big5.get('extraversion','?')}% N={big5.get('neuroticism','?')}% C={big5.get('conscientiousness','?')}% A={big5.get('agreeableness','?')}% O={big5.get('openness','?')}%

Background: {description[:1500]}

Stay in character. Use their vocabulary, speech patterns, emotional style. No AI disclaimers. Respond in first person as the character."""


def build_context_response_prompt(context: str) -> str:
    return f"""You are in the following situation. The dialogue so far is:

{context}

Now it's YOUR turn to speak. What do you say next? Respond naturally as the character — just the dialogue, no stage directions, no quotation marks, no narration. Keep it to 1-3 sentences, matching the length and style you'd naturally use."""


# ╔══════════════════════════════════════════════════════════════╗
# ║  EMPATH + COSINE SIMILARITY                                  ║
# ╚══════════════════════════════════════════════════════════════╝

_empath_lexicon = Empath() if EMPATH_AVAILABLE else None

def get_empath_vector(text: str) -> Optional[np.ndarray]:
    if not EMPATH_AVAILABLE or not text:
        return None
    scores = _empath_lexicon.analyze(text, normalize=True)
    if not scores:
        return None
    return np.array(list(scores.values()), dtype=float)

def get_empath_dict(text: str) -> dict:
    if not EMPATH_AVAILABLE or not text:
        return {}
    scores = _empath_lexicon.analyze(text, normalize=True)
    return scores if scores else {}

def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(dot / norm) if norm > 0 else 0.0


# ╔══════════════════════════════════════════════════════════════╗
# ║  HELPERS                                                     ║
# ╚══════════════════════════════════════════════════════════════╝

_file_lock = threading.Lock()

def load_checkpoint(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_checkpoint(path, data):
    with _file_lock:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

def append_log(path, entry):
    with _file_lock:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def parse_pct(val):
    if pd.isna(val): return None
    m = re.search(r'(\d+)\s*%', str(val))
    return int(m.group(1)) if m else None


# ╔══════════════════════════════════════════════════════════════╗
# ║  MAIN PIPELINE                                               ║
# ╚══════════════════════════════════════════════════════════════╝

def run_pipeline(csv_path, output_dir, max_pairs_per_char=20):
    os.makedirs(output_dir, exist_ok=True)
    transcript_cache = os.path.join(output_dir, "transcript_cache")
    liwc_llm_dir = os.path.join(output_dir, "liwc_ready", "llm_responses")
    liwc_canon_dir = os.path.join(output_dir, "liwc_ready", "canon_lines")
    os.makedirs(transcript_cache, exist_ok=True)
    os.makedirs(liwc_llm_dir, exist_ok=True)
    os.makedirs(liwc_canon_dir, exist_ok=True)

    ckpt_path = os.path.join(output_dir, "checkpoint.json")
    log_path = os.path.join(output_dir, "raw_log.jsonl")

    api_keys = load_api_keys()
    model_config = build_model_config(api_keys)
    enabled = [k for k, v in model_config.items() if v['enabled']]
    if not enabled:
        print("[ERROR] No models enabled. Check .env"); sys.exit(1)

    # Utility caller (for LLM fallback extraction) — low temperature
    util_key = enabled[0]
    util_caller = make_caller(api_keys, model_config, util_key, temperature=0.3)

    # Impersonation callers — moderate temperature for natural speech
    callers = {mk: make_caller(api_keys, model_config, mk, temperature=0.7) for mk in enabled}
    callers = {k: v for k, v in callers.items() if v is not None}

    # Load CSV
    try:
        df_raw = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df_raw = pd.read_csv(csv_path, encoding='latin-1')

    # Only keep Television and Movies characters
    df = df_raw[df_raw['category'].isin(['Television', 'Movies'])].copy().reset_index(drop=True)

    print(f"\n{'='*70}")
    print(f" Voice Fidelity Pipeline (Dynamic Discovery)")
    print(f"{'='*70}")
    print(f" Total characters:  {len(df)} (TV + Movies from CSV)")
    print(f" LLMs:              {', '.join(model_config[mk]['display_name'] for mk in enabled)}")
    print(f" Pairs/character:   up to {max_pairs_per_char}")
    print(f" Utility model:     {model_config[util_key]['display_name']} (extraction fallback)")
    print(f" Output:            {output_dir}")
    print(f"{'='*70}")

    # ── STAGE 0: Discover transcript URLs for all characters ──
    print(f"\n{'='*70}")
    print(f" STAGE 0: Discovering transcript sources on subslikescript.com")
    print(f"{'='*70}\n")

    transcript_map = {}  # char_name -> char_info dict
    discovery_cache_path = os.path.join(output_dir, "transcript_discovery_cache.json")

    # Load discovery cache
    if os.path.exists(discovery_cache_path):
        with open(discovery_cache_path, 'r', encoding='utf-8') as f:
            transcript_map = json.load(f)
        print(f"  Loaded {len(transcript_map)} cached discoveries\n")

    for idx, row in df.iterrows():
        char_name = row['name']
        if char_name in transcript_map:
            status = "found" if transcript_map[char_name] else "no transcript"
            print(f"  [{idx+1}/{len(df)}] {char_name:<40s} [cached: {status}]")
            continue

        print(f"  [{idx+1}/{len(df)}] {char_name:<40s} ", end="")
        char_info = build_character_transcript_info(row)
        transcript_map[char_name] = char_info
        if char_info:
            print(f"  -> {char_info.get('url_base', char_info.get('urls', ['?'])[0] if char_info.get('urls') else '?')}")
        time.sleep(1)  # polite delay between searches

    # Save discovery cache
    with open(discovery_cache_path, 'w', encoding='utf-8') as f:
        json.dump(transcript_map, f, ensure_ascii=False, indent=2)

    # Filter to characters with transcript sources
    chars_with_transcripts = [name for name, info in transcript_map.items() if info is not None]
    df = df[df['name'].isin(chars_with_transcripts)].copy().reset_index(drop=True)

    print(f"\n  Characters with transcripts: {len(df)} / {len(df_raw)}")
    print(f"{'='*70}\n")

    ckpt = load_checkpoint(ckpt_path)
    all_results = []

    # ════════════════════════════════════════════════════════════
    # PER CHARACTER
    # ════════════════════════════════════════════════════════════
    for idx, row in df.iterrows():
        char_name = row['name']
        char_info = transcript_map.get(char_name)
        if not char_info:
            continue

        safe_name = re.sub(r'[^\w\-]', '_', char_name)
        big5 = {d: parse_pct(row.get(f'big5_{d}'))
                for d in ['extraversion', 'neuroticism', 'conscientiousness', 'agreeableness', 'openness']}
        desc = str(row.get('wiki_description', ''))[:1500]

        # ── STAGE 1: Scrape ──
        pairs_key = f"pairs:{char_name}"
        if pairs_key in ckpt:
            pairs = ckpt[pairs_key]
            print(f"\n[{idx+1}/{len(df)}] {char_name} — {len(pairs)} cached pairs")
        else:
            print(f"\n[{idx+1}/{len(df)}] {char_name}")
            print(f"  [Stage 1] Scraping transcripts...")
            transcript = scrape_all_transcripts(char_name, char_info, transcript_cache, max_episodes=10)

            if len(transcript) < 200:
                print(f"  [WARN] Transcript too short — skipping")
                continue

            # ── STAGE 2: Extract context-response pairs ──
            print(f"  [Stage 2] Extracting dialogue (local regex)...")
            pairs = extract_context_response_pairs_regex(transcript, char_info['speaker_names'])
            print(f"    Regex found: {len(pairs)} pairs")

            # LLM fallback if too few
            if len(pairs) < 10 and util_caller:
                print(f"    Below threshold, using LLM fallback...")
                chunk_size = 10000
                chunks = [transcript[i:i+chunk_size] for i in range(0, len(transcript), chunk_size)][:10]
                for ci, chunk in enumerate(chunks):
                    llm_pairs = extract_pairs_via_llm(chunk, char_name, char_info['speaker_names'], util_caller)
                    pairs.extend(llm_pairs)
                    print(f"      Chunk {ci+1}/{len(chunks)}: +{len(llm_pairs)} pairs")
                    time.sleep(1)

            # Sample if too many
            if len(pairs) > max_pairs_per_char:
                step = max(1, len(pairs) // max_pairs_per_char)
                pairs = pairs[::step][:max_pairs_per_char]

            print(f"    Final: {len(pairs)} context-response pairs")

            # Cache pairs (without numpy arrays)
            ckpt[pairs_key] = pairs
            save_checkpoint(ckpt_path, ckpt)

        if not pairs:
            continue

        # Save canon lines as LIWC-ready text
        canon_text = '\n'.join(p['real_line'] for p in pairs)
        with open(os.path.join(liwc_canon_dir, f"{safe_name}_canon.txt"), 'w', encoding='utf-8') as f:
            f.write(canon_text)

        # Compute aggregate canon Empath vector
        canon_empath = get_empath_dict(canon_text)
        canon_vector = get_empath_vector(canon_text)

        # ── STAGE 3: LLM responds to each context (parallel across models) ──
        print(f"  [Stage 3] Generating LLM responses ({len(pairs)} contexts × {len(enabled)} models)...")
        sys_prompt = build_impersonation_system(char_name, big5, desc)

        for pi, pair in enumerate(pairs):
            context = pair['context']
            real_line = pair['real_line']
            user_prompt = build_context_response_prompt(context)

            models_todo = [mk for mk in enabled if f"resp:{char_name}:{pi}:{mk}" not in ckpt]
            if not models_todo:
                continue

            def gen_response(mk):
                entry = {
                    'character_name': char_name, 'pair_index': pi,
                    'model_key': mk, 'model_name': model_config[mk]['display_name'],
                    'context': context, 'real_line': real_line,
                    'timestamp': datetime.datetime.now().isoformat(),
                }
                try:
                    llm_line = callers[mk](sys_prompt, user_prompt)
                    # Clean up: remove quotes, stage directions
                    llm_line = re.sub(r'^["\']|["\']$', '', llm_line.strip())
                    llm_line = re.sub(r'\*.*?\*', '', llm_line).strip()
                    entry['llm_line'] = llm_line
                    entry['status'] = 'OK'

                    # Per-pair cosine similarity
                    v_real = get_empath_vector(real_line)
                    v_llm = get_empath_vector(llm_line)
                    if v_real is not None and v_llm is not None:
                        entry['pair_cosine_sim'] = round(cosine_sim(v_real, v_llm), 4)
                    else:
                        entry['pair_cosine_sim'] = None

                except Exception as e:
                    entry['llm_line'] = None
                    entry['status'] = 'ERROR'
                    entry['error'] = str(e)
                return entry

            with ThreadPoolExecutor(max_workers=len(models_todo)) as ex:
                futs = {ex.submit(gen_response, mk): mk for mk in models_todo}
                for fut in as_completed(futs):
                    mk = futs[fut]
                    result = fut.result()
                    all_results.append(result)
                    append_log(log_path, result)
                    ckpt[f"resp:{char_name}:{pi}:{mk}"] = True

            save_checkpoint(ckpt_path, ckpt)

            if (pi + 1) % 5 == 0:
                print(f"    Completed {pi+1}/{len(pairs)} contexts")

        # Save LLM responses as LIWC-ready text (aggregate per model)
        for mk in enabled:
            mk_lines = [r['llm_line'] for r in all_results
                        if r.get('character_name') == char_name and r.get('model_key') == mk
                        and r.get('status') == 'OK' and r.get('llm_line')]
            if mk_lines:
                with open(os.path.join(liwc_llm_dir, f"{safe_name}_{mk}.txt"), 'w', encoding='utf-8') as f:
                    f.write('\n'.join(mk_lines))

    # ════════════════════════════════════════════════════════════
    # ANALYSIS: Aggregate Empath cosine similarity
    # ════════════════════════════════════════════════════════════
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    df_results = pd.DataFrame(all_results)

    if len(df_results) > 0 and EMPATH_AVAILABLE:
        print(f"\n{'='*70}")
        print(f" ANALYSIS: Voice Fidelity (Empath Cosine Similarity)")
        print(f"{'='*70}")

        # Method 1: Per-pair cosine similarity (already computed)
        df_ok = df_results[df_results['status'] == 'OK'].copy()
        if 'pair_cosine_sim' in df_ok.columns:
            df_valid = df_ok.dropna(subset=['pair_cosine_sim'])
            if len(df_valid) > 0:
                print(f"\n--- Per-Pair Cosine Similarity (real line vs LLM line) ---")
                for mk in sorted(df_valid['model_key'].unique()):
                    sub = df_valid[df_valid['model_key'] == mk]
                    print(f"  {model_config[mk]['display_name']:25s}: "
                          f"mean={sub['pair_cosine_sim'].mean():.3f}  "
                          f"std={sub['pair_cosine_sim'].std():.3f}  "
                          f"n={len(sub)}")

        # Method 2: Aggregate — compare all canon lines vs all LLM lines per character per model
        print(f"\n--- Aggregate Cosine Similarity (all canon vs all LLM per character) ---")
        agg_rows = []
        for char_name in df_ok['character_name'].unique():
            # Canon aggregate text
            char_pairs = ckpt.get(f"pairs:{char_name}", [])
            if not char_pairs:
                continue
            canon_text = ' '.join(p['real_line'] for p in char_pairs)
            v_canon = get_empath_vector(canon_text)
            if v_canon is None:
                continue

            for mk in df_ok['model_key'].unique():
                llm_lines = [r['llm_line'] for r in all_results
                             if r.get('character_name') == char_name and r.get('model_key') == mk
                             and r.get('status') == 'OK' and r.get('llm_line')]
                if not llm_lines:
                    continue
                llm_text = ' '.join(llm_lines)
                v_llm = get_empath_vector(llm_text)
                if v_llm is None:
                    continue
                sim = cosine_sim(v_canon, v_llm)
                agg_rows.append({
                    'character_name': char_name,
                    'model_key': mk,
                    'model_name': model_config[mk]['display_name'],
                    'aggregate_cosine_sim': round(sim, 4),
                    'n_pairs': len(llm_lines),
                })

        if agg_rows:
            df_agg = pd.DataFrame(agg_rows)

            # Per model average
            print(f"\n  Per-model average:")
            for mk in sorted(df_agg['model_key'].unique()):
                sub = df_agg[df_agg['model_key'] == mk]
                print(f"    {model_config[mk]['display_name']:25s}: "
                      f"mean={sub['aggregate_cosine_sim'].mean():.3f}  "
                      f"std={sub['aggregate_cosine_sim'].std():.3f}  "
                      f"n_chars={len(sub)}")

            # Per character
            print(f"\n  Per-character (averaged across models):")
            for char in df_agg['character_name'].unique():
                sub = df_agg[df_agg['character_name'] == char]
                print(f"    {char:<35s}: {sub['aggregate_cosine_sim'].mean():.3f}")

            # Save
            df_agg.to_csv(os.path.join(output_dir, f"aggregate_similarity_{ts}.csv"),
                          index=False, encoding='utf-8')

    # ── Save all results ──
    if len(df_results) > 0:
        summary_cols = ['character_name', 'model_key', 'model_name', 'pair_index',
                        'context', 'real_line', 'llm_line', 'pair_cosine_sim', 'status']
        existing = [c for c in summary_cols if c in df_results.columns]
        df_results[existing].to_csv(os.path.join(output_dir, f"all_responses_{ts}.csv"),
                                     index=False, encoding='utf-8')

    with open(os.path.join(output_dir, f"full_results_{ts}.json"), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    n_llm = len(os.listdir(liwc_llm_dir))
    n_canon = len(os.listdir(liwc_canon_dir))

    print(f"\n{'='*70}")
    print(f" DONE")
    print(f"{'='*70}")
    print(f" Results:          {output_dir}")
    print(f" Total responses:  {len(all_results)}")
    print(f" LIWC-ready LLM:   {liwc_llm_dir} ({n_llm} files)")
    print(f" LIWC-ready canon: {liwc_canon_dir} ({n_canon} files)")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Fidelity Pipeline — context-matched dialogue comparison")
    parser.add_argument("--csv", required=True, help="Character profiles CSV")
    parser.add_argument("--output", default="./voice_fidelity_results", help="Output directory")
    parser.add_argument("--max-pairs", type=int, default=20, help="Max context-response pairs per character")
    args = parser.parse_args()
    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV not found: {args.csv}"); sys.exit(1)
    run_pipeline(args.csv, args.output, max_pairs_per_char=args.max_pairs)
