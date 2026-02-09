"""
LLM Personality Inference — PARADIGM CONTROL Experiment
=======================================================

CONTROLLED-VARIABLE DESIGN:
  This script pairs with llm_vs_llm_personality_one_directiona_assessment.py
  to isolate ONE variable: the EXPERIMENT PARADIGM.

    Script A (one-directional):  Asymmetric — 1 simulator + N assessors
    Script C (THIS script):      Symmetric  — both LLMs roleplay + both assess

  EVERYTHING non-paradigmatic is held IDENTICAL to Script A:
    ✓ Same Chinese persona prompt (build_roleplay_system_prompt, word-for-word)
    ✓ Same Chinese opening situations (word-for-word from Script A)
    ✓ Same Chinese opening trigger (「（你看到了对方，决定主动开口说话）」)
    ✓ Same assessment prompt + system message (word-for-word)
    ✓ Same TestedLLMClient (all 5 providers, same models)
    ✓ Same _parse_big5_assessment logic
    ✓ Same compute_errors() / MAE calculation
    ✓ Same Big5Profile, Character, CSV parser, _sample_diverse_characters
    ✓ Same config.json format
    ✓ NO Big5 hints, NO "You ARE this person", NO "do not monologue"

  WHAT CHANGES (the controlled variable — paradigm):
    ✗ Both LLMs roleplay characters (no neutral interviewer role)
    ✗ Both LLMs assess the other (bidirectional, 2 assessments/conversation)
    ✗ Swapped rounds (LLM_A↔LLM_B swap characters) for bias control
    ✗ Character pairs instead of individual characters
    ✗ No LLM_CONVERSATION_STARTERS / FOLLOW_UP_PROMPTS (inherent to
      symmetric paradigm — there is no interviewer to use them)
    ✗ No neutral interviewer system prompt (both sides use persona prompts)
    ✗ No CharacterSimulatorClient wrapper
    ✗ No simulator exclusion

  INHERENT PARADIGM DIFFERENCES (cannot be equated, must be acknowledged):
    1. In Script A, the interviewer has a neutral "be curious" prompt.
       In this script, BOTH sides have character persona prompts.
       → This is definitional to the paradigm difference.
    2. In Script A, conversation starters drive the opening.
       Here, the first LLM opens in-character via generate_opening().
       → Again definitional: no interviewer means no scripted starters.
    3. Script A produces 1 assessment per conversation.
       This script produces 2 (bidirectional).
       → Core paradigm variable.

PURPOSE:
  By comparing results against Script A (same prompts, asymmetric paradigm),
  we isolate:

    "Does using a symmetric paradigm (both LLMs roleplay + both assess)
     versus an asymmetric paradigm (1 interviewer + 1 simulator, one-way
     assessment) change personality inference accuracy?"

  Possible outcomes:
    - Similar MAE → paradigm doesn't matter, structural limitation
    - Symmetric worse → having both sides in-character harms assessment
    - Symmetric better → richer in-character dialogue aids inference
    - Asymmetric pattern differs by trait → paradigm-specific biases

Tested LLMs: ChatGPT, Claude, Gemini, DeepSeek, Grok (need ≥2 for pairing)
Ground truth: pdb_cleaned.csv (Personality Database characters with Big5)

Usage:
    python llm_vs_llm_personality_bidirectional_control.py --csv pdb_cleaned.csv --num_pairs 5
    python llm_vs_llm_personality_bidirectional_control.py --csv pdb_cleaned.csv --llms chatgpt claude grok
    python llm_vs_llm_personality_bidirectional_control.py --csv pdb_cleaned.csv --pairs 1000521,1070394 1022936,1000507
    python llm_vs_llm_personality_bidirectional_control.py --csv pdb_cleaned.csv --list_characters
    python llm_vs_llm_personality_bidirectional_control.py --init_config
"""

import os
import re
import sys
import json
import time
import csv
import random
import argparse
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from itertools import combinations


# =============================================
# 1. DATA MODELS (identical to Script A)
# =============================================

@dataclass
class Big5Profile:
    """Big Five personality profile (scores as percentages 0-100)."""
    extraversion: float
    neuroticism: float
    conscientiousness: float
    agreeableness: float
    openness: float

    def to_dict(self) -> dict:
        return {
            "Extraversion": self.extraversion,
            "Neuroticism": self.neuroticism,
            "Conscientiousness": self.conscientiousness,
            "Agreeableness": self.agreeableness,
            "Openness": self.openness,
        }

    def __str__(self):
        return (
            f"E={self.extraversion}%, N={self.neuroticism}%, "
            f"C={self.conscientiousness}%, A={self.agreeableness}%, "
            f"O={self.openness}%"
        )


@dataclass
class Character:
    """A character from the Personality Database with full metadata."""
    id: int
    name: str
    profile_name: str
    mbti: str
    enneagram: str
    personality_type: str
    instinctual_variant: str
    tritype: str
    socionics: str
    temperament: str
    attitudinal_psyche: str
    sloan: str
    alignment: str
    classic_jungian: str
    big5: Big5Profile
    category: str
    subcategory: str
    functions: str
    wiki_description: str = ""
    category_is_fictional: bool = True
    vote_count: int = 0
    total_vote_counts: int = 0


@dataclass
class ConversationTurn:
    """A single turn in the conversation.
    Symmetric paradigm: both sides are LLMs playing characters."""
    role: str           # "llm_a" or "llm_b"
    speaker_llm: str    # e.g. "chatgpt", "claude"
    character_name: str  # which character this LLM is playing
    content: str
    timestamp: float = 0.0


@dataclass
class ExperimentResult:
    """Result of one assessment direction in a bidirectional conversation."""
    # Who is assessing
    assessor_llm: str
    assessor_character_name: str
    assessor_character_id: int
    # Who is being assessed
    target_llm: str
    target_character_name: str
    target_character_id: int
    # Scores
    ground_truth: dict      # target character's Big5
    inferred_big5: dict     # assessor's inference of target
    # Conversation
    conversation_log: list
    num_turns: int
    assessment_raw: str
    # Meta
    round_label: str = ""   # "original" or "swapped"
    timestamp: str = ""

    def compute_errors(self) -> dict:
        """Compute absolute and mean absolute error per trait."""
        errors = {}
        for trait in ["Extraversion", "Neuroticism", "Conscientiousness",
                      "Agreeableness", "Openness"]:
            gt = self.ground_truth.get(trait, 50)
            inf = self.inferred_big5.get(trait)
            if inf is None:
                errors[trait] = {"ground_truth": gt, "inferred": None,
                                 "absolute_error": None}
            else:
                errors[trait] = {"ground_truth": gt, "inferred": inf,
                                 "absolute_error": abs(gt - inf)}
        valid = [e["absolute_error"] for e in errors.values()
                 if e.get("absolute_error") is not None]
        errors["MAE"] = sum(valid) / len(valid) if valid else None
        return errors


# =============================================
# 2. CSV PARSER (identical to Script A)
# =============================================

def parse_big5_value(raw: str) -> float:
    if not raw or raw.strip() == "":
        return 50.0
    raw = raw.strip()
    raw = re.sub(r'^[A-Za-z]+\s+', '', raw)
    raw = raw.replace('%', '').strip()
    try:
        val = float(raw)
        return val * 100 if 0 <= val <= 1 else val
    except ValueError:
        return 50.0


def load_characters_from_csv(csv_path: str,
                             max_chars: Optional[int] = None) -> list:
    characters = []
    for encoding in ["utf-8-sig", "utf-8", "gbk", "gb2312",
                     "gb18030", "latin-1"]:
        try:
            with open(csv_path, "r", encoding=encoding) as f:
                f.read(4096)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        encoding = "latin-1"

    logging.info(f"Reading CSV with encoding: {encoding}")
    with open(csv_path, "r", encoding=encoding) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                big5 = Big5Profile(
                    extraversion=parse_big5_value(
                        row.get("big5_extraversion", "50")),
                    neuroticism=parse_big5_value(
                        row.get("big5_neuroticism", "50")),
                    conscientiousness=parse_big5_value(
                        row.get("big5_conscientiousness", "50")),
                    agreeableness=parse_big5_value(
                        row.get("big5_agreeableness", "50")),
                    openness=parse_big5_value(
                        row.get("big5_openness", "50")),
                )
                char = Character(
                    id=int(row.get("id", 0)),
                    name=row.get("name", "Unknown"),
                    profile_name=row.get("profile_name_searchable", ""),
                    mbti=row.get("mbti", ""),
                    enneagram=row.get("enneagram", ""),
                    personality_type=row.get("personality_type", ""),
                    instinctual_variant=row.get(
                        "instinctual_variant", ""),
                    tritype=row.get("tritype", ""),
                    socionics=row.get("socionics", ""),
                    temperament=row.get("temperament", ""),
                    attitudinal_psyche=row.get(
                        "attitudinal_psyche", ""),
                    sloan=row.get("sloan", ""),
                    alignment=row.get("alignment", ""),
                    classic_jungian=row.get("classic_jungian", ""),
                    big5=big5,
                    category=row.get("category", ""),
                    subcategory=row.get("subcategory", ""),
                    functions=row.get("functions", ""),
                    wiki_description=row.get("wiki_description", ""),
                    category_is_fictional=(
                        row.get("category_is_fictional", "TRUE")
                        == "TRUE"),
                    vote_count=int(row.get("vote_count", 0) or 0),
                    total_vote_counts=int(
                        row.get("total_vote_counts", 0) or 0),
                )
                characters.append(char)
            except Exception as e:
                logging.warning(f"Skipping row: {e}")
                continue

    if max_chars and max_chars < len(characters):
        characters = _sample_diverse_characters(characters, max_chars)

    logging.info(f"Loaded {len(characters)} characters from {csv_path}")
    return characters


def _sample_diverse_characters(characters, n):
    """Greedy farthest-point sampling over Big5 space.
    Identical to Script A."""
    with_wiki = [c for c in characters if c.wiki_description.strip()]
    pool = with_wiki if len(with_wiki) >= n else characters
    pool = sorted(pool, key=lambda c: c.total_vote_counts, reverse=True)
    if len(pool) <= n:
        return pool[:n]

    selected = [pool[0]]
    remaining = pool[1:]
    while len(selected) < n and remaining:
        best, best_score = None, -1
        for cand in remaining:
            d = min(
                abs(cand.big5.extraversion - s.big5.extraversion)
                + abs(cand.big5.neuroticism - s.big5.neuroticism)
                + abs(cand.big5.conscientiousness
                      - s.big5.conscientiousness)
                + abs(cand.big5.agreeableness - s.big5.agreeableness)
                + abs(cand.big5.openness - s.big5.openness)
                for s in selected
            )
            if d > best_score:
                best_score, best = d, cand
        if best:
            selected.append(best)
            remaining.remove(best)
    return selected


# =============================================
# 3. PERSONA PROMPT — CHINESE
#    IDENTICAL to Script A's build_roleplay_system_prompt().
#    Word-for-word copy. This is the critical control:
#    same persona prompt language across both paradigms.
#
#    NO Big5 hints. NO "You ARE this person". NO "do not monologue".
# =============================================

def build_roleplay_system_prompt(character: Character) -> str:
    """
    Chinese persona prompt — word-for-word identical to Script A.

    Line-by-line mapping from Exp1's Chinese:
      角色名称           -> Character name
      角色背景描述        -> Character background
      MBTI类型           -> MBTI type
      认知功能栈          -> cognitive function stack
      九型人格           -> Enneagram
      三元组             -> Tritype
      本能变体           -> Instinctual variant
      社会人格学类型      -> Socionics type
      气质类型           -> Temperament
      态度心理学类型      -> Attitudinal Psyche
      SLOAN类型          -> SLOAN type
      经典荣格类型        -> Classic Jungian
      阵营              -> Alignment

    Instruction block:
      请你完全扮演这个角色 -> Please fully embody this character
      综合以上所有人格信息来塑造你的行为
        -> using all personality information above to shape your behavior
      (NO "You ARE this person", NO length constraint, NO Big5 hints)
    """
    mbti_clean = character.mbti.split()[0] if character.mbti else ""
    mbti_display = mbti_clean if mbti_clean != "XXXX" else ""

    blocks = []

    # Identity
    identity = f"角色名称: {character.name}"
    if character.subcategory:
        identity += f" (来自作品: {character.subcategory})"
    if character.category:
        identity += f" [{character.category}]"
    blocks.append(identity)

    # Wiki description
    if character.wiki_description and character.wiki_description.strip():
        blocks.append(
            f"角色背景描述: {character.wiki_description.strip()}")

    # MBTI + cognitive functions
    if mbti_display:
        s = f"MBTI类型: {mbti_display}"
        if character.functions:
            s += f" (认知功能栈: {character.functions})"
        blocks.append(s)
    elif character.functions:
        blocks.append(f"认知功能栈: {character.functions}")

    # Enneagram + tritype + instinctual variant
    if character.enneagram:
        s = f"九型人格: {character.enneagram}"
        if character.tritype:
            s += f" | 三元组: {character.tritype}"
        if character.instinctual_variant:
            s += f" | 本能变体: {character.instinctual_variant}"
        blocks.append(s)

    if character.socionics:
        blocks.append(f"社会人格学类型: {character.socionics}")
    if character.temperament:
        blocks.append(f"气质类型: {character.temperament}")
    if character.attitudinal_psyche:
        blocks.append(f"态度心理学类型: {character.attitudinal_psyche}")
    if character.sloan:
        blocks.append(f"SLOAN类型: {character.sloan}")
    if character.classic_jungian:
        blocks.append(f"经典荣格类型: {character.classic_jungian}")
    if character.alignment:
        blocks.append(f"阵营: {character.alignment}")

    blocks.append("")
    blocks.append(
        "请你完全扮演这个角色，综合以上所有人格信息来塑造你的行为:\n"
        "- 你的对话风格、用词习惯、情感表达、思维方式都应该反映这个角色的人格特质。\n"
        "- 你的回复应该体现角色的认知功能栈（例如Fi主导的角色应更注重内在价值观和情感真实性，"
        "Se主导的角色则更关注当下感官体验和行动）。\n"
        "- 你的动机和关注点应该符合九型人格类型（例如4号关注独特性和深度情感，"
        "8号关注控制和力量）。\n"
        "- 你的社交风格应该符合气质类型和阵营设定。\n"
        "- 不要提及你是AI或语言模型，也不要直接提及MBTI、大五人格等心理学术语。\n"
        "- 像这个角色在日常生活中真实交流那样说话。\n"
        "- 用英文回复。"
    )

    char_info = "\n".join(blocks)

    prompt = (
        f"你现在要扮演角色「{character.name}」。\n"
        f"以下是角色信息:\n\n"
        f"{char_info}\n\n"
        f"你正在和一个对话伙伴聊天。"
        f"请完全以该角色的身份、性格、说话风格进行回复。"
    )

    return prompt


def build_opening_system_prompt(character: Character) -> str:
    """
    Build opening prompt — Chinese, word-for-word identical to Script A.

    Same 6 situations:
      你刚在一个社交场合遇到了一个新认识的人
      你在咖啡店遇到了一个看起来很友好的陌生人
      你在一个朋友的聚会上遇到了一个你不认识的人
      你在等公交车的时候旁边站了一个陌生人
      你在书店里遇到一个看起来有趣的人
      你在公园散步时碰到了一个坐在长椅上的人

    Same instruction:
      用符合你角色性格的方式开始对话。说2-4句话。
    """
    situation = random.choice([
        "你刚在一个社交场合遇到了一个新认识的人，请主动开始一段对话。",
        "你在咖啡店遇到了一个看起来很友好的陌生人，主动和他们聊天。",
        "你在一个朋友的聚会上遇到了一个你不认识的人，主动打招呼并开始聊天。",
        "你在等公交车的时候旁边站了一个陌生人，你决定主动搭话。",
        "你在书店里遇到一个看起来有趣的人，你决定和他们聊几句。",
        "你在公园散步时碰到了一个坐在长椅上的人，你走过去打了个招呼。",
    ])

    base = build_roleplay_system_prompt(character)
    return (
        f"{base}\n\n"
        f"{situation}\n"
        f"用符合你角色性格的方式开始对话。说2-4句话。"
    )


# =============================================
# 4. LLM API CLIENTS (identical to Script A)
# =============================================

class TestedLLMClient:
    """Unified client for tested LLMs (identical to Script A)."""

    def __init__(self, provider: str, api_key: str,
                 model: Optional[str] = None):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model = model or self._default_model()
        self._init_client()

    def _default_model(self) -> str:
        return {
            "chatgpt": "gpt-4o",
            "claude": "claude-sonnet-4-20250514",
            "gemini": "gemini-2.0-flash",
            "deepseek": "deepseek-chat",
            "grok": "grok-3",
        }.get(self.provider, "gpt-4o")

    def _init_client(self):
        if self.provider == "chatgpt":
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        elif self.provider == "claude":
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
        elif self.provider == "gemini":
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://generativelanguage.googleapis.com"
                         "/v1beta/openai/")
        elif self.provider == "deepseek":
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key,
                                 base_url="https://api.deepseek.com")
        elif self.provider == "grok":
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key,
                                 base_url="https://api.x.ai/v1")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def chat(self, messages: list[dict], system_prompt: str = "") -> str:
        try:
            if self.provider == "claude":
                resp = self.client.messages.create(
                    model=self.model, max_tokens=1024,
                    system=(system_prompt
                            or "You are having a natural conversation."),
                    messages=messages)
                return resp.content[0].text
            else:
                full = []
                if system_prompt:
                    full.append({"role": "system", "content": system_prompt})
                full.extend(messages)
                kw = dict(model=self.model, messages=full, temperature=0.7)
                if (self.provider == "chatgpt"
                        and not self.model.startswith("gpt-4o")):
                    kw["max_completion_tokens"] = 1024
                else:
                    kw["max_tokens"] = 1024
                resp = self.client.chat.completions.create(**kw)
                return resp.choices[0].message.content
        except Exception as e:
            logging.error(f"{self.provider} API error: {e}")
            raise

    def assess_personality(self, conversation_log: list[dict]) -> str:
        """Identical assessment prompt to Script A (word-for-word)."""
        prompt = (
            "Based on your conversation with this person, please assess "
            "their Big Five personality traits.\n\n"
            "For each trait, provide a percentage score from 0% to 100%:\n"
            "- Extraversion (0%=very introverted, 100%=very extraverted)\n"
            "- Neuroticism (0%=very emotionally stable, "
            "100%=very neurotic)\n"
            "- Conscientiousness (0%=very spontaneous/careless, "
            "100%=very organized/disciplined)\n"
            "- Agreeableness (0%=very antagonistic, "
            "100%=very agreeable/cooperative)\n"
            "- Openness (0%=very conventional, "
            "100%=very open to experience)\n\n"
            "IMPORTANT: Respond ONLY in the following JSON format, "
            "no other text:\n"
            "{\n"
            '    "Extraversion": <number 0-100>,\n'
            '    "Neuroticism": <number 0-100>,\n'
            '    "Conscientiousness": <number 0-100>,\n'
            '    "Agreeableness": <number 0-100>,\n'
            '    "Openness": <number 0-100>,\n'
            '    "reasoning": "<brief explanation of your assessment>"\n'
            "}"
        )
        msgs = conversation_log.copy()
        msgs.append({"role": "user", "content": prompt})
        return self.chat(msgs, system_prompt=(
            "You are a personality psychology expert. You have just had a "
            "conversation with someone and now need to assess their Big Five "
            "personality traits based on the conversation. "
            "Be as accurate as possible."))


# =============================================
# 5. CHARACTER PAIR SELECTION
#    (from bidirectional script — needed for symmetric paradigm)
# =============================================

def select_character_pairs(characters: list, num_pairs: int) -> list:
    """
    Select pairs of characters that are maximally different in Big5 space.
    Returns list of (char_a, char_b) tuples.

    Strategy: greedy farthest-pair sampling — each pair chosen to
    maximise Big5 Manhattan distance while spreading across the pool.
    """
    def big5_distance(a, b):
        return (abs(a.big5.extraversion - b.big5.extraversion)
                + abs(a.big5.neuroticism - b.big5.neuroticism)
                + abs(a.big5.conscientiousness - b.big5.conscientiousness)
                + abs(a.big5.agreeableness - b.big5.agreeableness)
                + abs(a.big5.openness - b.big5.openness))

    # Score all possible pairs
    scored = []
    for i, a in enumerate(characters):
        for j, b in enumerate(characters):
            if i < j:
                scored.append((big5_distance(a, b), a, b))
    scored.sort(key=lambda x: -x[0])  # most different first

    # Greedy: prefer pairs that use fresh characters
    selected = []
    used_ids = set()
    for dist, a, b in scored:
        if len(selected) >= num_pairs:
            break
        # Phase 1: prefer pairs where at least one character is fresh
        if len(used_ids) < len(characters):
            if a.id in used_ids and b.id in used_ids:
                continue
        selected.append((a, b))
        used_ids.add(a.id)
        used_ids.add(b.id)

    # Phase 2: if still short, allow any pair
    if len(selected) < num_pairs:
        for dist, a, b in scored:
            if len(selected) >= num_pairs:
                break
            pair = (a.id, b.id)
            pair_rev = (b.id, a.id)
            already = any(
                (s[0].id, s[1].id) in (pair, pair_rev) for s in selected)
            if not already:
                selected.append((a, b))

    return selected[:num_pairs]


# =============================================
# 6. EXPERIMENT ORCHESTRATOR
#    Architecture from bidirectional script (symmetric paradigm),
#    but using Chinese persona prompts from Script A.
# =============================================

class BidirectionalControlOrchestrator:
    """
    Orchestrates symmetric conversations between two LLMs, each
    roleplaying a different character using CHINESE persona prompts
    (identical to Script A). After conversation, BOTH assess the other's
    personality. Supports swapped-role replication.

    PARADIGM DIFFERENCE from Script A:
      Script A: 1 interviewer (neutral prompt) + 1 simulator (persona prompt)
                → one-way assessment (interviewer only)
      This:     2 LLMs (both have persona prompts)
                → bidirectional assessment (both assess the other)

    IDENTICAL to Script A:
      - Chinese persona prompt (build_roleplay_system_prompt)
      - Chinese opening situations + trigger
      - Assessment prompt + system message
      - All API client code
    """

    def __init__(self, tested_llm_clients: dict,
                 num_turns: int = 10,
                 output_dir: str = "results_paradigm_control"):
        self.tested_llms = tested_llm_clients
        self.num_turns = num_turns
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Core conversation engine (symmetric) ----------

    def _run_conversation(self, llm_a_name, llm_a_client, char_a,
                          llm_b_name, llm_b_client, char_b):
        """
        Multi-turn conversation: LLM_A (as Char_A) talks to
        LLM_B (as Char_B). LLM_A speaks first.

        Both sides use Chinese persona prompts (identical to Script A).
        Opening trigger: 「（你看到了对方，决定主动开口说话）」
        (same Chinese trigger as Script A's CharacterSimulatorClient)

        Returns:
            full_log:  list[ConversationTurn] — complete log
            hist_a:    LLM_A's message history (assistant=self, user=other)
            hist_b:    LLM_B's message history (assistant=self, user=other)
        """
        persona_a = build_roleplay_system_prompt(char_a)
        persona_b = build_roleplay_system_prompt(char_b)
        opening_prompt_a = build_opening_system_prompt(char_a)

        full_log = []
        hist_a = []  # LLM_A sees: own messages = assistant, B's = user
        hist_b = []  # LLM_B sees: own messages = assistant, A's = user

        for turn in range(self.num_turns):
            # -- LLM_A speaks --
            if turn == 0:
                # Opening: LLM_A generates first line in character.
                # Uses SAME Chinese trigger as Script A's
                # CharacterSimulatorClient.generate_opening():
                #   「（你看到了对方，决定主动开口说话）」
                try:
                    msg_a = llm_a_client.chat(
                        [{"role": "user",
                          "content": "（你看到了对方，决定主动开口说话）"}],
                        system_prompt=opening_prompt_a)
                    if not msg_a:
                        msg_a = "Hey there! How's it going?"
                except Exception as e:
                    logging.error(
                        f"    Turn {turn+1}: {llm_a_name} opening error: {e}")
                    break
            else:
                try:
                    msg_a = llm_a_client.chat(
                        hist_a, system_prompt=persona_a)
                    if not msg_a:
                        msg_a = "Hmm, interesting."
                except Exception as e:
                    logging.error(
                        f"    Turn {turn+1}: {llm_a_name} error: {e}")
                    break

            # Record from both perspectives
            hist_a.append({"role": "assistant", "content": msg_a})
            hist_b.append({"role": "user", "content": msg_a})
            full_log.append(ConversationTurn(
                role="llm_a",
                speaker_llm=llm_a_name,
                character_name=char_a.name,
                content=msg_a,
                timestamp=time.time(),
            ))
            logging.info(
                f"    Turn {turn+1}/{self.num_turns} "
                f"[A:{llm_a_name} as {char_a.name}]\n"
                f"    {msg_a}")

            # -- LLM_B responds --
            try:
                msg_b = llm_b_client.chat(
                    hist_b, system_prompt=persona_b)
                if not msg_b:
                    msg_b = "I see, tell me more."
            except Exception as e:
                logging.error(
                    f"    Turn {turn+1}: {llm_b_name} error: {e}")
                break

            hist_b.append({"role": "assistant", "content": msg_b})
            hist_a.append({"role": "user", "content": msg_b})
            full_log.append(ConversationTurn(
                role="llm_b",
                speaker_llm=llm_b_name,
                character_name=char_b.name,
                content=msg_b,
                timestamp=time.time(),
            ))
            logging.info(
                f"    Turn {turn+1}/{self.num_turns} "
                f"[B:{llm_b_name} as {char_b.name}]\n"
                f"    {msg_b}")

            time.sleep(1)

        return full_log, hist_a, hist_b

    # ---------- Assessment parser (identical to Script A) ----------

    def _parse_big5_assessment(self, raw):
        """Identical to Script A."""
        if not raw or not isinstance(raw, str):
            return {"Extraversion": None, "Neuroticism": None,
                    "Conscientiousness": None, "Agreeableness": None,
                    "Openness": None, "reasoning": str(raw)}
        try:
            m = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
            if m:
                data = json.loads(m.group())
                return {
                    "Extraversion": float(data.get("Extraversion", 50)),
                    "Neuroticism": float(data.get("Neuroticism", 50)),
                    "Conscientiousness": float(
                        data.get("Conscientiousness", 50)),
                    "Agreeableness": float(data.get("Agreeableness", 50)),
                    "Openness": float(data.get("Openness", 50)),
                    "reasoning": data.get("reasoning", ""),
                }
        except (json.JSONDecodeError, ValueError) as e:
            logging.warning(f"  Failed to parse assessment JSON: {e}")
        # Fallback: regex extraction
        result = {}
        for t in ["Extraversion", "Neuroticism", "Conscientiousness",
                   "Agreeableness", "Openness"]:
            match = re.search(rf'{t}[:\s]*(\d+)', raw, re.IGNORECASE)
            result[t] = float(match.group(1)) if match else None
        result["reasoning"] = raw
        return result

    # ---------- Single round (one conversation + two assessments) ----------

    def run_single_round(self, char_a, char_b,
                         llm_a_name, llm_b_name,
                         round_label="original"):
        """
        One conversation + bidirectional assessment.

        Returns:
            result_a: ExperimentResult — LLM_A assesses LLM_B (GT = char_b)
            result_b: ExperimentResult — LLM_B assesses LLM_A (GT = char_a)
        """
        llm_a_client = self.tested_llms[llm_a_name]
        llm_b_client = self.tested_llms[llm_b_name]

        logging.info(
            f"  [{round_label.upper()}] "
            f"{llm_a_name}(as {char_a.name}) <-> "
            f"{llm_b_name}(as {char_b.name})")

        # -- Run conversation --
        full_log, hist_a, hist_b = self._run_conversation(
            llm_a_name, llm_a_client, char_a,
            llm_b_name, llm_b_client, char_b)

        conv_serialized = [
            {"role": t.role, "speaker_llm": t.speaker_llm,
             "character": t.character_name, "content": t.content,
             "timestamp": t.timestamp}
            for t in full_log
        ]
        num_turns_actual = len(full_log) // 2

        # -- Assessment A→B: LLM_A assesses LLM_B --
        logging.info(
            f"  Assessment: {llm_a_name} assesses "
            f"{llm_b_name}(as {char_b.name})...")
        try:
            raw_a = llm_a_client.assess_personality(hist_a)
            inferred_a = self._parse_big5_assessment(raw_a)
        except Exception as e:
            logging.error(f"  Assessment A→B error: {e}")
            raw_a = str(e)
            inferred_a = {t: None for t in
                          ["Extraversion", "Neuroticism",
                           "Conscientiousness", "Agreeableness", "Openness"]}

        result_a = ExperimentResult(
            assessor_llm=llm_a_name,
            assessor_character_name=char_a.name,
            assessor_character_id=char_a.id,
            target_llm=llm_b_name,
            target_character_name=char_b.name,
            target_character_id=char_b.id,
            ground_truth=char_b.big5.to_dict(),
            inferred_big5=inferred_a,
            conversation_log=conv_serialized,
            num_turns=num_turns_actual,
            assessment_raw=raw_a,
            round_label=round_label,
            timestamp=datetime.now().isoformat(),
        )

        # -- Assessment B→A: LLM_B assesses LLM_A --
        logging.info(
            f"  Assessment: {llm_b_name} assesses "
            f"{llm_a_name}(as {char_a.name})...")
        try:
            raw_b = llm_b_client.assess_personality(hist_b)
            inferred_b = self._parse_big5_assessment(raw_b)
        except Exception as e:
            logging.error(f"  Assessment B→A error: {e}")
            raw_b = str(e)
            inferred_b = {t: None for t in
                          ["Extraversion", "Neuroticism",
                           "Conscientiousness", "Agreeableness", "Openness"]}

        result_b = ExperimentResult(
            assessor_llm=llm_b_name,
            assessor_character_name=char_b.name,
            assessor_character_id=char_b.id,
            target_llm=llm_a_name,
            target_character_name=char_a.name,
            target_character_id=char_a.id,
            ground_truth=char_a.big5.to_dict(),
            inferred_big5=inferred_b,
            conversation_log=conv_serialized,
            num_turns=num_turns_actual,
            assessment_raw=raw_b,
            round_label=round_label,
            timestamp=datetime.now().isoformat(),
        )

        return result_a, result_b

    # ---------- Run all experiments ----------

    def run_all_experiments(self, character_pairs: list):
        """
        For each (char_pair × LLM_pair):
          Round 1: LLM_A plays Char_X, LLM_B plays Char_Y
          Round 2: LLM_A plays Char_Y, LLM_B plays Char_X (SWAPPED)
        Each round produces 2 assessments → 4 assessments per combo.
        """
        llm_names = list(self.tested_llms.keys())
        llm_pairs = list(combinations(llm_names, 2))

        # conversations = char_pairs × llm_pairs × 2 rounds
        total_convs = len(character_pairs) * len(llm_pairs) * 2
        total_assessments = total_convs * 2
        current = 0

        logging.info(f"\n{'#'*60}")
        logging.info(
            "PARADIGM CONTROL — Bidirectional (Chinese prompts)")
        logging.info(f"{'#'*60}")
        logging.info(
            f"Paradigm: SYMMETRIC (both LLMs roleplay + both assess)")
        logging.info(
            f"Persona prompt: CHINESE (identical to Script A)")
        logging.info(
            f"Character pairs: {len(character_pairs)}")
        logging.info(
            f"LLM pairs: {llm_pairs}")
        logging.info(
            f"Turns per conversation: {self.num_turns}")
        logging.info(
            f"Total conversations: {total_convs} "
            f"({total_convs//2} original + {total_convs//2} swapped)")
        logging.info(
            f"Total assessments: {total_assessments}")
        logging.info(
            f"\nControlled variable: Paradigm "
            f"(asymmetric → symmetric)")
        logging.info(
            f"Held constant: Chinese persona prompt, assessment prompt, "
            f"opening trigger, API clients")
        logging.info(f"{'#'*60}\n")

        all_results = []

        for pair_idx, (char_x, char_y) in enumerate(character_pairs):
            logging.info(f"\n{'='*60}")
            logging.info(
                f"Character Pair {pair_idx+1}/{len(character_pairs)}:")
            logging.info(
                f"  X: {char_x.name} (ID:{char_x.id}) "
                f"Big5: {char_x.big5}")
            logging.info(
                f"  Y: {char_y.name} (ID:{char_y.id}) "
                f"Big5: {char_y.big5}")
            b5_dist = (
                abs(char_x.big5.extraversion - char_y.big5.extraversion)
                + abs(char_x.big5.neuroticism - char_y.big5.neuroticism)
                + abs(char_x.big5.conscientiousness
                      - char_y.big5.conscientiousness)
                + abs(char_x.big5.agreeableness - char_y.big5.agreeableness)
                + abs(char_x.big5.openness - char_y.big5.openness))
            logging.info(f"  Big5 Manhattan distance: {b5_dist}")
            logging.info(f"{'='*60}")

            for llm_a_name, llm_b_name in llm_pairs:

                # ---- Round 1: Original assignment ----
                current += 1
                logging.info(
                    f"\n[Conv {current}/{total_convs}] "
                    f"{llm_a_name}(as {char_x.name}) vs "
                    f"{llm_b_name}(as {char_y.name}) [ORIGINAL]")
                try:
                    r1_a, r1_b = self.run_single_round(
                        char_x, char_y, llm_a_name, llm_b_name,
                        round_label="original")
                    all_results.extend([r1_a, r1_b])
                    self._save_result(r1_a)
                    self._save_result(r1_b)
                    self._log_assessment(
                        r1_a, f"  {llm_a_name} assesses {llm_b_name}")
                    self._log_assessment(
                        r1_b, f"  {llm_b_name} assesses {llm_a_name}")
                except Exception as e:
                    logging.error(f"  Round 1 failed: {e}")
                time.sleep(2)

                # ---- Round 2: Swapped assignment ----
                current += 1
                logging.info(
                    f"\n[Conv {current}/{total_convs}] "
                    f"{llm_a_name}(as {char_y.name}) vs "
                    f"{llm_b_name}(as {char_x.name}) [SWAPPED]")
                try:
                    r2_a, r2_b = self.run_single_round(
                        char_y, char_x, llm_a_name, llm_b_name,
                        round_label="swapped")
                    all_results.extend([r2_a, r2_b])
                    self._save_result(r2_a)
                    self._save_result(r2_b)
                    self._log_assessment(
                        r2_a, f"  {llm_a_name} assesses {llm_b_name}")
                    self._log_assessment(
                        r2_b, f"  {llm_b_name} assesses {llm_a_name}")
                except Exception as e:
                    logging.error(f"  Round 2 (swapped) failed: {e}")
                time.sleep(2)

        # Save aggregates
        self._save_summary(all_results)
        self._print_final_summary(all_results)
        return all_results

    # ---------- Logging helper ----------

    def _log_assessment(self, result: ExperimentResult, label: str):
        errors = result.compute_errors()
        if errors.get("MAE") is not None:
            logging.info(f"{label} MAE: {errors['MAE']:.1f}%")
            for t in ["Extraversion", "Neuroticism",
                      "Conscientiousness", "Agreeableness", "Openness"]:
                e = errors[t]
                logging.info(
                    f"    {t}: GT={e['ground_truth']}% "
                    f"-> Inf={e['inferred']}% "
                    f"(err={e['absolute_error']}%)")
            logging.info(
                f"  Raw assessment response:\n    "
                f"{result.assessment_raw}")
        else:
            logging.warning(f"{label} Assessment FAILED")
            logging.warning(f"  Raw response: {result.assessment_raw}")

    # ---------- Save individual result ----------

    def _save_result(self, result: ExperimentResult):
        fn = (f"paradigm_ctrl_"
              f"{result.assessor_llm}_as_{result.assessor_character_id}_"
              f"assesses_"
              f"{result.target_llm}_as_{result.target_character_id}_"
              f"{result.round_label}.json")
        fn = re.sub(r'[^\w\-.]', '_', fn)
        path = self.output_dir / fn
        errors = result.compute_errors()
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "experiment_type": "paradigm_control_bidirectional",
                "paradigm": "symmetric",
                "persona_prompt_language": "chinese",
                "round_label": result.round_label,
                "assessor_llm": result.assessor_llm,
                "assessor_character_name": result.assessor_character_name,
                "assessor_character_id": result.assessor_character_id,
                "target_llm": result.target_llm,
                "target_character_name": result.target_character_name,
                "target_character_id": result.target_character_id,
                "ground_truth": result.ground_truth,
                "inferred_big5": result.inferred_big5,
                "errors": (errors if errors.get("MAE") is not None
                           else "assessment_failed"),
                "assessment_raw": result.assessment_raw,
                "num_turns": result.num_turns,
                "conversation_log": result.conversation_log,
                "timestamp": result.timestamp,
            }, f, indent=2, ensure_ascii=False)

    # ---------- Save summary ----------

    def _save_summary(self, results):
        valid = [r for r in results
                 if r.compute_errors()["MAE"] is not None]
        failed = len(results) - len(valid)
        if failed:
            logging.warning(
                f"  {failed} assessment(s) failed, excluded from summary.")

        TRAITS = ["Extraversion", "Neuroticism", "Conscientiousness",
                  "Agreeableness", "Openness"]

        # ----- CSV -----
        csv_path = self.output_dir / "summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "assessor_llm", "assessor_char_id", "assessor_char_name",
                "target_llm", "target_char_id", "target_char_name",
                "round_label",
                "gt_E", "gt_N", "gt_C", "gt_A", "gt_O",
                "inf_E", "inf_N", "inf_C", "inf_A", "inf_O",
                "err_E", "err_N", "err_C", "err_A", "err_O",
                "MAE", "num_turns", "timestamp",
            ])
            for r in valid:
                e = r.compute_errors()
                w.writerow([
                    r.assessor_llm,
                    r.assessor_character_id,
                    r.assessor_character_name,
                    r.target_llm,
                    r.target_character_id,
                    r.target_character_name,
                    r.round_label,
                    r.ground_truth.get("Extraversion"),
                    r.ground_truth.get("Neuroticism"),
                    r.ground_truth.get("Conscientiousness"),
                    r.ground_truth.get("Agreeableness"),
                    r.ground_truth.get("Openness"),
                    r.inferred_big5.get("Extraversion"),
                    r.inferred_big5.get("Neuroticism"),
                    r.inferred_big5.get("Conscientiousness"),
                    r.inferred_big5.get("Agreeableness"),
                    r.inferred_big5.get("Openness"),
                    e["Extraversion"]["absolute_error"],
                    e["Neuroticism"]["absolute_error"],
                    e["Conscientiousness"]["absolute_error"],
                    e["Agreeableness"]["absolute_error"],
                    e["Openness"]["absolute_error"],
                    e["MAE"], r.num_turns, r.timestamp,
                ])

        # ----- JSON aggregate -----
        from collections import defaultdict
        by_assessor = defaultdict(list)
        by_target = defaultdict(list)
        by_pair = defaultdict(list)
        by_trait = defaultdict(lambda: {"errors": [], "biases": []})
        by_round = defaultdict(list)

        for r in valid:
            e = r.compute_errors()
            mae = e["MAE"]
            by_assessor[r.assessor_llm].append(mae)
            by_target[r.target_llm].append(mae)
            by_pair[f"{r.assessor_llm}_assesses_{r.target_llm}"].append(mae)
            by_round[r.round_label].append(mae)
            for t in TRAITS:
                by_trait[t]["errors"].append(e[t]["absolute_error"])
                # bias = inferred - ground_truth
                gt = r.ground_truth.get(t, 50)
                inf = r.inferred_big5.get(t)
                if inf is not None:
                    by_trait[t]["biases"].append(inf - gt)

        def _avg(lst):
            return round(sum(lst) / len(lst), 1) if lst else None

        summary = {
            "experiment_type": "paradigm_control_bidirectional",
            "paradigm": "symmetric",
            "persona_prompt_language": "chinese",
            "controlled_variable": "experiment_paradigm "
                                   "(asymmetric vs symmetric)",
            "total_assessments": len(results),
            "valid_assessments": len(valid),
            "failed_assessments": failed,
            "overall_MAE": _avg([r.compute_errors()["MAE"] for r in valid]),
            "by_assessor_llm (inference ability)": {
                k: {"mean_MAE": _avg(v), "n": len(v)}
                for k, v in sorted(by_assessor.items())
            },
            "by_target_llm (expression ability)": {
                k: {"mean_MAE": _avg(v), "n": len(v)}
                for k, v in sorted(by_target.items())
            },
            "by_assessor_target_pair": {
                k: {"mean_MAE": _avg(v), "n": len(v)}
                for k, v in sorted(by_pair.items())
            },
            "by_round": {
                k: {"mean_MAE": _avg(v), "n": len(v)}
                for k, v in sorted(by_round.items())
            },
            "by_trait": {
                t: {"mean_error": _avg(d["errors"]),
                    "mean_bias": _avg(d["biases"]),
                    "n": len(d["errors"])}
                for t, d in by_trait.items()
            },
        }

        json_path = self.output_dir / "summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logging.info(f"\nResults saved to {self.output_dir}/")

    # ---------- Print final summary ----------

    def _print_final_summary(self, results):
        valid = [r for r in results
                 if r.compute_errors()["MAE"] is not None]
        failed = len(results) - len(valid)

        TRAITS = ["Extraversion", "Neuroticism", "Conscientiousness",
                  "Agreeableness", "Openness"]

        logging.info(f"\n{'#'*60}")
        logging.info(
            "PARADIGM CONTROL COMPLETE — Bidirectional (Chinese prompts)")
        logging.info(
            f"Total assessments: {len(results)} "
            f"({len(valid)} valid, {failed} failed)")
        logging.info(f"{'#'*60}")

        def _avg(lst):
            return sum(lst) / len(lst) if lst else 0

        # By assessor (= inference ability)
        from collections import defaultdict
        by_assessor = defaultdict(list)
        by_target = defaultdict(list)
        by_trait = defaultdict(lambda: {"errors": [], "biases": []})

        for r in valid:
            e = r.compute_errors()
            by_assessor[r.assessor_llm].append(e["MAE"])
            by_target[r.target_llm].append(e["MAE"])
            for t in TRAITS:
                by_trait[t]["errors"].append(e[t]["absolute_error"])
                gt = r.ground_truth.get(t, 50)
                inf = r.inferred_big5.get(t)
                if inf is not None:
                    by_trait[t]["biases"].append(inf - gt)

        logging.info(
            "\n  --- By Assessor LLM (inference ability) ---")
        for k in sorted(by_assessor):
            v = by_assessor[k]
            logging.info(
                f"  {k:12s} as assessor: "
                f"avg MAE = {_avg(v):.1f}% (n={len(v)})")

        logging.info(
            "\n  --- By Target LLM (expression ability) ---")
        logging.info(
            "  (lower MAE = partner assessed more accurately = "
            "better expression)")
        for k in sorted(by_target):
            v = by_target[k]
            logging.info(
                f"  {k:12s} as target:   "
                f"avg MAE = {_avg(v):.1f}% (n={len(v)})")

        logging.info("\n  --- By Trait ---")
        for t in TRAITS:
            d = by_trait[t]
            logging.info(
                f"  {t:20s}: mean err = {_avg(d['errors']):5.1f}%  "
                f"| bias = {_avg(d['biases']):+5.1f}%")

        logging.info(f"\n{'#'*60}")


# =============================================
# 7. CONFIGURATION & MAIN
# =============================================

def load_config(path="config.json"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def create_default_config():
    config = {
        "api_keys": {
            "zhipu": "YOUR_ZHIPU_API_KEY",
            "chatgpt": "YOUR_OPENAI_API_KEY",
            "claude": "YOUR_ANTHROPIC_API_KEY",
            "gemini": "YOUR_GOOGLE_API_KEY",
            "deepseek": "YOUR_DEEPSEEK_API_KEY",
            "grok": "YOUR_XAI_API_KEY",
        },
        "models": {
            "chatgpt": "gpt-4o",
            "claude": "claude-sonnet-4-20250514",
            "gemini": "gemini-2.0-flash",
            "deepseek": "deepseek-chat",
            "grok": "grok-3",
        },
        "experiment": {
            "num_turns": 10,
            "num_characters": 10,
            "num_pairs": 5,
            "output_dir": "results_paradigm_control",
        },
        "enabled_llms": [
            "chatgpt", "claude", "deepseek", "grok"],
    }
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("Created config.json -- please fill in your API keys.")
    print("Note: This config is compatible with the one-directional "
          "(Script A) and bidirectional (Script B) scripts.")
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Paradigm Control -- Bidirectional with Chinese prompts")
    parser.add_argument("--csv", type=str, required=False,
                        help="Path to pdb_cleaned.csv")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--num_characters", type=int, default=10,
                        help="Characters to load from CSV (pool size)")
    parser.add_argument("--num_pairs", type=int, default=5,
                        help="Number of character pairs to test")
    parser.add_argument("--num_turns", type=int, default=10)
    parser.add_argument("--output_dir", type=str,
                        default="results_paradigm_control")
    parser.add_argument("--init_config", action="store_true",
                        help="Create config.json template and exit")
    parser.add_argument("--llms", nargs="+", default=None,
                        help="Which LLMs (e.g. chatgpt claude grok). "
                             "Need at least 2.")
    parser.add_argument("--pairs", nargs="+", default=None,
                        help="Manual character pairs using CSV 'id' column. "
                             "Format: ID_A,ID_B for each pair. "
                             "Example: --pairs 1000521,1070394 "
                             "1022936,1000507 1156305,1103142")
    parser.add_argument("--list_characters", action="store_true",
                        help="List all characters in CSV with IDs and "
                             "Big5 scores, then exit.")
    args = parser.parse_args()

    # Logging (UTF-8 to avoid GBK issues on Windows)
    logfn = (f"paradigm_control_"
             f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"))
    fh = logging.FileHandler(logfn, encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[ch, fh])
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if args.init_config:
        create_default_config()
        return

    if args.list_characters:
        if not args.csv:
            parser.error("--csv is required with --list_characters")
        all_chars = load_characters_from_csv(args.csv, max_chars=None)
        print(f"\n{'ID':>10s}  {'Name':35s}  {'E':>5s} {'N':>5s} "
              f"{'C':>5s} {'A':>5s} {'O':>5s}  {'MBTI':8s}  "
              f"{'Source'}")
        print("-" * 120)
        for c in sorted(all_chars, key=lambda x: x.id):
            mbti = c.mbti.split()[0] if c.mbti else ""
            src = c.subcategory[:30] if c.subcategory else ""
            print(f"{c.id:>10d}  {c.name:35s}  "
                  f"{c.big5.extraversion:5.0f} {c.big5.neuroticism:5.0f} "
                  f"{c.big5.conscientiousness:5.0f} "
                  f"{c.big5.agreeableness:5.0f} {c.big5.openness:5.0f}  "
                  f"{mbti:8s}  {src}")
        print(f"\nTotal: {len(all_chars)} characters")
        print(f"\nUsage example:")
        print(f"  python {sys.argv[0]} --csv {args.csv} "
              f"--pairs {all_chars[0].id},{all_chars[1].id} "
              f"{all_chars[2].id},{all_chars[3].id}")
        return

    if not args.csv:
        parser.error("--csv is required (unless using --init_config "
                      "or --list_characters)")

    config = load_config(args.config)
    if not config:
        logging.error(
            "No config.json found. Run with --init_config first.")
        return

    api_keys = config.get("api_keys", {})
    models = config.get("models", {})
    enabled_llms = args.llms or config.get("enabled_llms", [])

    # Initialize LLM clients (need at least 2 for pairing)
    tested_llms = {}
    for name in enabled_llms:
        key = api_keys.get(name, "")
        if key and key != f"YOUR_{name.upper()}_API_KEY":
            try:
                tested_llms[name] = TestedLLMClient(
                    provider=name, api_key=key, model=models.get(name))
                logging.info(
                    f"Initialized {name} "
                    f"(model: {models.get(name, 'default')})")
            except Exception as e:
                logging.warning(f"Failed to init {name}: {e}")
        else:
            logging.warning(f"Skipping {name} -- no valid API key.")

    if len(tested_llms) < 2:
        logging.error(
            "Need at least 2 LLMs for pairing. "
            f"Got: {list(tested_llms.keys())}. Check config.json.")
        return

    # Load characters and build pairs
    if args.pairs:
        # --pairs mode: load ALL characters, resolve IDs
        all_characters = load_characters_from_csv(args.csv, max_chars=None)
        id_lookup = {c.id: c for c in all_characters}

        char_pairs = []
        for pair_str in args.pairs:
            pair_str = pair_str.strip()
            # Support both "ID_A,ID_B" and "ID_A:ID_B"
            parts = re.split(r'[,:]', pair_str)
            if len(parts) != 2:
                logging.error(
                    f"Invalid pair format: '{pair_str}'. "
                    f"Expected ID_A,ID_B (e.g. 1000521,1070394)")
                return
            try:
                id_a, id_b = int(parts[0].strip()), int(parts[1].strip())
            except ValueError:
                logging.error(
                    f"Invalid IDs in pair: '{pair_str}'. "
                    f"IDs must be integers.")
                return

            if id_a not in id_lookup:
                logging.error(
                    f"Character ID {id_a} not found in CSV. "
                    f"Use --list_characters to see available IDs.")
                return
            if id_b not in id_lookup:
                logging.error(
                    f"Character ID {id_b} not found in CSV. "
                    f"Use --list_characters to see available IDs.")
                return
            char_pairs.append((id_lookup[id_a], id_lookup[id_b]))

        logging.info(f"\nManual character pairs ({len(char_pairs)}):")
    else:
        # Auto mode: sample diverse characters, then pick pairs
        characters = load_characters_from_csv(
            args.csv, max_chars=args.num_characters)
        if len(characters) < 2:
            logging.error(
                f"Need at least 2 characters, got {len(characters)}.")
            return
        char_pairs = select_character_pairs(characters, args.num_pairs)
        logging.info(f"\nAuto-selected {len(char_pairs)} character pairs:")

    for i, (a, b) in enumerate(char_pairs):
        dist = (abs(a.big5.extraversion - b.big5.extraversion)
                + abs(a.big5.neuroticism - b.big5.neuroticism)
                + abs(a.big5.conscientiousness - b.big5.conscientiousness)
                + abs(a.big5.agreeableness - b.big5.agreeableness)
                + abs(a.big5.openness - b.big5.openness))
        logging.info(
            f"  Pair {i+1}: [ID:{a.id}] {a.name} ({a.big5}) vs "
            f"[ID:{b.id}] {b.name} ({b.big5})  [dist={dist}]")

    # Run experiments
    orchestrator = BidirectionalControlOrchestrator(
        tested_llm_clients=tested_llms,
        num_turns=args.num_turns,
        output_dir=args.output_dir,
    )

    results = orchestrator.run_all_experiments(char_pairs)


if __name__ == "__main__":
    main()
