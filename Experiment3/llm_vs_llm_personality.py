"""
LLM Personality Inference — CONTROL Experiment (LLM replaces CharGLM)
=====================================================================

CONTROLLED-VARIABLE DESIGN:
  This script is a faithful replica of llm_personality_test_bidirectional.py
  with EXACTLY ONE variable changed:

    CharGLM-4  →  a designated "simulator" LLM (e.g. GPT-4o, DeepSeek)

  EVERYTHING else is held IDENTICAL to the bidirectional (Exp1) script:
    ✓ Same asymmetric protocol (interviewer LLM vs. character simulator)
    ✓ Same dual modes (llm_first, char_first)
    ✓ Same 19 conversation starters (word-for-word from Exp1)
    ✓ Same 12 follow-up prompts (word-for-word from Exp1)
    ✓ Same interviewer system prompts (word-for-word from Exp1)
    ✓ Same assessment prompt + system message (word-for-word from Exp1)
    ✓ Same ConversationTurn / ExperimentResult data models
    ✓ Same num_turns, same saving format, same summary statistics
    ✓ Roleplay prompt = faithful English translation of Exp1's Chinese
      CharGLM prompt (NO Big5 hints, NO "You ARE this person",
      NO "do not monologue")
    ✓ Same opening situations (translated from Exp1's Chinese)

  The simulator LLM is EXCLUDED from the tested (assessor) LLMs to
  avoid self-assessment contamination.

PURPOSE:
  By comparing results against llm_personality_test_bidirectional.py
  (which uses CharGLM-4), we isolate the single variable:

    "Does replacing CharGLM with a general-purpose LLM as the character
     expresser change personality inference accuracy?"

  If both scripts produce similar error patterns → structural limitation.
  If only CharGLM shows certain biases → CharGLM-specific artifact.

Tested LLMs (assessors): ChatGPT, Claude, Gemini, DeepSeek, Grok
Simulator LLM (replaces CharGLM): configurable, default ChatGPT
Ground truth: pdb_cleaned.csv (Personality Database characters with Big5)

Usage:
    python llm_vs_llm_personality.py --csv pdb_cleaned.csv --num_characters 5
    python llm_vs_llm_personality.py --csv pdb_cleaned.csv --simulator_llm deepseek
    python llm_vs_llm_personality.py --csv pdb_cleaned.csv --llms claude grok
    python llm_vs_llm_personality.py --csv pdb_cleaned.csv --modes llm_first
    python llm_vs_llm_personality.py --csv pdb_cleaned.csv --characters 1000521 1070394
    python llm_vs_llm_personality.py --csv pdb_cleaned.csv --list_characters
    python llm_vs_llm_personality.py --init_config
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


# =============================================
# 1. DATA MODELS (identical to Exp1)
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
    Identical to Exp1: role is 'character' or 'llm'."""
    role: str       # "character" or "llm"
    content: str
    timestamp: float = 0.0


@dataclass
class ExperimentResult:
    """Result of a single character x assessor LLM x mode experiment.
    Identical to Exp1 + one added field: simulator_llm."""
    character_name: str
    character_id: int
    tested_llm: str
    simulator_llm: str      # NEW: which LLM played the character
    mode: str               # "llm_first" or "char_first"
    ground_truth: dict
    inferred_big5: dict
    conversation_log: list
    num_turns: int
    assessment_raw: str
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
# 2. CSV PARSER (identical to Exp1)
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
    Identical to Exp1."""
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
# 3. SIMULATOR ROLEPLAY PROMPT
#    Faithful English translation of Exp1's Chinese CharGLM prompt.
#    NO Big5 hints. NO "You ARE this person". NO "do not monologue".
# =============================================

def build_roleplay_system_prompt(character: Character) -> str:
    """
    Faithful English translation of Exp1's build_charglm_persona().

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
    Build opening prompt for char_first mode.
    Faithful translation of Exp1's CharGLM generate_opening().

    Exp1 situations (Chinese -> English):
      你刚在一个社交场合遇到了一个新认识的人 -> social gathering
      你在咖啡店遇到了一个看起来很友好的陌生人 -> coffee shop
      你在一个朋友的聚会上遇到了一个你不认识的人 -> friend's party
      你在等公交车的时候旁边站了一个陌生人 -> bus stop
      你在书店里遇到一个看起来有趣的人 -> bookstore
      你在公园散步时碰到了一个坐在长椅上的人 -> park bench

    Exp1 instruction:
      用符合你角色性格的方式开始对话。说2-4句话。
      -> Start the conversation in character. Say 2-4 sentences.
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
# 4. LLM API CLIENTS
# =============================================

class TestedLLMClient:
    """Unified client for tested LLMs (identical to Exp1)."""

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
        """Identical assessment prompt to Exp1 (word-for-word)."""
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


class CharacterSimulatorClient:
    """
    Drop-in replacement for Exp1's CharGLMClient.

    Wraps a TestedLLMClient to roleplay characters using an English
    system prompt (faithful translation of Exp1's Chinese prompt).
    Provides the same interface as CharGLMClient:
        - generate_response(character, conversation_history, user_message)
        - generate_opening(character)

    This is the ONLY architectural change from Exp1.
    """

    def __init__(self, llm_client: TestedLLMClient, provider_name: str):
        self.llm_client = llm_client
        self.provider_name = provider_name

    def generate_response(self, character: Character,
                          conversation_history: list[dict],
                          user_message: str) -> str:
        """Generate an in-character response (replaces CharGLM._call)."""
        persona_prompt = build_roleplay_system_prompt(character)
        messages = conversation_history.copy()
        messages.append({"role": "user", "content": user_message})
        try:
            return self.llm_client.chat(messages,
                                         system_prompt=persona_prompt)
        except Exception as e:
            logging.error(f"CharacterSimulator ({self.provider_name}) "
                          f"API error: {e}")
            raise

    def generate_opening(self, character: Character) -> str:
        """
        Generate a character-initiated opening message.
        Replaces CharGLM.generate_opening().

        Exp1 equivalent:
          messages = [system, {"role":"user","content":"（你看到了对方...）"}]
        """
        opening_prompt = build_opening_system_prompt(character)
        try:
            return self.llm_client.chat(
                [{"role": "user",
                  "content":  "（你看到了对方，决定主动开口说话）"}],
                system_prompt=opening_prompt)
        except Exception as e:
            logging.error(f"CharacterSimulator ({self.provider_name}) "
                          f"opening generation error: {e}")
            raise


# =============================================
# 5. CONVERSATION TOPICS (identical to Exp1, word-for-word)
# =============================================

LLM_CONVERSATION_STARTERS = [
    "Hi there! I'd love to get to know you. What's something you've been really excited about lately?",
    "Tell me about a memorable experience you've had recently that left a big impression on you.",
    "What's your take on social media? Do you think it's more helpful or harmful to society?",
    "If you could change one thing about the world, what would it be and why?",
    "Imagine you're stranded on a deserted island with three items of your choice. What would you pick and why?",
    "If you were given $1 million to start a business, what kind of business would you create?",
    "Can you describe your ideal day from morning to night? Don't hold back on the details!",
    "If you could live in any fictional world, which one would you choose and what would you do there?",
    "How do you usually handle disagreements with friends or family?",
    "What qualities do you value most in a close friend?",
    "How would your best friend describe you to someone who's never met you?",
    "What's something you've been wanting to improve about yourself?",
    "What's the most spontaneous thing you've ever done?",
    "When you're feeling really overwhelmed, what do you usually do to cope?",
    "Do you prefer having a detailed plan or going with the flow? Why?",
    "What do you think makes someone truly trustworthy?",
    "When someone you care about is going through a tough time, how do you usually respond?",
    "How do you usually approach a big decision that could go either way?",
    "What's a moment in your life that really changed how you see things?",
]

FOLLOW_UP_PROMPTS = [
    "That's interesting! Can you tell me more about that?",
    "I see. What made you feel that way?",
    "How did that experience change your perspective?",
    "What would you do differently if you could go back?",
    "That's a unique take. How do others around you usually react to that?",
    "I'm curious - do you think that says something about your personality?",
    "How does that compare to how you used to think about it?",
    "What emotions come up when you think about that?",
    "Do you find yourself in situations like that often?",
    "That's really thoughtful. What drives you to think that way?",
    "Has anyone ever surprised you with how they reacted to something like that?",
    "What's the hardest part about dealing with that kind of situation?",
]


# =============================================
# 6. EXPERIMENT ORCHESTRATOR
#    Identical flow to Exp1's ExperimentOrchestrator.
#    Only change: self.charglm -> self.simulator
# =============================================

class ControlExperimentOrchestrator:
    """
    Orchestrates dual-mode multi-turn conversations between a
    character simulator LLM (replacing CharGLM) and tested LLMs,
    then collects personality assessments.

    IDENTICAL flow to Exp1's ExperimentOrchestrator:
      - Same asymmetric protocol (interviewer vs character)
      - Same _run_llm_first / _run_char_first engines
      - Same one-way assessment (interviewer assesses character)

    Modes:
        llm_first  -- Tested LLM speaks first (interviewer role)
        char_first -- Character simulator speaks first (in character)
    """

    MODES = ["llm_first", "char_first"]

    def __init__(self, simulator: CharacterSimulatorClient,
                 simulator_name: str,
                 tested_llm_clients: dict,
                 num_turns: int = 10,
                 output_dir: str = "results_control"):
        self.simulator = simulator
        self.simulator_name = simulator_name
        self.tested_llms = tested_llm_clients
        self.num_turns = num_turns
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Core conversation engines (identical to Exp1) ----------

    def _run_llm_first(self, character, llm_name, llm_client):
        """
        LLM-first mode: tested LLM initiates and drives the conversation.
        Flow: LLM asks -> Simulator answers (in character) -> LLM follows up
        Identical to Exp1 _run_llm_first (CharGLM -> simulator).
        """
        llm_hist = []   # tested LLM message history
        full_log = []   # complete conversation log

        # Exp1 interviewer system prompt (word-for-word)
        sys_prompt = (
            "You are having a natural, friendly conversation with someone "
            "you just met. Be curious, ask follow-up questions, and try to "
            "understand who this person is. Keep your responses "
            "conversational and not too long (2-4 sentences)."
        )

        for turn in range(self.num_turns):
            # -- Tested LLM speaks (interviewer) --
            if turn == 0:
                llm_msg = random.choice(LLM_CONVERSATION_STARTERS)
            else:
                try:
                    llm_msg = llm_client.chat(llm_hist,
                                              system_prompt=sys_prompt)
                    if llm_msg is None:
                        llm_msg = random.choice(FOLLOW_UP_PROMPTS)
                except Exception as e:
                    logging.error(f"    Turn {turn}: LLM error: {e}")
                    break

            llm_hist.append({"role": "assistant", "content": llm_msg})
            full_log.append(ConversationTurn("llm", llm_msg, time.time()))
            logging.info(
                f"    Turn {turn+1}/{self.num_turns} "
                f"[LLM] {str(llm_msg)[:800000]}...")

            # -- Character simulator responds (replaces CharGLM) --
            sim_hist = self._to_simulator_history(full_log)
            try:
                char_msg = self.simulator.generate_response(
                    character, sim_hist, llm_msg)
            except Exception as e:
                logging.error(
                    f"    Turn {turn}: Simulator error: {e}")
                break
            if char_msg is None:
                logging.warning(
                    f"    Turn {turn}: Simulator empty response")
                break

            llm_hist.append({"role": "user", "content": char_msg})
            full_log.append(
                ConversationTurn("character", char_msg, time.time()))
            logging.info(
                f"    Turn {turn+1}/{self.num_turns} "
                f"[SIM] {str(char_msg)[:800000]}...")
            time.sleep(1)

        return llm_hist, full_log

    def _run_char_first(self, character, llm_name, llm_client):
        """
        Character-first mode: Simulator initiates the conversation.
        Flow: Simulator opens -> LLM responds -> Simulator continues
        Identical to Exp1 _run_char_first (CharGLM -> simulator).
        """
        llm_hist = []
        full_log = []

        # Exp1 responder system prompt (word-for-word)
        sys_prompt = (
            "You are having a natural, friendly conversation with someone "
            "you just met. They approached you and started talking. Respond "
            "naturally, be curious about them, ask follow-up questions, and "
            "try to understand who this person is. Keep your responses "
            "conversational and not too long (2-4 sentences)."
        )

        for turn in range(self.num_turns):
            # -- Character simulator speaks --
            if turn == 0:
                try:
                    char_msg = self.simulator.generate_opening(character)
                except Exception as e:
                    logging.error(
                        f"    Turn {turn}: Simulator opening error: {e}")
                    break
            else:
                sim_hist = self._to_simulator_history(full_log)
                last_llm = (llm_hist[-1]["content"]
                            if llm_hist else "")
                try:
                    char_msg = self.simulator.generate_response(
                        character, sim_hist, last_llm)
                except Exception as e:
                    logging.error(
                        f"    Turn {turn}: Simulator error: {e}")
                    break

            if char_msg is None:
                logging.warning(
                    f"    Turn {turn}: Simulator empty response")
                break

            llm_hist.append({"role": "user", "content": char_msg})
            full_log.append(
                ConversationTurn("character", char_msg, time.time()))
            logging.info(
                f"    Turn {turn+1}/{self.num_turns} "
                f"[SIM] {str(char_msg)[:800000]}...")

            # -- Tested LLM responds --
            try:
                llm_msg = llm_client.chat(llm_hist,
                                          system_prompt=sys_prompt)
                if llm_msg is None:
                    llm_msg = random.choice(FOLLOW_UP_PROMPTS)
            except Exception as e:
                logging.error(f"    Turn {turn}: LLM error: {e}")
                break

            llm_hist.append({"role": "assistant", "content": llm_msg})
            full_log.append(ConversationTurn("llm", llm_msg, time.time()))
            logging.info(
                f"    Turn {turn+1}/{self.num_turns} "
                f"[LLM] {str(llm_msg)[:800000]}...")
            time.sleep(1)

        return llm_hist, full_log

    # ---------- Helpers ----------

    def _to_simulator_history(self, full_log):
        """Convert full_log to simulator perspective
        (character=assistant, llm=user). Mirrors Exp1 _to_charglm_history."""
        out = []
        for t in full_log:
            if t.role == "llm":
                out.append({"role": "user", "content": t.content})
            elif t.role == "character":
                out.append({"role": "assistant", "content": t.content})
        return out

    def _parse_big5_assessment(self, raw):
        """Identical to Exp1."""
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
        result = {}
        for t in ["Extraversion", "Neuroticism", "Conscientiousness",
                   "Agreeableness", "Openness"]:
            match = re.search(rf'{t}[:\s]*(\d+)', raw, re.IGNORECASE)
            result[t] = float(match.group(1)) if match else None
        result["reasoning"] = raw
        return result

    # ---------- Single experiment (identical to Exp1) ----------

    def run_single_experiment(self, character, llm_name, mode):
        """Identical flow to Exp1 run_single_experiment."""
        llm_client = self.tested_llms[llm_name]
        logging.info(
            f"  [{mode.upper()}] {character.name} "
            f"<-> {llm_name} (sim: {self.simulator_name})")

        if mode == "llm_first":
            llm_hist, full_log = self._run_llm_first(
                character, llm_name, llm_client)
        else:
            llm_hist, full_log = self._run_char_first(
                character, llm_name, llm_client)

        # -- Personality Assessment (identical prompt to Exp1) --
        logging.info(
            f"  Requesting assessment from {llm_name} [{mode}]...")
        try:
            raw = llm_client.assess_personality(llm_hist)
            inferred = self._parse_big5_assessment(raw)
        except Exception as e:
            logging.error(f"  Assessment error: {e}")
            raw = str(e)
            inferred = {t: None for t in
                        ["Extraversion", "Neuroticism",
                         "Conscientiousness", "Agreeableness", "Openness"]}

        return ExperimentResult(
            character_name=character.name,
            character_id=character.id,
            tested_llm=llm_name,
            simulator_llm=self.simulator_name,
            mode=mode,
            ground_truth=character.big5.to_dict(),
            inferred_big5=inferred,
            conversation_log=[
                {"role": t.role, "content": t.content,
                 "timestamp": t.timestamp}
                for t in full_log
            ],
            num_turns=len(full_log) // 2,
            assessment_raw=raw,
            timestamp=datetime.now().isoformat(),
        )

    # ---------- Run all experiments (identical iteration to Exp1) ----------

    def run_all_experiments(self, characters):
        """
        For each character x tested LLM pair, run BOTH modes.
        Total experiments = characters x LLMs x len(MODES).
        Identical iteration structure to Exp1.
        """
        all_results = []
        total = len(characters) * len(self.tested_llms) * len(self.MODES)
        current = 0

        for char in characters:
            logging.info(f"\n{'='*60}")
            logging.info(f"Character: {char.name} (ID: {char.id})")
            logging.info(
                f"Source: {char.subcategory} [{char.category}]")
            logging.info(f"Ground Truth Big5: {char.big5}")
            info = f"MBTI: {char.mbti}"
            if char.enneagram:
                info += f" | Ennea: {char.enneagram}"
            if char.tritype:
                info += f" | Tritype: {char.tritype}"
            logging.info(info)
            extras = []
            if char.socionics:
                extras.append(f"Socionics: {char.socionics}")
            if char.temperament:
                extras.append(f"Temp: {char.temperament}")
            if char.alignment:
                extras.append(f"Align: {char.alignment}")
            if extras:
                logging.info(" | ".join(extras))
            if char.wiki_description:
                logging.info(
                    f"Wiki: {char.wiki_description[:100]}...")
            logging.info(f"{'='*60}")

            for llm_name in self.tested_llms:
                for mode in self.MODES:
                    current += 1
                    logging.info(
                        f"\n[{current}/{total}] {llm_name} x "
                        f"{char.name} [{mode}]")
                    try:
                        result = self.run_single_experiment(
                            char, llm_name, mode)
                        all_results.append(result)
                        self._save_result(result)

                        errors = result.compute_errors()
                        if errors["MAE"] is not None:
                            logging.info(
                                f"  MAE: {errors['MAE']:.1f}%")
                            for t in ["Extraversion", "Neuroticism",
                                      "Conscientiousness",
                                      "Agreeableness", "Openness"]:
                                logging.info(
                                    f"    {t}: "
                                    f"GT={errors[t]['ground_truth']}% "
                                    f"-> Inf={errors[t]['inferred']}% "
                                    f"(err={errors[t]['absolute_error']}%)"
                                )
                        else:
                            logging.warning(
                                "  Assessment failed")
                    except Exception as e:
                        logging.error(
                            f"  Experiment failed: {e}")
                    time.sleep(2)

        self._save_summary(all_results)
        return all_results

    # ---------- Saving (identical to Exp1 + simulator_llm field) ----------

    def _save_result(self, result):
        fn = (f"{result.character_id}_{result.character_name}_"
              f"{result.tested_llm}_{result.mode}.json")
        fn = re.sub(r'[^\w\-.]', '_', fn)
        path = self.output_dir / fn
        errors = result.compute_errors()
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "character_name": result.character_name,
                "character_id": result.character_id,
                "tested_llm": result.tested_llm,
                "simulator_llm": result.simulator_llm,
                "mode": result.mode,
                "num_turns": result.num_turns,
                "ground_truth": result.ground_truth,
                "inferred_big5": result.inferred_big5,
                "errors": (errors if errors.get("MAE") is not None
                           else "assessment_failed"),
                "assessment_raw": result.assessment_raw,
                "conversation_log": result.conversation_log,
                "timestamp": result.timestamp,
            }, f, indent=2, ensure_ascii=False)

    def _save_summary(self, results):
        """Identical summary format to Exp1."""
        valid = [r for r in results
                 if r.compute_errors()["MAE"] is not None]
        failed = len(results) - len(valid)
        if failed:
            logging.warning(
                f"  {failed} experiment(s) failed, excluded from summary.")

        # CSV (same columns as Exp1 + simulator_llm)
        csv_path = self.output_dir / "summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "character_id", "character_name", "tested_llm",
                "simulator_llm", "mode",
                "gt_E", "gt_N", "gt_C", "gt_A", "gt_O",
                "inf_E", "inf_N", "inf_C", "inf_A", "inf_O",
                "err_E", "err_N", "err_C", "err_A", "err_O",
                "MAE", "num_turns", "timestamp",
            ])
            for r in valid:
                e = r.compute_errors()
                w.writerow([
                    r.character_id, r.character_name, r.tested_llm,
                    r.simulator_llm, r.mode,
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

        # JSON aggregate (identical to Exp1)
        summary = self._compute_aggregate_stats(valid)
        json_path = self.output_dir / "summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logging.info(f"\nResults saved to {self.output_dir}/")

    def _compute_aggregate_stats(self, results):
        """Aggregate by LLM, by mode, by LLM x mode, and by trait.
        Identical to Exp1."""
        TRAITS = ["Extraversion", "Neuroticism", "Conscientiousness",
                  "Agreeableness", "Openness"]
        by_llm, by_mode, by_llm_mode, by_trait = {}, {}, {}, {}

        for r in results:
            e = r.compute_errors()
            llm, mode = r.tested_llm, r.mode
            lm = f"{llm}_{mode}"

            for key, bkt in [(llm, by_llm), (mode, by_mode),
                             (lm, by_llm_mode)]:
                if key not in bkt:
                    bkt[key] = {"MAE_list": [], "trait_errors": {}}
                bkt[key]["MAE_list"].append(e["MAE"])
                for t in TRAITS:
                    bkt[key]["trait_errors"].setdefault(
                        t, []).append(e[t]["absolute_error"])

            for t in TRAITS:
                by_trait.setdefault(t, []).append(
                    e[t]["absolute_error"])

        def _s(bkt):
            out = {}
            for k, d in bkt.items():
                ml = d["MAE_list"]
                out[k] = {
                    "mean_MAE": sum(ml) / len(ml) if ml else 0,
                    "n": len(ml),
                    "trait_mean_errors": {
                        t: sum(es) / len(es) if es else 0
                        for t, es in d["trait_errors"].items()
                    },
                }
            return out

        return {
            "total_experiments": len(results),
            "simulator_llm": self.simulator_name,
            "by_llm": _s(by_llm),
            "by_mode": _s(by_mode),
            "by_llm_mode": _s(by_llm_mode),
            "by_trait": {
                t: {"mean_error": sum(es)/len(es) if es else 0,
                     "n": len(es)}
                for t, es in by_trait.items()
            },
        }


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
            "num_characters": 5,
            "output_dir": "results_control",
        },
        "enabled_llms": [
            "chatgpt", "claude", "gemini", "deepseek", "grok"],
        "simulator_llm": "chatgpt",
    }
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("Created config.json -- please fill in your API keys.")
    print("Note: 'simulator_llm' specifies which LLM replaces CharGLM.")
    return config


def main():
    parser = argparse.ArgumentParser(
        description="LLM Personality Inference -- CONTROL (LLM replaces "
                    "CharGLM)")
    parser.add_argument("--csv", type=str, required=False,
                        help="Path to pdb_cleaned.csv")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--num_characters", type=int, default=5,
                        help="Characters to sample (used only if "
                             "--characters is not set)")
    parser.add_argument("--num_turns", type=int, default=10)
    parser.add_argument("--output_dir", type=str,
                        default="results_control")
    parser.add_argument("--init_config", action="store_true")
    parser.add_argument("--llms", nargs="+", default=None,
                        help="Which LLMs to test as assessors "
                             "(e.g. chatgpt claude)")
    parser.add_argument("--modes", nargs="+", default=None,
                        choices=["llm_first", "char_first"],
                        help="Which modes (default: both)")
    parser.add_argument("--simulator_llm", type=str, default=None,
                        help="LLM to use as character simulator "
                             "(replaces CharGLM). Default: chatgpt. "
                             "Must have a valid API key in config.")
    parser.add_argument("--characters", nargs="+", default=None,
                        help="Exact character IDs from CSV to use. "
                             "Ensures same characters across experiments. "
                             "Example: --characters 1000521 1070394 1022936")
    parser.add_argument("--list_characters", action="store_true",
                        help="List all characters in CSV with IDs and "
                             "Big5 scores, then exit.")
    args = parser.parse_args()

    # Logging (UTF-8 to avoid GBK issues on Windows)
    logfn = (f"control_experiment_"
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
        if len(all_chars) >= 3:
            ids = [str(c.id) for c in
                   sorted(all_chars, key=lambda x: x.id)[:3]]
            print(f"\nExample: python {sys.argv[0]} --csv {args.csv} "
                  f"--characters {' '.join(ids)}")
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

    # -- Character Simulator LLM (replaces CharGLM) --
    sim_name = (args.simulator_llm
                or config.get("simulator_llm", "chatgpt"))
    sim_key = api_keys.get(sim_name, "")
    if not sim_key or sim_key == f"YOUR_{sim_name.upper()}_API_KEY":
        logging.error(
            f"No valid API key for simulator LLM '{sim_name}'. "
            f"Check config.json.")
        return
    try:
        sim_llm_client = TestedLLMClient(
            provider=sim_name, api_key=sim_key,
            model=models.get(sim_name))
        simulator = CharacterSimulatorClient(sim_llm_client, sim_name)
        logging.info(
            f"Character simulator: {sim_name} "
            f"(model: {models.get(sim_name, 'default')}) "
            f"[replaces CharGLM]")
    except Exception as e:
        logging.error(f"Failed to initialize simulator '{sim_name}': {e}")
        return

    # -- Tested LLMs (assessors, identical to Exp1) --
    tested_llms = {}
    for name in enabled_llms:
        # Exclude the simulator LLM from being tested against itself
        if name == sim_name:
            logging.info(
                f"Excluding {name} from tested LLMs "
                f"(same as simulator).")
            continue
        key = api_keys.get(name, "")
        if key and key != f"YOUR_{name.upper()}_API_KEY":
            tested_llms[name] = TestedLLMClient(
                provider=name, api_key=key, model=models.get(name))
            logging.info(
                f"Initialized {name} "
                f"(model: {models.get(name, 'default')})")
        else:
            logging.warning(f"Skipping {name} -- no valid API key.")

    if not tested_llms:
        logging.error("No tested LLMs initialized. Check config.json.")
        return

    # Allow restricting modes via CLI
    if args.modes:
        ControlExperimentOrchestrator.MODES = args.modes

    # -- Load characters (identical to Exp1) --
    if args.characters:
        all_characters = load_characters_from_csv(args.csv, max_chars=None)
        id_lookup = {c.id: c for c in all_characters}

        characters = []
        for cid_str in args.characters:
            try:
                cid = int(cid_str.strip())
            except ValueError:
                logging.error(
                    f"Invalid character ID: '{cid_str}'. "
                    f"IDs must be integers.")
                return
            if cid not in id_lookup:
                logging.error(
                    f"Character ID {cid} not found in CSV. "
                    f"Use --list_characters to see available IDs.")
                return
            characters.append(id_lookup[cid])

        logging.info(
            f"Using {len(characters)} specified characters: "
            f"{[c.name for c in characters]}")
    else:
        characters = load_characters_from_csv(
            args.csv, max_chars=args.num_characters)

    if not characters:
        logging.error("No characters loaded.")
        return

    orchestrator = ControlExperimentOrchestrator(
        simulator=simulator,
        simulator_name=sim_name,
        tested_llm_clients=tested_llms,
        num_turns=args.num_turns,
        output_dir=args.output_dir,
    )

    n_modes = len(ControlExperimentOrchestrator.MODES)
    total = len(characters) * len(tested_llms) * n_modes

    logging.info(f"\n{'#'*60}")
    logging.info(
        "CONTROL EXPERIMENT -- Personality Inference (LLM replaces CharGLM)")
    logging.info(f"{'#'*60}")
    logging.info(
        f"Character simulator: {sim_name} "
        f"(replaces CharGLM-4)")
    logging.info(
        f"Characters: {len(characters)} | "
        f"Tested LLMs (assessors): {list(tested_llms.keys())}")
    logging.info(f"Modes: {ControlExperimentOrchestrator.MODES}")
    logging.info(f"Turns per conversation: {args.num_turns}")
    logging.info(f"Total experiments: {total}")
    logging.info(
        f"\nSingle controlled variable: CharGLM -> {sim_name}")
    logging.info(
        f"All other variables identical to bidirectional experiment.")
    logging.info(f"{'#'*60}\n")

    results = orchestrator.run_all_experiments(characters)

    # -- Final summary (identical format to Exp1) --
    logging.info(f"\n{'#'*60}")
    logging.info("CONTROL EXPERIMENT COMPLETE")
    logging.info(f"Simulator: {sim_name}")
    logging.info(f"Total experiments: {len(results)}")

    for llm_name in tested_llms:
        for mode in ControlExperimentOrchestrator.MODES:
            subset = [r for r in results
                      if r.tested_llm == llm_name and r.mode == mode]
            if not subset:
                continue
            maes = [r.compute_errors()["MAE"] for r in subset
                    if r.compute_errors()["MAE"] is not None]
            if maes:
                avg = sum(maes) / len(maes)
                logging.info(
                    f"  {llm_name} [{mode}]: "
                    f"avg MAE = {avg:.1f}% ({len(maes)} valid)")
            else:
                logging.info(
                    f"  {llm_name} [{mode}]: all failed")

    # Overall mode comparison
    for mode in ControlExperimentOrchestrator.MODES:
        subset = [r for r in results if r.mode == mode]
        maes = [r.compute_errors()["MAE"] for r in subset
                if r.compute_errors()["MAE"] is not None]
        if maes:
            logging.info(
                f"  Overall [{mode}]: "
                f"avg MAE = {sum(maes)/len(maes):.1f}% "
                f"({len(maes)} exps)")

    logging.info(f"{'#'*60}")


if __name__ == "__main__":
    main()
