"""
=============================================================================
 GriefBot — a grief-support chatbot on LangGraph
=============================================================================

                              intake
                                |
          +---------------------+---------------------+
          |                     |                     |
    moderation            validation            crisis_guard      <- concurrent
          |                     |                     |
          +---------------------+---------------------+
                                |
                              gate          <- waits for all three, then routes
                                |
     +-----------+--------------+--------------+
     |           |              |              |
crisis_response refuse      offtopic       remember
     |           |              |              |
     |           |              |          retrieve
     |           |              |              |
     |           |              |         supervisor <---+
     |           |              |          /   |   \     |
     |           |              |   emotional coping ... -+
     |           |              |              |
     |           |              |          compose <-> critic
     +-----------+--------------+--------------+----------> END

 SECTIONS
   1  Setup and configuration
   2  Rate limiting
   3  LLM access
   4  Prompts
   5  Crisis detection
   6  Fact extraction (memory)
   7  Retrieval
   8  State
   9  Helpers
  10  Guard nodes
  11  Routing
  12  Memory and retrieval nodes
  13  Specialists and supervisor
  14  Compose and critic
  15  Terminal nodes
  16  Graph
  17  Public API
  18  Web layer
=============================================================================
"""

# =============================================================================
# 1. SETUP AND CONFIGURATION
# =============================================================================
from __future__ import annotations

import atexit
import hashlib
import hmac
import json
import logging
import math
import operator
import os
import re
import sqlite3
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from groq import Groq

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv(override=True)          # .env wins over a stale shell variable

HERE = Path(__file__).resolve().parent

log = logging.getLogger("griefbot")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)

client = Groq(api_key=os.getenv("GROQ"))

BIG_MODEL = os.getenv("BIG_MODEL", "openai/gpt-oss-120b")     # writes replies
SMALL_MODEL = os.getenv("SMALL_MODEL", "openai/gpt-oss-20b")  # classifies

# Guards try these in order. Different models have separate rate-limit buckets,
# so a throttled small model falls through to the big one.
GUARD_MODELS = [SMALL_MODEL, BIG_MODEL]

# max_tokens counts in full against Groq's tokens-per-minute budget, not just
# what the model returns. gpt-oss needs headroom for its reasoning phase, but
# 1024 per guard burned the free-tier minute budget in a single turn.
GUARD_MAX_TOKENS = int(os.getenv("GUARD_MAX_TOKENS", "512"))
REPLY_MAX_TOKENS = int(os.getenv("REPLY_MAX_TOKENS", "500"))

KEEP_MESSAGES = 6        # verbatim transcript sent to the model
MAX_HOPS = 2             # supervisor delegation ceiling
MAX_CRITIQUE_ROUNDS = 1  # rewrite ceiling
REQUEST_TIMEOUT_S = 30

DB_PATH = HERE / "griefbot_memory.db"
PDF_DIR = HERE / "pdfs"


# =============================================================================
# 2. RATE LIMITING
# =============================================================================
# Groq's free tier caps tokens per minute. Reacting to 429s works but wastes a
# round trip and makes latency spiky. A client-side budget is better: we know
# roughly what each call will cost before we make it, so we can wait a moment
# instead of being rejected.

class TokenBudget:
    """Sliding-window throttle. Thread-safe because Flask serves each request
    on its own thread and LangGraph runs parallel nodes in a thread pool."""

    def __init__(self, tokens_per_minute: int, window_s: float = 60.0):
        self.limit = tokens_per_minute
        self.window = window_s
        self._events: deque = deque()      # (timestamp, tokens)
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0][0] > self.window:
            self._events.popleft()

    def used(self) -> int:
        with self._lock:
            self._prune(time.monotonic())
            return sum(t for _, t in self._events)

    def acquire(self, tokens: int, max_wait_s: float = 30.0) -> None:
        """Block until there is room in the window, then record the spend."""
        deadline = time.monotonic() + max_wait_s
        while True:
            with self._lock:
                now = time.monotonic()
                self._prune(now)
                spent = sum(t for _, t in self._events)
                if spent + tokens <= self.limit or not self._events:
                    self._events.append((now, tokens))
                    return
                oldest = self._events[0][0]
                wait = max(0.05, self.window - (now - oldest) + 0.05)
            if time.monotonic() + wait > deadline:
                # Don't stall a user forever. Let the call through and let the
                # provider's own 429 handling take over.
                log.warning("token budget wait exceeded %.0fs, proceeding", max_wait_s)
                with self._lock:
                    self._events.append((time.monotonic(), tokens))
                return
            log.debug("token budget full, waiting %.1fs", wait)
            time.sleep(wait)


# Slightly under the real limit, so we throttle before the provider does.
TPM_LIMIT = int(os.getenv("TPM_LIMIT", "7000"))
BUDGET = TokenBudget(TPM_LIMIT)


def estimate_tokens(system: str, user: str, max_tokens: int) -> int:
    """Groq bills 'requested' tokens as prompt + max_tokens, so estimate both.
    Roughly four characters per token for English."""
    return (len(system) + len(user)) // 4 + max_tokens


# =============================================================================
# 3. LLM ACCESS
# =============================================================================
# Two functions, because the jobs are different:
#   call()  -> classification. Small model, one isolated question, temperature 0.
#   ask()   -> writing. Big model, conversation history, some warmth.

RETRY_DELAY_RE = re.compile(r"try again in ([\d.]+)s", re.IGNORECASE)


def _is_rate_limit(err: Exception) -> bool:
    s = str(err)
    return "429" in s or "rate_limit" in s.lower()


def _is_auth_error(err: Exception) -> bool:
    s = str(err)
    return "401" in s or "invalid_api_key" in s.lower()


def _retry_after(err: Exception, attempt: int) -> float:
    """Groq's 429 states exactly how long to wait. Prefer that over guessing."""
    m = RETRY_DELAY_RE.search(str(err))
    if m:
        return min(float(m.group(1)) + 0.25, 15.0)
    return min(0.5 * (2 ** attempt), 8.0)


def _extract_text(resp) -> str:
    """gpt-oss models reason before answering. If the budget goes into thinking,
    `content` is empty and the reasoning lands in a separate field."""
    msg = resp.choices[0].message
    text = (msg.content or "").strip()
    if not text:
        text = (getattr(msg, "reasoning", None) or "").strip()
    return text


def _chat(model: str, messages: List[Dict[str, str]], *,
          temperature: float, max_tokens: int) -> str:
    prompt_chars = sum(len(m.get("content") or "") for m in messages)
    BUDGET.acquire(prompt_chars // 4 + max_tokens)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=REQUEST_TIMEOUT_S,
    )
    return _extract_text(resp)


def _with_retry(models: List[str], messages, *, temperature, max_tokens,
                label: str) -> str:
    last_err: Optional[Exception] = None
    for model in models:
        for attempt in range(3):
            try:
                text = _chat(model, messages,
                             temperature=temperature, max_tokens=max_tokens)
                record_model_success()
                return text
            except Exception as e:                      # noqa: BLE001
                last_err = e
                if _is_rate_limit(e):
                    # A 429 is explicitly transient. Treating it as a failure
                    # means backpressure silently disables a safety guard.
                    delay = _retry_after(e, attempt)
                    log.warning("%s rate limited on %s, waiting %.1fs",
                                label, model, delay)
                    time.sleep(delay)
                    continue
                if _is_auth_error(e):
                    break                               # no point retrying
                log.warning("%s failed on %s: %s", label, model, e)
                break                                   # try the next model
    raise last_err or RuntimeError(f"{label}: all models failed")


def call(system: str, user: str, max_tokens: int = None) -> str:
    """One classification question. Deterministic."""
    return _with_retry(
        GUARD_MODELS,
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        temperature=0,
        max_tokens=max_tokens or GUARD_MAX_TOKENS,
        label="call",
    )


def ask(messages: List[Dict[str, str]], system: str = None,
        max_tokens: int = None) -> str:
    """Write a reply. Takes conversation history plus a system prompt."""
    return _with_retry(
        [BIG_MODEL, SMALL_MODEL],
        [{"role": "system", "content": system or SUPPORT_SYSTEM}] + list(messages),
        temperature=0.6,
        max_tokens=max_tokens or REPLY_MAX_TOKENS,
        label="ask",
    )


def clean(t: str) -> str:
    """Strip the <think>...</think> block some models emit."""
    return re.sub(r"<think>.*?</think>", "", t or "", flags=re.DOTALL).strip()


def last_label(text: str, labels: set, default: str) -> str:
    """Take the LAST matching label in the output.

    These models think out loud, and the thinking mentions every option:
        "We need to decide if this is unsafe or safe. ... safe"
    A first-match search returns "unsafe" and refuses a grieving user.
    The real verdict is always last.
    """
    words = re.findall(r"[a-z]+", (text or "").lower())
    for w in reversed(words):
        if w in labels:
            return w
    return default


# --- degraded mode -----------------------------------------------------------
# A live run with a dead key routed EVERY message to crisis, because the crisis
# guard failed closed on each exception. Failing closed is right when a check
# RAN and was ambiguous; it is wrong when the check COULD NOT RUN. And the
# lexical tier needs no network, so recall survived the outage anyway.
_FAILURE_STREAK = 0
DEGRADED_AFTER = 3


def record_model_failure(err: Exception) -> bool:
    global _FAILURE_STREAK
    _FAILURE_STREAK += 1
    if _FAILURE_STREAK == DEGRADED_AFTER:
        log.critical("model tier down after %d failures (%s). "
                     "Falling back to deterministic checks only.",
                     DEGRADED_AFTER, type(err).__name__)
    return _FAILURE_STREAK >= DEGRADED_AFTER


def record_model_success() -> None:
    global _FAILURE_STREAK
    if _FAILURE_STREAK >= DEGRADED_AFTER:
        log.info("model tier recovered")
    _FAILURE_STREAK = 0


def is_degraded() -> bool:
    return _FAILURE_STREAK >= DEGRADED_AFTER


# =============================================================================
# 4. PROMPTS
# =============================================================================
# Structure: stable identity first (providers cache on prefix match), then
# retrieved context, then memory, then the turn-specific instruction.

SPECIALIST_BASE = """You are a warm, caring human companion in a grief-support
service for a UK user.

Anything you are told about this person's loss was told to you BY THEM, in this
conversation. It is their own information. Answer questions about it directly
and never refuse to repeat something they told you themselves.

Do not diagnose, and do not give medical advice.

Sound like a person who genuinely cares, not a form. Be warm, gentle and human.
It is good to acknowledge feelings and to show you are moved. Vary how you open
- do not start every reply the same way - and never mechanically parrot the
person's own words back at them ("You said you feel...").

Avoid these, which ring hollow or hurt:
- a reflex scripted apology on every single turn ("I'm so sorry for your loss"
  used as a formula)
- minimising or hurrying grief: "time heals", "a better place", "at least...",
  "stay strong", "be strong", "everything happens for a reason"
- claiming to know exactly how they feel
- afterlife talk ("she's watching over you") they did not raise themselves
- advice, suggestions or services the person did not ask for

Warmth is welcome; clichés are not. Speak simply, and from the heart.
"""

SUPPORT_SYSTEM = SPECIALIST_BASE + """
Short, warm, human sentences. No lists, no headings. Two to four sentences.
Where it fits naturally, gently invite the person to share a little more.
"""

EMOTIONAL_PROMPT = SPECIALIST_BASE + """
You are the emotional-support specialist.

Respond warmly and naturally, the way someone who truly cares would - present,
gentle, unhurried. Two to four short sentences. No advice, no logistics, no
services; stay with the feelings.

If the person is only saying hello, or the conversation is just beginning,
welcome them warmly, let them know this is a safe space to talk about whatever
they are carrying, and gently invite them to share what is on their mind.

Unless the pacing note below tells you otherwise, end with a gentle, open
invitation to say more, or one soft question - something that helps them keep
talking. Never an interrogation, never more than one question.

{pacing}

What we remember about them:
{facts}

Background you may draw on (do not quote or cite it):
{context}
"""

COPING_PROMPT = SPECIALIST_BASE + """
You are the coping-and-wellbeing specialist. The person has ASKED for
suggestions, so giving them is right.

Offer two or three concrete, gentle things many bereaved people find help.
Ordinary ground: sleep and eating routines, getting outside, keeping something
of theirs close, writing to them, telling one person how you actually are,
letting the waves come rather than fighting them, care with alcohol, planning
gently for hard dates.

Rules:
- Suggest, never instruct. "Some people find..." not "You should..."
- Say plainly there is no correct way and no timetable.
- No stages of grief. No "closure". No promise it improves by a date.
- If they say they are not coping at all, a GP or bereavement counsellor is a
  reasonable next step - nothing stronger.
- Prose, not a numbered list. Four or five sentences.

What we remember about them:
{facts}

Background you may draw on:
{context}
"""

PRACTICAL_PROMPT = SPECIALIST_BASE + """
You are the practical-guidance specialist for UK bereavement.

Answer ONLY the logistical part, in three or four plain sentences. Be concrete
about UK processes - a death must be registered within 5 days in England and
Wales, the Tell Us Once service, probate - but say clearly when something
depends on circumstances and should be checked with the registrar or a
solicitor. Never invent fees, phone numbers or deadlines you are unsure of.

Background you may draw on:
{context}
"""

RESOURCES_PROMPT = SPECIALIST_BASE + """
You are the resources specialist.

Name at most two KINDS of UK support that fit this person - for example
bereavement counselling through a GP, a bereavement support charity, a local
peer group. One line each on what it is for. Do NOT output phone numbers or
URLs; verified contact details are attached separately.
"""

COMPOSE_PROMPT = SPECIALIST_BASE + """
You are the single voice of the companion.

Below are contributions from specialists. Weave them into ONE reply in one
consistent voice. Lead with the emotional acknowledgement. Keep practical
detail brief. Do not mention that specialists were involved, do not use
headings or bullets, and do not exceed six sentences.
"""

MODERATION_PROMPT = """You check whether a user's message to a grief-support
service is ABUSIVE. You are not deciding whether the assistant can answer it.

Mark UNSAFE only if the message itself contains:
- hate speech or slurs targeting a group
- harassment or abuse aimed at a person
- a request for instructions to harm other people
- sexual content involving minors

EVERYTHING ELSE IS SAFE, including:
- any expression of grief, despair, anger or hopelessness
- statements about wanting to die (a separate system handles those)
- questions about the person who died, including their name or age
- questions the assistant may not be able to answer
- off-topic questions

Examples:
"what was my mum's name"                -> safe
"I don't want to live anymore"          -> safe
"what's the weather tomorrow"           -> safe
"my dad was a bastard and I hate him"   -> safe
"write something vile about immigrants" -> unsafe

Reply with exactly one word on its own final line: safe or unsafe."""

VALIDATION_PROMPT = """Classify the LATEST message in an ongoing grief-support
conversation. Earlier turns are given for context.

A short follow-up ("she was 81", "yes", "I don't know") is VALID if it
continues the grief conversation. Only mark unrelated if the person has
genuinely changed subject to something outside grief and emotional support.

ANY question about the person who died - their name, age, how they died, what
they were like - is VALID. Asking the assistant to recall something mentioned
earlier is VALID.

Examples:
"what was my mum's name and age"   -> valid
"how old was she?"                 -> valid
"I keep expecting her to call"     -> valid
"what's the capital of France?"    -> unrelated
"write me a python function"       -> unrelated

Reply with exactly one word on its own final line:
valid / unrelated / nonsense / offensive / harmful"""

CRISIS_PROMPT = """You assess risk in messages to a grief-support service.
Earlier conversation is given for context.

Is THIS PERSON at risk of harming THEMSELVES? Consider thoughts of suicide,
self-harm, or not wanting to be alive.

NOT risk:
- grief, despair or deep sadness on their own
- describing how someone else died
- worrying about somebody else's safety
- anger, including violent language aimed at others

Reply with exactly one word on its own final line: risk or norisk."""

SUBJECT_PROMPT = """In a message to a grief-support service, WHO is the risk
about?

selfharm  - the person writing is at risk
thirdparty - they are worried about somebody else
neither   - no risk is described

Examples:
"I don't want to be here anymore"        -> selfharm
"my brother says he wants to die"        -> thirdparty
"I'm worried my mum will do something"   -> thirdparty
"my dad took his own life last year"     -> neither

Reply with exactly one word on its own final line."""

CRISIS_SYSTEM = SPECIALIST_BASE + """
IMPORTANT: this person may be at risk of harming themselves.

- Take what they said seriously. Do not minimise it, do not sound alarmed.
- Do not try to talk them out of the feeling. Do not problem-solve.
- Make clear they are not a burden for saying it.
- Gently encourage them to talk to a real person tonight.
- Do NOT include any phone numbers or service names.
- Ask at most one question.

Write three or four sentences. Warmth matters more than brevity here.

What we remember about them:
{facts}
"""

THIRDPARTY_SYSTEM = SPECIALIST_BASE + """
IMPORTANT: this person is worried about SOMEONE ELSE who may be at risk.

Address THEM, not the person they are worried about. They are frightened and
carrying something heavy.

- Acknowledge how hard it is to be the one who notices.
- Say plainly that asking someone directly whether they are thinking of suicide
  does not plant the idea - it is one of the most useful things they can do.
- Encourage them to stay alongside rather than fix it.
- Remind them they are allowed to get support for themselves too.
- Do NOT include phone numbers; they are attached separately.

Write four or five sentences.
"""

CRITIC_PROMPT = """You review DRAFT REPLIES from a grief-support companion
before they are sent.

You are shown the user's message and the draft. Judge the draft only. Do not
answer it, do not refuse it, and do not comment on whether the information in
it should be shared - it is the user's own information.

Be proportionate. A reply that is warm, short, and does not claim to know how
they feel should PASS. Only fail it for a real problem, not for style you would
have written differently.

IMPORTANT: if the user ASKED for tips, advice or ways to cope, advice in the
draft is correct and must NOT be failed for that reason.

FAIL only if the draft:
- gives advice the user did not ask for
- uses a platitude: "time heals", "a better place", "at least...", "stay strong"
- claims to know how they feel
- rushes them toward acceptance, closure or moving on
- exceeds six sentences
- asks more than one question
- states a fact about their loss that was not given to it
- opens with "I'm sorry for your loss" as a formula

End your response with exactly one word on its own final line: pass or fail.
If fail, put one short sentence before it saying what to change."""

EXTRACT_PROMPT = """Extract durable facts from a message to a grief-support
service. Return ONLY a JSON object.

Keys (omit any key you have no explicit evidence for - never guess):
  deceased_name    the NAME of the person who died. A first name is enough.
  relationship     their relation to the user (mother, wife, friend, dog)
  age              age at death
  cause            cause of death
  time_since_loss  how long ago
  key_dates        funeral, birthday or anniversary dates mentioned
  user_name        the user's own name
  support          people or services they say they have
  helps            things they say help them

Pay particular attention to NAMES. "I lost my wife Priya" means
deceased_name is "Priya" and relationship is "wife". "My mum Margaret died"
means deceased_name is "Margaret" and relationship is "mother".

If there are no new facts, return {}."""

# Contact details live HERE, in the repo. A model that invents a plausible
# helpline number for someone in crisis at 2am is the worst failure this
# product has.
HELPLINES = """

If things get heavier tonight, these people are there right now:
- Samaritans - 116 123 (free, 24/7)
- Shout - text SHOUT to 85258
- Cruse Bereavement Support - 0808 808 1677
If you are in immediate danger, please call 999."""

THIRDPARTY_HELPLINES = """

These are for you as much as for them:
- Samaritans - 116 123 (free, 24/7, and they take calls from people worried
  about someone else)
- Papyrus HOPELINE247 - 0800 068 4141 (if the person you are worried about is
  under 35)
- If there is immediate danger, call 999."""


# =============================================================================
# 5. CRISIS DETECTION
# =============================================================================
# Three tiers, cheapest first, combined with OR:
#   0  lexical tripwire   ~0ms, no network
#   1  subject classifier (is the risk about them, or someone else?)
#   2  LLM risk assessment
#
# OR, not AND. We accept false alarms so we never miss a real one. Showing a
# helpline to someone who did not need it is an awkward moment. Missing someone
# who did is unrecoverable. Unequal costs, unequal thresholds.

CRISIS_WORDS = re.compile(
    r"\bkill(?:ing)? myself\b"
    r"|\bend(?:ing)? (?:my life|it all|things|my own life)\b"
    r"|\btak(?:e|ing) my own life\b"
    r"|\bdon'?t want to (?:live|be here|go on|carry on|exist|wake up)\b"
    r"|\bwant(?:s|ing)? to (?:die|disappear|not exist|not be here)\b"
    r"|\bbetter off (?:dead|without me)\b"
    r"|\b(?:no|what'?s the|there'?s no) (?:point|reason)\b[^.?!]{0,25}"
    r"\b(?:living|life|go(?:ing)? on|carry(?:ing)? on|being here)\b"
    r"|\bsuicid(?:e|al)\b"
    r"|\bself[- ]harm(?:ing|ed)?\b"
    r"|\bhurt(?:ing)? myself\b"
    r"|\bcut(?:ting)? myself\b"
    r"|\boverdos(?:e|ing)\b"
    r"|\bcan'?t (?:go on|carry on|do this any ?more|keep going)\b"
    r"|\bwish(?:ed)? I (?:was|were) (?:dead|gone|not here)\b"
    r"|\bnot (?:be|being) (?:around|here) any ?more\b",
    re.IGNORECASE,
)

# The counterweight. Without it, "this grief is killing me" triggers - and a bot
# that shows a suicide helpline whenever someone says they're tired teaches
# people to ignore it, which is its own safety failure.
IDIOMS = re.compile(
    r"\bkilling me\b|\bdying to (?:know|see|hear|meet)\b"
    r"|\bdied (?:of|from) (?:laughter|embarrassment)\b",
    re.IGNORECASE,
)

# These always win, whatever idiom surrounds them.
UNAMBIGUOUS = re.compile(
    r"\bkill(?:ing)? myself\b|\bsuicid(?:e|al)\b|\bend(?:ing)? my (?:own )?life\b"
    r"|\btak(?:e|ing) my own life\b|\bhurt(?:ing)? myself\b|\bself[- ]harm",
    re.IGNORECASE,
)

# Someone else is the subject. "my brother wants to die" is a different problem
# from "I want to die", and the reply has to address a different person.
THIRD_PARTY = re.compile(
    r"\b(my|his|her|their|a)\s+(mum|mom|mother|dad|father|wife|husband|partner|"
    r"son|daughter|brother|sister|friend|mate|colleague|nan|gran|grandad|"
    r"child|kid|boy|girl|neighbour|neighbor)\b[^.?!]{0,40}"
    r"\b(wants? to die|wants? to end|said|says|told me|might|is going to|"
    r"talking about|threatening)\b"
    r"|\b(?:i'?m |i am )?(?:worried|scared|frightened) (?:about|for) "
    r"(?:my|his|her|their|him|her|them)\b",
    re.IGNORECASE,
)


def lexical_crisis(text: str) -> bool:
    """High-recall tripwire. Unambiguous intent bypasses idiom suppression
    entirely, so no idiom list can ever mask a clear disclosure."""
    t = text or ""
    if UNAMBIGUOUS.search(t):
        return True
    if not CRISIS_WORDS.search(t):
        return False
    return not IDIOMS.search(t)


# Reflexive language is inherently first-person: "kill myself" can only be
# about the speaker. The ambiguous phrases ("can't go on", "wants to die") can
# describe anyone, which is why subject resolution has to be separate from
# risk detection.
FIRST_PERSON_RISK = re.compile(
    r"\bmyself\b"
    r"|\bi\b[^.?!]{0,30}\b(?:want|don'?t want|can'?t go on|can'?t carry on|"
    r"wish i|am done|give up)\b"
    r"|\b(?:i'?m|i am)\b[^.?!]{0,30}\b(?:done|finished|giving up)\b",
    re.IGNORECASE,
)


def lexical_third_party(text: str) -> bool:
    return bool(THIRD_PARTY.search(text or ""))


def resolve_subject(text: str) -> str:
    """Who is the risk about? Returns selfharm | thirdparty | neither.

    Without this, "my brother says he wants to die" trips the crisis detector
    and then the reply is addressed to the wrong person - it tells someone
    frightened about their brother that THEY are not a burden.
    """
    t = text or ""
    # Reflexive intent always wins, even alongside worry about someone else.
    if UNAMBIGUOUS.search(t) or FIRST_PERSON_RISK.search(t):
        return "selfharm"
    if THIRD_PARTY.search(t):
        return "thirdparty"
    if CRISIS_WORDS.search(t):
        return "selfharm"
    return "neither"


# =============================================================================
# 6. FACT EXTRACTION (memory)
# =============================================================================
# The original bot extracted open-ended triples on every turn into a knowledge
# graph that was never read. Every message paid latency, tokens and a failure
# surface for data nobody queried.
#
# This version: a closed schema, extracted conditionally, and actually read back
# into the prompt. Memory you never read is not memory.

_NUM = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|few)"

FACT_HINTS = re.compile(
    r"\b(my|his|her|their|our)\s+(mum|mom|mother|dad|father|wife|husband|"
    r"partner|son|daughter|brother|sister|nan|nana|gran|grandma|grandad|"
    r"grandmother|grandfather|friend|dog|cat|pet|baby|child)\b"
    r"|\b(?:called|named)\s+[A-Z]\w+"
    r"|\bwas\s+\d{1,3}\b|\b\d{1,3}\s*(?:years?\s*old|yo)\b"
    r"|\b(died|passed away|passed|lost|funeral|cremation|burial|cancer|stroke|"
    r"heart attack|accident|suicide|overdose|illness)\b"
    r"|\b(last|this)\s+(week|month|year|spring|summer|autumn|winter|christmas)\b"
    rf"|\b{_NUM}\s*(?:days?|weeks?|months?|years?)\s+ago\b"
    r"|\bmy name is\b|\bi'?m\s+[A-Z]\w+",
    re.IGNORECASE,
)

# A capitalised word right after a relationship term is almost always the name.
# The model misses these sometimes; this is a deterministic backstop.
# Scoped (?i:...) so the prefix matches "My nan" as well as "my nan", while the
# NAME group still requires a capital. A blanket re.IGNORECASE here would match
# "my mum died" and record "died" as her name.
NAME_AFTER_RELATION = re.compile(
    r"\b(?i:my|our)\s+(?i:(mum|mom|mother|dad|father|wife|husband|partner|son|"
    r"daughter|brother|sister|nan|nana|gran|grandma|grandad|grandmother|"
    r"grandfather|friend|dog|cat))\s*,?\s+([A-Z][a-z]{1,20})\b"
)
NAME_AFTER_CALLED = re.compile(r"\b(?i:called|named)\s+([A-Z][a-z]{1,20})\b")

RELATION_CANON = {
    "mum": "mother", "mom": "mother", "mother": "mother",
    "dad": "father", "father": "father",
    "wife": "wife", "husband": "husband", "partner": "partner",
    "son": "son", "daughter": "daughter",
    "brother": "brother", "sister": "sister",
    "nan": "grandmother", "nana": "grandmother", "gran": "grandmother",
    "grandma": "grandmother", "grandmother": "grandmother",
    "grandad": "grandfather", "grandfather": "grandfather",
    "friend": "friend", "dog": "dog", "cat": "cat",
}

FACT_KEYS = {"deceased_name", "relationship", "age", "cause", "time_since_loss",
             "key_dates", "user_name", "support", "helps"}

PRETTY = {"deceased_name": "Who died", "relationship": "Their relationship to the user",
          "age": "Age at death", "cause": "Cause of death",
          "time_since_loss": "Time since the loss", "key_dates": "Important dates",
          "user_name": "User's name", "support": "Support around them",
          "helps": "What helps them"}

# Words the model sometimes returns as a "name" that plainly aren't.
NOT_NAMES = {"mum", "mom", "mother", "dad", "father", "wife", "husband",
             "partner", "unknown", "none", "n/a", "na", "null", "her", "him",
             "she", "he", "they", "nan"}


def might_have_facts(text: str) -> bool:
    """Cheap pre-filter. Most messages carry no new facts ('yeah', 'it's just
    hard'), and paying a model to confirm that is waste. Skips roughly 60% of
    extraction calls for the cost of one regex."""
    return bool(FACT_HINTS.search(text or ""))


def lexical_facts(text: str) -> Dict[str, str]:
    """Deterministic backstop for the field the model most often drops.

    A live run showed 'I lost my wife Priya in the spring' extracting the
    timeframe but NOT the name - the single most important field in the whole
    schema. Same principle as the crisis tier: where being wrong is costly, put
    a plain rule next to the model.
    """
    out: Dict[str, str] = {}
    t = text or ""

    m = NAME_AFTER_RELATION.search(t)
    if m:
        rel, name = m.group(1).lower(), m.group(2)
        if name.lower() not in NOT_NAMES:
            out["deceased_name"] = name
            out["relationship"] = RELATION_CANON.get(rel, rel)
    else:
        m2 = NAME_AFTER_CALLED.search(t)
        if m2 and m2.group(1).lower() not in NOT_NAMES:
            out["deceased_name"] = m2.group(1)

    if "relationship" not in out:
        for word, canon in RELATION_CANON.items():
            if re.search(rf"\b(?:my|our)\s+{word}\b", t, re.IGNORECASE):
                out["relationship"] = canon
                break

    m3 = re.search(r"\b(?:was|aged)\s+(\d{1,3})\b|\b(\d{1,3})\s*years?\s*old\b",
                   t, re.IGNORECASE)
    if m3:
        age = m3.group(1) or m3.group(2)
        if 0 < int(age) < 120:
            out["age"] = age
    return out


def parse_json_object(raw: str) -> dict:
    """Models wrap JSON in prose or code fences. Dig the object out."""
    body = re.sub(r"^```(?:json)?|```$", "", clean(raw), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", body, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:                                   # noqa: BLE001
        return {}


def sanitise_facts(raw: dict) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (raw or {}).items():
        if k not in FACT_KEYS:
            continue
        if v in (None, "", [], {}):
            continue
        s = str(v).strip()
        if s.lower() in {"unknown", "n/a", "na", "null", "none", "not mentioned"}:
            continue
        if k == "deceased_name" and (s.lower() in NOT_NAMES or len(s) > 40):
            continue
        if k == "age":
            digits = re.search(r"\d{1,3}", s)
            if not digits or not (0 < int(digits.group()) < 120):
                continue
            s = digits.group()
        out[k] = s
    return out


def format_facts(facts: Dict[str, Any]) -> str:
    """Render memory for the prompt. THIS is the line the original bot was
    missing - it wrote to a knowledge graph and never read it back."""
    if not facts:
        return "(nothing remembered yet)"
    return "\n".join(f"- {PRETTY.get(k, k)}: {v}" for k, v in facts.items() if v)


# =============================================================================
# 7. RETRIEVAL
# =============================================================================
# The rebuild dropped the original bot's document search, so it was answering
# purely from the model. This restores grounding.
#
# BM25 (keyword) + optional dense vectors (semantic), fused with Reciprocal
# Rank Fusion. RRF uses RANK, not score: BM25 scores are unbounded and
# corpus-dependent while cosine sits in a fixed range, so they are not
# comparable and cannot simply be averaged.

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "that",
         "this", "for", "on", "with", "as", "was", "were", "be", "been", "are",
         "i", "you", "my", "your", "at", "by", "from", "but", "not", "can"}


def tokenize(text: str) -> List[str]:
    return [t for t in (w.lower() for w in _TOKEN_RE.findall(text or ""))
            if t not in _STOP]


class BM25:
    """Okapi BM25.

    k1 controls term-frequency saturation - the 10th occurrence of a word adds
    far less than the 2nd. b controls length normalisation. 1.5 / 0.75 are the
    standard defaults, and the ones Azure AI Search uses.

    This is why BM25 beats plain TF-IDF, which has neither.
    """

    def __init__(self, chunks: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus = [tokenize(c) for c in chunks]
        self.N = len(self.corpus)
        self.doc_len = [len(c) for c in self.corpus]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        df: Dict[str, int] = {}
        self.tf: List[Dict[str, int]] = []
        for toks in self.corpus:
            counts: Dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)
            for t in counts:
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
                    for t, n in df.items()}

    def search(self, query: str, k: int = 12) -> List[tuple]:
        q = tokenize(query)
        if not q or not self.N:
            return []
        scores = [0.0] * self.N
        for term in q:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, counts in enumerate(self.tf):
                f = counts.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (
                    1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        ranked = sorted(range(self.N), key=lambda i: -scores[i])[:k]
        return [(i, scores[i]) for i in ranked if scores[i] > 0]


class DenseIndex:
    """Sentence-transformer embeddings. Optional: if the package is missing we
    degrade to keyword-only rather than failing. Graceful degradation of a
    retrieval tier is a feature, not a compromise."""

    def __init__(self, chunks: List[str]):
        self.available = False
        self._model = None
        self._matrix = None
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
            self._np = np
            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self._matrix = np.asarray(
                self._model.encode(chunks, normalize_embeddings=True),
                dtype="float32")
            self.available = True
            log.info("dense index ready: %s vectors", self._matrix.shape[0])
        except Exception as e:                          # noqa: BLE001
            log.info("dense retrieval unavailable (%s) - keyword-only", type(e).__name__)

    def search(self, query: str, k: int = 12) -> List[tuple]:
        if not self.available:
            return []
        qv = self._model.encode([query], normalize_embeddings=True)[0]
        sims = self._matrix @ self._np.asarray(qv, dtype="float32")
        order = self._np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order]


def reciprocal_rank_fusion(rankings, k_const: int = 60, top_k: int = 3):
    """score(d) = sum over lists of 1 / (k + rank(d)).  k = 60 is the value from
    the original paper and the Azure AI Search default."""
    fused: Dict[int, float] = {}
    for ranking in rankings:
        for rank, (idx, _score) in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k_const + rank)
    return sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]


FALLBACK_CORPUS = """
Grief has no fixed timeline and no correct order. Waves of sadness can arrive
long after the loss, often triggered by ordinary things: a song, a smell, an
empty chair at a table. There is no stage you are supposed to have reached.

When supporting someone who is grieving, listening matters more than fixing.
Acknowledge the feeling before offering anything else. Avoid phrases that
minimise the loss, such as saying it was for the best or that they should be
over it by now.

Anniversaries, birthdays and holidays are commonly the hardest days. Planning
gently for them in advance can help - deciding in advance what you will do, and
giving yourself permission to change your mind on the day.

Grief affects sleep, appetite and concentration. These physical effects are
normal. If they persist and interfere with daily life, support from a GP or a
bereavement counsellor is appropriate.

Children grieve differently from adults, often in short bursts, moving between
distress and play. This is normal and is not a sign they are unaffected. Use
plain words: "died", not "passed away" or "lost", which children take literally.

Sorting belongings has no deadline. Many people find it easier to keep one or
two things that carry the person's presence, and to do the rest slowly, with
someone else in the house.

Bereavement by suicide carries additional weight: stigma, guilt, and repeated
unanswerable questions about why. Support specifically for this exists and is
different from general bereavement support.

Registering a death in England and Wales must normally happen within five days.
The Tell Us Once service lets you report a death to most government departments
in one go. Probate may be needed before an estate can be distributed.
"""


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> List[str]:
    """Paragraph-aware chunking.

    Character-count splitting cuts sentences and pronouns lose their referents,
    which makes a passage useless on its own. The original bot used 400/100,
    far too small for prose. 900/150 keeps a coherent thought together.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], ""
    for p in paras:
        if len(current) + len(p) + 2 <= size:
            current = f"{current}\n\n{p}" if current else p
        else:
            if current:
                chunks.append(current)
                tail = current[-overlap:] if overlap else ""
                current = f"{tail}\n\n{p}" if tail else p
            else:
                chunks.append(p[:size])
                current = p[size - overlap:]
    if current:
        chunks.append(current)
    return [c for c in chunks if len(c.strip()) > 40]


def load_corpus() -> List[str]:
    texts: List[str] = []
    pdfs = sorted(PDF_DIR.glob("*.pdf")) if PDF_DIR.exists() else []
    for pdf in pdfs:
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(pdf))
            body = "\n\n".join((page.extract_text() or "") for page in reader.pages)
            if body.strip():
                texts.append(body)
                log.info("loaded corpus: %s (%d pages)", pdf.name, len(reader.pages))
        except Exception as e:                          # noqa: BLE001
            log.warning("could not read %s: %s", pdf.name, e)
    if not texts:
        log.info("no PDFs found in %s - using built-in corpus", PDF_DIR)
        texts = [FALLBACK_CORPUS]
    chunks: List[str] = []
    for t in texts:
        chunks.extend(chunk_text(t))
    log.info("corpus indexed: %d chunks", len(chunks))
    return chunks


class Retriever:
    def __init__(self):
        self.chunks = load_corpus()
        self.bm25 = BM25(self.chunks)
        self.dense = DenseIndex(self.chunks)

    def search(self, query: str, k: int = 3) -> List[str]:
        # Over-fetch from each tier before fusing: a chunk ranked 5th by both
        # can fuse into the top 2.
        rankings = [r for r in (self.bm25.search(query, k * 4),
                                self.dense.search(query, k * 4)) if r]
        if not rankings:
            return []
        return [self.chunks[i] for i, _ in
                reciprocal_rank_fusion(rankings, top_k=k)]


RETRIEVER: Optional[Retriever] = None


def get_retriever() -> Retriever:
    global RETRIEVER
    if RETRIEVER is None:
        RETRIEVER = Retriever()
    return RETRIEVER


SMALLTALK = {"hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "bye",
             "yeah", "yes", "no", "sure", "cheers"}


def needs_retrieval(query: str) -> bool:
    """Retrieving for 'hi' costs tokens and actively degrades the answer by
    stuffing irrelevant passages into the prompt."""
    q = (query or "").strip().lower().strip("?!. ")
    if q in SMALLTALK:
        return False
    return len(q.split()) > 2


def format_context(chunks: List[str]) -> str:
    if not chunks:
        return "(no reference material for this turn)"
    return "\n\n".join(f"[{i}] {c.strip()}" for i, c in enumerate(chunks, 1))


# =============================================================================
# 8. STATE
# =============================================================================
# Every node receives the whole state and returns a PARTIAL UPDATE - a dict
# containing only the keys it changed. LangGraph merges it using the reducers
# declared below.
#
# A reducer is how concurrent writes to the same key are combined. Three guards
# write `guards` at the same time; without a reducer LangGraph raises
# InvalidUpdateError rather than silently dropping two of three results.

def merge_guards(old, new):
    """operator.add would append forever, and the checkpointer carries state
    into the next turn - so turn 1's guards were still present on turn 5 and the
    gate was reading stale results. This resets at the start of each turn."""
    if new == ["__RESET__"]:
        return []
    return (old or []) + (new or [])


def merge_facts(old, new):
    """New values win; an empty value never erases a known one. This is why
    memory can be permanent at a fixed token cost - the dict is updated, not
    re-accumulated."""
    merged = dict(old or {})
    for k, v in (new or {}).items():
        if v not in (None, "", [], {}):
            merged[k] = v
    return merged


class State(TypedDict, total=False):
    # --- input ---
    query: str
    session_id: str

    # add_messages APPENDS, which is what accumulates the transcript
    messages: Annotated[List[BaseMessage], add_messages]

    # three concurrent writers
    guards: Annotated[List[Dict[str, Any]], merge_guards]

    # --- guard findings (single writer each, so no reducer: last wins) ---
    verdict: str
    crisis: bool
    crisis_certain: bool     # True only when the DETERMINISTIC tier fired
    subject: str             # selfharm | thirdparty | neither
    route: str

    # --- working data ---
    facts: Annotated[Dict[str, Any], merge_facts]
    context: List[str]
    contributions: Annotated[List[Dict[str, str]], merge_guards]
    hops: int
    next_agent: str

    # --- output ---
    draft: str
    critique: str
    attempts: int
    answer: str
    escalated: bool


# =============================================================================
# 9. HELPERS
# =============================================================================
def to_groq_messages(history: List[BaseMessage]) -> List[Dict[str, str]]:
    """LangGraph stores message objects; Groq wants plain dicts."""
    return [{"role": "assistant" if isinstance(m, AIMessage) else "user",
             "content": m.content}
            for m in history if (m.content or "").strip()]


def recent_context(state: State, n: int = 4) -> str:
    """Recent turns plus the current message, for the guards.

    Guards judging a message in isolation reject every natural follow-up -
    "how old was she?" is not about grief on its own. Deliberately a SMALL
    window: they only need enough to resolve a pronoun, and more context makes
    classification worse, not better.
    """
    prior = state.get("messages", [])[-(n + 1):-1]
    if not prior:
        return state.get("query", "")
    lines = [f"{'BOT' if isinstance(m, AIMessage) else 'USER'}: {m.content}"
             for m in prior]
    return ("Conversation so far:\n" + "\n".join(lines)
            + f"\n\nLatest message to judge: {state.get('query', '')}")


FAMILY = (r"mum|mom|mother|dad|father|wife|husband|partner|son|daughter|brother|"
          r"sister|nan|nana|gran|grandma|grandad|grandmother|grandfather|"
          r"friend|dog|cat|pet|baby|child")

LOSS_REFERENCE = re.compile(
    rf"\b(my|our|his|her|their)\s+(?:{FAMILY})(?:'?s)?\b"
    r"|\b(funeral|cremation|burial|grave|ashes|grief|grieving|bereave\w*|"
    r"died|death|dying|passed away|loss|anniversary|memorial)\b"
    r"|\bmiss(?:ing)?\s+(her|him|them)\b"
    r"|\b(her|his|their)\s+(name|age|birthday|things|clothes|room|voice)\b",
    re.IGNORECASE,
)

PRONOUNS = re.compile(r"\b(she|he|they|her|him|his|their|hers|them)\b", re.IGNORECASE)


def references_loss(state: State) -> bool:
    """Deterministic check: is this message part of the grief conversation?

    An LLM classifier asked "is this about grief?" answers "unrelated" for
    "what was my mum's name and age" - in isolation that is a question about a
    name. One wrong classification then told a grieving person they were
    off-topic. Same principle as the crisis word list: where being wrong is
    costly, put a deterministic rule next to the model and let it win.
    """
    q = state.get("query") or ""
    if LOSS_REFERENCE.search(q):
        return True
    # A third-person pronoun on a bereavement service almost always means the
    # person who died. Being wrong costs a slightly-too-warm reply; the other
    # error tells a grieving person they are in the wrong place.
    if PRONOUNS.search(q):
        return True
    low = q.lower()
    for v in (state.get("facts") or {}).values():
        if isinstance(v, str) and len(v) > 2 and v.lower() in low:
            return True
    return False


def trace(node: str, t0: float, **extra) -> Dict[str, Any]:
    return {"node": node, "ms": round((time.perf_counter() - t0) * 1000), **extra}


# =============================================================================
# 10. GUARD NODES
# =============================================================================
def node_intake(state: State) -> dict:
    """Record the message and clear every per-turn field.

    With a checkpointer you must decide for each key whether it is per-turn or
    per-conversation. `messages` and `facts` persist; everything below resets.
    Getting that wrong produces bugs that only appear on turn two.
    """
    q = (state.get("query") or "").strip()
    return {
        "query": q,
        "messages": [HumanMessage(content=q)] if q else [],
        "guards": ["__RESET__"],
        "contributions": ["__RESET__"],
        "crisis": False,
        "crisis_certain": False,
        "subject": "",
        "context": [],
        "hops": 0,
        "next_agent": "",
        "draft": "",
        "critique": "",
        "attempts": 0,
        "answer": "",
        "escalated": False,
    }


def node_moderation(state: State) -> dict:
    """Is the message itself abusive? A per-message property, so no context.

    FAILS OPEN. Only an explicit "unsafe" blocks. An outage or a parse failure
    must never stonewall a grieving person. Contrast with node_crisis, which
    fails closed - different stakes, opposite default.
    """
    t0 = time.perf_counter()
    try:
        raw = clean(call(MODERATION_PROMPT, state.get("query", "")))
        verdict = last_label(raw, {"safe", "unsafe"}, "safe")
    except Exception as e:                              # noqa: BLE001
        record_model_failure(e)
        log.warning("moderation unavailable, failing open: %s", type(e).__name__)
        verdict = "safe"
    return {"guards": [{"name": "moderation", "passed": verdict != "unsafe",
                        "label": verdict,
                        "ms": round((time.perf_counter() - t0) * 1000)}]}


def node_validation(state: State) -> dict:
    """Is this still the grief conversation? Needs context to judge follow-ups."""
    t0 = time.perf_counter()
    try:
        raw = clean(call(VALIDATION_PROMPT, recent_context(state)))
        verdict = last_label(
            raw, {"valid", "unrelated", "nonsense", "offensive", "harmful"}, "valid")
    except Exception as e:                              # noqa: BLE001
        record_model_failure(e)
        log.warning("validation unavailable, defaulting to valid: %s", type(e).__name__)
        verdict = "valid"

    # Never tell someone they are off-topic about their own loss.
    if verdict in ("unrelated", "nonsense") and references_loss(state):
        log.info("validation override: %s -> valid (references the loss)", verdict)
        verdict = "valid"

    return {"verdict": verdict,
            "guards": [{"name": "validation", "passed": verdict == "valid",
                        "label": verdict,
                        "ms": round((time.perf_counter() - t0) * 1000)}]}


def node_crisis(state: State) -> dict:
    """Is someone at risk, and is it this person or somebody else?

    Three tiers combined with OR. The lexical tier needs no network, which is
    why crisis recall survives a total provider outage.
    """
    t0 = time.perf_counter()
    query = state.get("query", "")

    lexical_hit = lexical_crisis(query)                 # tier 0, instant
    lexical_subject = resolve_subject(query)            # deterministic

    model_says = False
    model_subject = None
    try:
        raw = clean(call(CRISIS_PROMPT, recent_context(state)))
        model_says = last_label(raw, {"risk", "norisk"}, "norisk") == "risk"
        if lexical_hit or model_says:
            sub_raw = clean(call(SUBJECT_PROMPT, query))
            model_subject = last_label(
                sub_raw, {"selfharm", "thirdparty", "neither"}, None)
    except Exception as e:                              # noqa: BLE001
        # One failure fails closed for THIS message - the check was uncertain.
        # Sustained failure means the provider is down, so trust the lexical
        # tier rather than calling every user a crisis.
        model_says = not record_model_failure(e)

    is_crisis = lexical_hit or model_says

    # Deterministic subject wins where it is confident; the model only fills in
    # when the regex saw nothing either way.
    if lexical_subject != "neither":
        subject = lexical_subject
    elif model_subject in ("selfharm", "thirdparty"):
        subject = model_subject
    else:
        subject = "selfharm"
    return {
        "crisis": is_crisis,
        "crisis_certain": lexical_hit,
        "subject": subject if is_crisis else "neither",
        "guards": [{"name": "crisis", "passed": not is_crisis,
                    "label": ("crisis:" + subject) if is_crisis else "clear",
                    "ms": round((time.perf_counter() - t0) * 1000)}],
    }


# =============================================================================
# 11. ROUTING
# =============================================================================
GREETING_RE = re.compile(
    r"^\s*(hi+|hey+|hello+|heya|hiya|yo|howdy|"
    r"good\s*(morning|afternoon|evening)|"
    r"how are you|how'?s it going|how are things|are you (there|around)|"
    r"anyone (there|here)|thanks?|thank you|cheers|ok(ay)?)"
    r"[\s!.,]*$", re.IGNORECASE)


def is_greeting(text: str) -> bool:
    """Greetings and tiny openers ('hi', 'hello', 'thanks'). These are how a
    conversation starts, not off-topic questions - they deserve a warm welcome,
    not a brush-off."""
    t = (text or "").strip()
    if not t:
        return True
    return bool(GREETING_RE.match(t))


def node_gate(state: State) -> dict:
    """Runs AFTER all three guards, so it is the first point where all three
    results are visible - none of them could see each other.

    THE ORDER OF THESE CHECKS IS THE SAFETY DESIGN.
    """
    g = {x["name"]: x for x in state.get("guards", []) if isinstance(x, dict)}
    log.info("gate | %s", [(x["name"], x["label"], f"{x['ms']}ms")
                           for x in state.get("guards", []) if isinstance(x, dict)])

    subject = state.get("subject", "neither")

    # 1. Worried about someone else. Checked FIRST because the reply is
    #    addressed to a different person - getting this wrong tells a
    #    frightened relative that THEY are not a burden.
    if state.get("crisis") and subject == "thirdparty":
        return {"route": "thirdparty"}

    # 2. A deterministic first-person crisis hit outranks everything else.
    if state.get("crisis_certain"):
        return {"route": "crisis"}

    moderation_failed = not g.get("moderation", {}).get("passed", True)
    verdict = state.get("verdict", "valid")

    # 3. A MODEL-ONLY crisis flag on a message moderation calls abusive is
    #    almost always the risk model over-firing on violent language. Handing
    #    a helpline to someone demanding hate speech helps nobody.
    if moderation_failed and verdict in ("offensive", "harmful"):
        return {"route": "refuse"}

    if state.get("crisis"):
        return {"route": "crisis"}
    if moderation_failed:
        return {"route": "refuse"}

    if verdict in ("harmful", "offensive"):
        return {"route": "refuse"}

    if verdict == "unrelated":
        # A greeting or opener is not off-topic - it is how someone starts. Give
        # it a warm welcome that invites them to talk, not a brush-off.
        if is_greeting(state.get("query", "")):
            log.info("gate: greeting -> support (warm welcome)")
            return {"route": "support"}
        # Once we know who died, one "unrelated" verdict is far more likely to
        # be a classifier error than a real topic change - and the cost of being
        # wrong is telling a bereaved person they are in the wrong place.
        if state.get("facts") or references_loss(state):
            log.info("gate override: unrelated -> support (established conversation)")
            return {"route": "support"}
        return {"route": "offtopic"}

    return {"route": "support"}


def pick_route(state: State) -> str:
    return state.get("route", "support")


# =============================================================================
# 12. MEMORY AND RETRIEVAL NODES
# =============================================================================
def node_remember(state: State) -> dict:
    """Pull durable facts out of the message.

    On the SUPPORT branch only: the crisis path must stay the fastest route in
    the graph, and there is no point extracting biography from a message we are
    about to refuse.
    """
    t0 = time.perf_counter()
    query = state.get("query", "")
    if not might_have_facts(query):
        return {}                                       # no model call at all

    # Deterministic extraction first. A live run showed the model extracting
    # the timeframe from "I lost my wife Priya in the spring" but NOT the name -
    # the single most important field in the schema.
    facts = lexical_facts(query)

    try:
        parsed = sanitise_facts(parse_json_object(
            call(EXTRACT_PROMPT, query, max_tokens=400)))
        # The model may add fields the regex cannot see (cause, dates), but the
        # regex wins on the fields it is confident about.
        merged = {**parsed, **facts}
    except Exception as e:                              # noqa: BLE001
        record_model_failure(e)
        log.warning("fact extraction failed (non-fatal): %s", type(e).__name__)
        merged = facts

    if merged:
        log.info("remembered %s in %dms", merged,
                 round((time.perf_counter() - t0) * 1000))
    return {"facts": merged} if merged else {}


def node_retrieve(state: State) -> dict:
    """Grounding. Skipped for small talk."""
    t0 = time.perf_counter()
    query = state.get("query", "")
    if not needs_retrieval(query):
        return {"context": []}
    try:
        chunks = get_retriever().search(query, k=3)
    except Exception as e:                              # noqa: BLE001
        log.warning("retrieval failed (non-fatal): %s", e)
        chunks = []
    log.debug("retrieved %d chunks in %dms", len(chunks),
              round((time.perf_counter() - t0) * 1000))
    return {"context": chunks}


# =============================================================================
# 13. SPECIALISTS AND SUPERVISOR
# =============================================================================
#  remember -> retrieve -> supervisor --> emotional  --+
#                            ^  |         coping    --|
#                            |  |         practical --+--> back to supervisor
#                            |  |         resources --+
#                            |  +--> compose -> critic -> END
#                            +---------+
#
# Every specialist reports BACK to the hub. That is what makes this a
# supervisor and not a swarm - in a swarm the receiving agent stays active,
# and a specialist holding the conversation could bypass the guardrail layer.

SPECIALISTS = {"emotional", "coping", "practical", "resources"}

# Most grief messages need only the emotional specialist, and paying a model
# call to be told that is waste. These detect the only cases where another
# specialist could be warranted; if none fire, routing costs zero model calls.
COPING_HINTS = re.compile(
    r"\b(tips?|advice|advise|suggestions?|recommend|"
    r"what (can|should|do) i do|how (do|can) i (cope|manage|deal|get through|handle)|"
    r"help me (cope|get through|deal|manage)|"
    r"ways? to (cope|manage|deal|get through)|"
    r"anything (that|which) helps|what helps|how do people)\b", re.IGNORECASE)

PRACTICAL_HINTS = re.compile(
    r"\b(register(ing)?|registrar|death certificate|funeral|cremation|burial|"
    r"probate|will|estate|executor|inherit|employer|hr|bank|pension|benefits|"
    r"tell us once|paperwork|admin|arrange|arrangements|solicitor|"
    r"how do i tell|what do i do about|who do i (tell|inform|contact))\b",
    re.IGNORECASE)

RESOURCE_HINTS = re.compile(
    r"\b(support group|counsell?ing|counsell?or|therapy|therapist|gp|doctor|"
    r"charity|helpline|service|where can i|who can i (talk|speak)|"
    r"is there (anyone|anything|help)|need help|get help)\b", re.IGNORECASE)


def specialist_signals(text: str) -> set:
    out = set()
    if COPING_HINTS.search(text or ""):
        out.add("coping")
    if PRACTICAL_HINTS.search(text or ""):
        out.add("practical")
    if RESOURCE_HINTS.search(text or ""):
        out.add("resources")
    return out


def node_supervisor(state: State) -> dict:
    """Decide which specialist contributes next, or finish."""
    done = [c["agent"] for c in state.get("contributions", [])
            if isinstance(c, dict)]
    hops = state.get("hops", 0)

    if hops >= MAX_HOPS:
        return {"next_agent": "compose"}
    remaining = SPECIALISTS - set(done)
    if not remaining:
        return {"next_agent": "compose"}

    signals = specialist_signals(state.get("query", "")) - set(done)

    # FAST PATH: no model call. The supervisor only thinks when the decision is
    # genuinely ambiguous.
    if not signals:
        if not done:
            return {"next_agent": "emotional", "hops": hops + 1}
        return {"next_agent": "compose"}

    # The emotional specialist always leads, even when something else is needed.
    if "emotional" in remaining:
        return {"next_agent": "emotional", "hops": hops + 1}

    candidates = remaining & signals
    if not candidates:
        return {"next_agent": "compose"}

    try:
        raw = clean(call(SUPERVISOR_PROMPT,
                         f"User message: {state.get('query', '')}\n"
                         f"Already contributed: {done or 'none'}\n"
                         f"Candidates: {sorted(candidates)}"))
        choice = last_label(raw, candidates | {"finish"}, sorted(candidates)[0])
    except Exception as e:                              # noqa: BLE001
        record_model_failure(e)
        choice = sorted(candidates)[0]

    if choice == "finish" or choice not in candidates:
        return {"next_agent": "compose"}
    log.info("supervisor -> %s", choice)
    return {"next_agent": choice, "hops": hops + 1}


SUPERVISOR_PROMPT = """You coordinate a small team supporting someone grieving.

- emotional : sits with feelings. Acknowledgement and reflection.
- coping    : gentle wellbeing suggestions, WHEN THE PERSON HAS ASKED.
- practical : UK bereavement admin - registering a death, funerals, probate.
- resources : names kinds of UK support services.

Given the message and who has already contributed, decide who goes NEXT.

- MOST messages need ONLY "emotional". Do not over-delegate.
- Never pick the same specialist twice.
- Reply "finish" as soon as the contributions cover what was asked.

Reply with exactly one word on its own final line:
emotional, coping, practical, resources, or finish."""


def pick_specialist(state: State) -> str:
    return state.get("next_agent", "compose")


PACING_HINT = ("This person has been asked a question recently. Do not ask "
               "another one - just be present.")


def _pacing(state: State) -> str:
    """Stop the bot interrogating people.

    A reply that ends in a question every single turn reads as an interview,
    not company. If the last thing we said was a question, don't ask again.
    """
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage):
            return PACING_HINT if "?" in (m.content or "") else ""
    return ""


def _specialist(name: str, system: str, state: State, **fmt) -> dict:
    """Run one specialist and report BACK to the supervisor.

    Uses ask() - the big model with conversation history - not call(), which is
    one isolated question to the small model. Specialists WRITE; guards
    CLASSIFY. Getting that wrong left the specialists with no conversational
    thread at all.
    """
    recent = state.get("messages", [])[-KEEP_MESSAGES:]
    try:
        text = clean(ask(to_groq_messages(recent),
                         system=system.format(**fmt) if fmt else system))
    except Exception as e:                              # noqa: BLE001
        record_model_failure(e)
        log.warning("specialist %s failed: %s", name, type(e).__name__)
        text = ""
    return {"contributions": [{"agent": name, "content": text}]}


def node_emotional(state: State) -> dict:
    return _specialist("emotional", EMOTIONAL_PROMPT, state,
                       facts=format_facts(state.get("facts", {})),
                       context=format_context(state.get("context", [])),
                       pacing=_pacing(state))


def node_coping(state: State) -> dict:
    return _specialist("coping", COPING_PROMPT, state,
                       facts=format_facts(state.get("facts", {})),
                       context=format_context(state.get("context", [])))


def node_practical(state: State) -> dict:
    return _specialist("practical", PRACTICAL_PROMPT, state,
                       context=format_context(state.get("context", [])))


def node_resources(state: State) -> dict:
    return _specialist("resources", RESOURCES_PROMPT, state)


# =============================================================================
# 14. COMPOSE AND CRITIC
# =============================================================================
def node_compose(state: State) -> dict:
    """Merge the specialists into one voice.

    Skipped entirely when there is one contribution - which is most turns.
    Nothing to merge, so paying for a rewrite adds latency and risk.
    """
    contribs = [c for c in state.get("contributions", [])
                if isinstance(c, dict) and (c.get("content") or "").strip()]

    if not contribs:
        return {"draft": "I'm here. Tell me what's happening."}

    if len(contribs) == 1 and not state.get("critique"):
        return {"draft": contribs[0]["content"]}        # no model call

    blob = "\n\n".join(f"[{c['agent']}] {c['content']}" for c in contribs)
    system = COMPOSE_PROMPT
    if state.get("critique"):
        system += f"\n\nYour previous attempt was rejected: {state['critique']}"

    recent = state.get("messages", [])[-KEEP_MESSAGES:]
    try:
        merged = clean(ask(to_groq_messages(recent),
                           system=system + "\n\nContributions:\n" + blob))
    except Exception as e:                              # noqa: BLE001
        record_model_failure(e)
        merged = "\n\n".join(c["content"] for c in contribs)
    log.debug("composed %d contributions", len(contribs))
    return {"draft": merged}


def node_critic(state: State) -> dict:
    """Review the draft before it ships.

    Reflection works because reviewing is easier than writing - the same model
    that slips in a platitude will identify it immediately when shown the
    sentence.

    FAILS OPEN. A critic that cannot make itself understood must not block a
    reply to a grieving person.
    """
    draft = (state.get("draft") or "").strip()
    attempts = state.get("attempts", 0)

    def accept(text: str, why: str) -> dict:
        log.info("critic: %s", why)
        return {"answer": text, "messages": [AIMessage(content=text)]}

    if not draft:
        return accept("I'm here. Tell me what's happening.", "empty draft")

    # Bound first. An unbounded model-driven loop is an unbounded bill and a
    # request that never returns.
    if attempts >= MAX_CRITIQUE_ROUNDS:
        return accept(draft, f"ship after {attempts} rewrite(s)")

    try:
        # The critic must see the USER'S MESSAGE too, or it cannot tell
        # requested advice from unsolicited advice - and rejects both.
        raw = clean(call(CRITIC_PROMPT,
                         f"What the user said:\n{state.get('query', '')}\n\n"
                         f"Draft reply to review:\n{draft}",
                         max_tokens=400))
    except Exception as e:                              # noqa: BLE001
        record_model_failure(e)
        return accept(draft, "critic unavailable, shipping draft")

    verdict = last_label(raw, {"pass", "fail"}, "pass")
    if verdict == "pass":
        return accept(draft, f"pass on attempt {attempts + 1}")

    note = raw.rsplit("fail", 1)[0].strip()[-200:] or "too long or too advisory"
    log.info("critic: fail -> %s", note)
    return {"critique": note, "attempts": attempts + 1}


def after_critic(state: State) -> str:
    """`answer` is only set when the critic accepted."""
    return "done" if state.get("answer") else "retry"


# =============================================================================
# 15. TERMINAL NODES
# =============================================================================
def node_crisis_response(state: State) -> dict:
    """The model writes the human sentences; the FILE supplies the numbers.

    If the model call fails entirely the helplines still ship. The part that
    can save someone never depends on the model working.
    """
    try:
        warm = clean(ask(
            to_groq_messages(state.get("messages", [])[-KEEP_MESSAGES:]),
            system=CRISIS_SYSTEM.format(facts=format_facts(state.get("facts", {}))),
            max_tokens=400))
    except Exception as e:                              # noqa: BLE001
        record_model_failure(e)
        warm = ("Thank you for telling me that. What you're carrying sounds "
                "heavier than anyone should hold on their own tonight.")
    answer = warm.strip() + HELPLINES
    log.warning(json.dumps({"event": "crisis_escalation",
                            "session": (state.get("session_id") or "")[:8],
                            "subject": "selfharm"}))
    return {"answer": answer, "escalated": True,
            "messages": [AIMessage(content=answer)]}


def node_thirdparty_response(state: State) -> dict:
    """Someone worried about another person.

    Addressed to THEM, not the person they are worried about. Without this the
    bot says "you're not a burden" to someone frightened about their brother.
    """
    try:
        warm = clean(ask(
            to_groq_messages(state.get("messages", [])[-KEEP_MESSAGES:]),
            system=THIRDPARTY_SYSTEM, max_tokens=400))
    except Exception as e:                              # noqa: BLE001
        record_model_failure(e)
        warm = ("It's frightening to be the one who notices, and it says a lot "
                "that you're taking it seriously.")
    answer = warm.strip() + THIRDPARTY_HELPLINES
    log.warning(json.dumps({"event": "crisis_escalation",
                            "session": (state.get("session_id") or "")[:8],
                            "subject": "thirdparty"}))
    return {"answer": answer, "escalated": True,
            "messages": [AIMessage(content=answer)]}


def node_refuse(state: State) -> dict:
    """Deterministic. No model call - nothing to gain, and you do not want a
    model improvising on a harmful input."""
    a = ("I can't help with that one. If something painful is behind it, "
         "I'm still here.")
    return {"answer": a, "messages": [AIMessage(content=a)]}


def node_offtopic(state: State) -> dict:
    a = ("That's outside what I can help with, but I'm here if something "
         "heavier is sitting with you.")
    return {"answer": a, "messages": [AIMessage(content=a)]}


# =============================================================================
# 16. GRAPH
# =============================================================================
def build_graph(checkpointer):
    b = StateGraph(State)

    for name, fn in [
        ("intake", node_intake),
        ("moderation", node_moderation),
        ("validation", node_validation),
        ("crisis_guard", node_crisis),        # NOT "crisis" - that is a state key
        ("gate", node_gate),
        ("remember", node_remember),
        ("retrieve", node_retrieve),
        ("supervisor", node_supervisor),
        ("emotional", node_emotional),
        ("coping", node_coping),
        ("practical", node_practical),
        ("resources", node_resources),
        ("compose", node_compose),
        ("critic", node_critic),
        ("crisis_response", node_crisis_response),
        ("thirdparty_response", node_thirdparty_response),
        ("refuse", node_refuse),
        ("offtopic", node_offtopic),
    ]:
        b.add_node(name, fn)

    b.add_edge(START, "intake")

    # Three arrows OUT of intake  = the guards run at the same time.
    # Three arrows INTO gate      = gate waits until all three have finished.
    for guard in ("moderation", "validation", "crisis_guard"):
        b.add_edge("intake", guard)
        b.add_edge(guard, "gate")

    b.add_conditional_edges("gate", pick_route, {
        "support":    "remember",
        "crisis":     "crisis_response",
        "thirdparty": "thirdparty_response",
        "refuse":     "refuse",
        "offtopic":   "offtopic",
    })

    # Support branch: remember, ground, then the team.
    b.add_edge("remember", "retrieve")
    b.add_edge("retrieve", "supervisor")

    b.add_conditional_edges("supervisor", pick_specialist, {
        "emotional": "emotional",
        "coping":    "coping",
        "practical": "practical",
        "resources": "resources",
        "compose":   "compose",
    })

    # Every specialist reports BACK to the hub.
    for s in ("emotional", "coping", "practical", "resources"):
        b.add_edge(s, "supervisor")

    # compose -> critic, and the critic can point BACK. An edge going backwards
    # is all a loop is.
    b.add_edge("compose", "critic")
    b.add_conditional_edges("critic", after_critic, {
        "retry": "compose",
        "done":  END,
    })

    for terminal in ("crisis_response", "thirdparty_response", "refuse", "offtopic"):
        b.add_edge(terminal, END)

    return b.compile(checkpointer=checkpointer)


# --- persistence -------------------------------------------------------------
# Absolute path so the database is always beside app.py. A relative path
# silently creates a SECOND empty database if you launch from elsewhere, which
# looks exactly like "memory stopped working after a restart".
#
# check_same_thread=False: Flask serves each request on a different thread.
# isolation_level=None:    autocommit, so a write is durable immediately rather
#                          than waiting for a commit that never comes on Ctrl+C.
_conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, isolation_level=None)
_conn.execute("PRAGMA journal_mode=WAL")
_conn.execute("PRAGMA synchronous=FULL")
atexit.register(lambda: (_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)"),
                         _conn.close()))

graph = build_graph(SqliteSaver(_conn))
log.info("memory db: %s", DB_PATH)


# =============================================================================
# 17. PUBLIC API
# =============================================================================
SAFE_FALLBACK = ("I'm having trouble finding my words just now. Can you give me "
                 "a moment and try again?")


def ask_bot(query: str, session_id: str = "default") -> Dict[str, Any]:
    """One turn. Never raises - a grief bot that 500s is worse than one that
    says something plain."""
    t0 = time.perf_counter()
    try:
        out = graph.invoke(
            {"query": query, "session_id": session_id},
            {"configurable": {"thread_id": session_id}},
        )
        answer = (out.get("answer") or "").strip() or SAFE_FALLBACK
        return {
            "response": answer,
            "route": out.get("route", "unknown"),
            "escalated": bool(out.get("escalated")),
            "degraded": is_degraded(),
            "latency_ms": round((time.perf_counter() - t0) * 1000),
        }
    except Exception as e:                              # noqa: BLE001
        log.exception("turn failed for session %s", session_id[:8])
        return {
            "response": SAFE_FALLBACK,
            "route": "error",
            "escalated": False,
            "degraded": True,
            "latency_ms": round((time.perf_counter() - t0) * 1000),
        }


def session_facts(session_id: str) -> Dict[str, Any]:
    """What the bot remembers. A demo aid, and a GDPR subject-access route -
    people are entitled to see the personal data you hold about them."""
    try:
        snap = graph.get_state({"configurable": {"thread_id": session_id}})
        values = snap.values or {}
        return {"facts": values.get("facts", {}),
                "turns": len(values.get("messages", [])) // 2}
    except Exception:                                   # noqa: BLE001
        return {"facts": {}, "turns": 0}


def forget_session(session_id: str) -> bool:
    """Delete everything held for one conversation.

    'Delete my data' has to actually clear the checkpointer rows, not just hide
    them. This is special-category data under UK GDPR.
    """
    try:
        with _conn:
            _conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (session_id,))
            _conn.execute("DELETE FROM writes WHERE thread_id = ?", (session_id,))
        log.info("erased session %s", session_id[:8])
        return True
    except Exception as e:                              # noqa: BLE001
        log.error("erase failed: %s", e)
        return False


# =============================================================================
# 18. WEB LAYER
# =============================================================================
# Thin on purpose: read JSON, call ask_bot, return JSON. All the thinking is in
# the graph.

app = Flask(__name__)

# The original allowed any origin while also setting a session cookie. A
# wildcard origin plus credentials is the combination that lets any website on
# the internet make requests on a user's behalf.
ALLOWED_ORIGINS = [o.strip() for o in
                   os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,"
                             "http://localhost:5173").split(",") if o.strip()]
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}},
     supports_credentials=True)

# --- session identity --------------------------------------------------------
# The original took session_id straight from the request body. Since session_id
# IS the memory key, anyone who learned another user's id could read their grief
# conversation. The client no longer names its own thread: it presents a signed
# token and the server derives the id. In production this becomes real auth and
# the id derives from an authenticated subject claim.
_SECRET = (os.getenv("SESSION_SECRET") or "").encode() or os.urandom(32)
REQUIRE_TOKEN = os.getenv("REQUIRE_SESSION_TOKEN", "1") != "0"


def mint_token() -> str:
    sid = uuid.uuid4().hex
    sig = hmac.new(_SECRET, sid.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{sid}.{sig}"


def verify_token(token: str) -> Optional[str]:
    try:
        sid, sig = token.split(".", 1)
    except (ValueError, AttributeError):
        return None
    expected = hmac.new(_SECRET, sid.encode(), hashlib.sha256).hexdigest()[:32]
    # compare_digest, not ==, to avoid a timing side channel.
    return sid if hmac.compare_digest(sig, expected) else None


def resolve_session() -> tuple:
    """Returns (session_id, error_response)."""
    token = request.headers.get("X-Session-Token")
    if token:
        sid = verify_token(token)
        if sid:
            return sid, None
        return None, (jsonify({"error": "invalid session token"}), 401)
    if REQUIRE_TOKEN:
        return None, (jsonify({"error": "missing X-Session-Token; "
                                        "POST /session first"}), 401)
    data = request.get_json(silent=True) or {}
    return (data.get("session_id") or "default"), None


# --- simple per-session rate limit -------------------------------------------
_HITS: Dict[str, deque] = {}
_HITS_LOCK = threading.Lock()
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "20"))


def rate_limited(session_id: str) -> bool:
    now = time.monotonic()
    with _HITS_LOCK:
        q = _HITS.setdefault(session_id, deque())
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            return True
        q.append(now)
        return False


@app.route("/health")
def health():
    return {"status": "ok", "degraded": is_degraded(),
            "tokens_used_last_min": BUDGET.used()}, 200


@app.route("/session", methods=["POST"])
def create_session():
    return jsonify({"session_token": mint_token()})


@app.route("/ask", methods=["POST"])
def ask_route():                       # not ask() - that name is the LLM helper
    session_id, err = resolve_session()
    if err:
        return err

    data = request.get_json(silent=True) or {}
    query = (data.get("question") or "").strip()
    if not query:
        return jsonify({"error": "Missing question"}), 400
    if len(query) > 4000:
        return jsonify({"error": "Message too long"}), 400
    if rate_limited(session_id):
        return jsonify({"error": "Too many messages, please slow down"}), 429

    result = ask_bot(query, session_id)
    result["session_id"] = session_id
    return jsonify(result)


@app.route("/session/state", methods=["GET"])
def state_route():
    session_id, err = resolve_session()
    if err:
        return err
    return jsonify(session_facts(session_id))


@app.route("/session/forget", methods=["POST"])
def forget_route():
    session_id, err = resolve_session()
    if err:
        return err
    ok = forget_session(session_id)
    return jsonify({"erased": ok}), (200 if ok else 500)


def warmup() -> None:
    """Build the index once at startup rather than inside the first request."""
    try:
        get_retriever()
    except Exception as e:                              # noqa: BLE001
        log.warning("warmup failed (non-fatal): %s", e)


if __name__ == "__main__":
    warmup()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
