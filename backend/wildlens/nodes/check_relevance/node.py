"""
NODE — Check message relevance (text turns only)

Gate for text turns only — classifies user_message into "on_topic" /
"small_talk" / "off_topic" before any RAG retrieval or persona generation is
attempted, so a nonsense or off-topic message doesn't pay for either.
"""
from __future__ import annotations

import logging
import re
import weakref
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from wildlens.data.species_lookup import canonical_common_name, find_mentioned_species
from wildlens.nodes._shared import _invoke_with_retry, wrap_untrusted
from wildlens.nodes.check_relevance.prompts import (
    _OFF_TOPIC_EXEMPLARS,
    _ON_TOPIC_EXEMPLARS,
    _RELEVANCE_CONTEXT_LINE,
    _RELEVANCE_PROMPT,
)
from wildlens.state import WildlensState

log = logging.getLogger("safari_guide.nodes")

# Generic wildlife/safari vocabulary — a match here (without a specific
# species mention) is enough to treat a message as on_topic without paying
# for the LLM fallback below. Word-boundary matched, not substring (see
# _contains_any_word) — a naive substring check would false-positive on
# unrelated text purely by bad luck (e.g. "hi" inside "this").
_WILDLIFE_KEYWORDS = {
    "diet", "eat", "eats", "eating", "feed", "feeding", "prey", "predator",
    "predators", "habitat", "territory", "nocturnal", "diurnal", "hunt",
    "hunts", "hunting", "hunter", "dangerous", "danger", "threat",
    "threatened", "endangered", "conservation", "extinct", "extinction",
    "pack", "herd", "pride", "migration", "migrate", "safari", "animal",
    "animals", "wildlife", "species", "speed", "lifespan", "weight", "size",
    "camouflage", "breed", "breeding", "mate", "mating", "cub", "cubs",
    "calf", "calves", "sleep", "sleeps", "active", "nest", "nesting",
    "poaching", "savanna", "savannah", "serengeti", "tour", "guide",
}

# Small talk directed at Kate personally — treated as its own bucket (not
# folded into on_topic) so it can skip retrieve_information entirely; see
# node_check_relevance and route_after_relevance in graphs.py.
_SMALL_TALK_PHRASES = {
    "hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye",
    "good morning", "good afternoon", "good evening", "how are you",
}

# Filler/intensifier words stripped before comparing a message against
# _SMALL_TALK_PHRASES — lets "thanks so much!" and "hey there!" still count
# as pure small talk without requiring an exact phrase match, while any
# OTHER leftover word (a real question, a name, a topic) fails the match.
_SMALL_TALK_FILLER_WORDS = {
    "so", "much", "very", "really", "a", "lot", "too", "again", "there",
    "now", "just", "man", "buddy", "friend",
}


def _contains_any_word(text: str, phrases: set[str]) -> bool:
    """
    Word-boundary match against any phrase in *phrases* (case-insensitive).
    Deliberately not substring matching — see find_mentioned_species's
    docstring for why that's unsafe (e.g. "ass" inside "password").
    """
    lowered = text.lower()
    return any(re.search(rf"\b{re.escape(phrase)}\b", lowered) for phrase in phrases)


def _is_small_talk(text: str) -> bool:
    """
    True only when the ENTIRE message (modulo punctuation and a handful of
    filler words) IS a small-talk phrase — deliberately NOT a contains-
    anywhere match. A message like "hi, where is my mom?" or "hey, does it
    bite?" contains "hi"/"hey" but is not small talk; a fixed phrase list
    can never enumerate every way a real (possibly entirely off-topic)
    question might be phrased, so instead of trying to keyword-match every
    such case, this makes the free fast path strict enough that anything
    with real additional content simply isn't "small talk" and falls
    through to the embedding classifier / LLM fallback below, which are
    actually equipped to judge arbitrary phrasing.
    """
    words = re.findall(r"[a-z']+", text.lower())
    cleaned = " ".join(w for w in words if w not in _SMALL_TALK_FILLER_WORDS)
    return cleaned in _SMALL_TALK_PHRASES


def _is_wildlife_related(text: str) -> bool:
    return _contains_any_word(text, _WILDLIFE_KEYWORDS)


def _llm_classify_relevance(
    message: str, llm: BaseChatModel, session_species: list[str] | None = None,
) -> tuple[str, bool]:
    """
    Cheap LLM fallback for messages the free heuristics above can't classify.

    session_species (most-recent-first, see node_check_relevance) is threaded
    into the prompt as a disambiguation hint only — it does not change the
    strict ON_TOPIC/OFF_TOPIC output contract, just makes the classifier aware
    that a pronoun/contextual follow-up may refer to a recently-discussed
    animal rather than being genuinely unrelated.

    Returns (status, classification_failed). Fails OPEN (status="on_topic")
    on any API error, empty response, or a reply that doesn't clearly start
    with "OFF" — a wasted RAG+generation call on a rare weird message is
    cheaper than wrongly refusing a legitimate wildlife question. Returns
    classification_failed=True only on an actual error (not a merely-unclear
    reply) so a persistently broken classifier is observable via
    message_relevance rather than silently defaulting open forever with no
    signal that the gate has effectively stopped doing anything.
    """
    try:
        context_line = (
            _RELEVANCE_CONTEXT_LINE.format(species=", ".join(session_species))
            if session_species else ""
        )
        response = _invoke_with_retry(llm, [HumanMessage(
            content=_RELEVANCE_PROMPT.format(context_line=context_line, message=wrap_untrusted(message))
        )])
        content = (response.content or "").strip()
        first_token = content.split()[0].upper() if content else ""
        status = "off_topic" if first_token.startswith("OFF") else "on_topic"
        return status, False
    except Exception as exc:
        log.warning("   → relevance classification failed, defaulting to on_topic: %s", exc)
        return "on_topic", True


# Minimum cosine-similarity gap between the closer and farther exemplar
# cluster for the embedding tier to decide outright. Below this margin the
# message is genuinely ambiguous, not confidently either — falls through to
# the LLM instead of trusting a low-confidence embedding call.
_RELEVANCE_MARGIN = 0.08

# Cached per embeddings-instance (WeakKeyDictionary, not id()-keyed — avoids
# any risk of a garbage-collected instance's id being reused by an unrelated
# later instance and serving it stale exemplar vectors from a different
# embedding space; entries are dropped automatically when the embeddings
# instance itself is GC'd). The real app constructs exactly one embeddings
# instance for the process lifetime (see rag/factory.py's init_rag(), never
# explicitly torn down), so this caches exactly once in production; each
# test's fake-embeddings instance gets its own independent entry.
_exemplar_embedding_cache: "weakref.WeakKeyDictionary[Any, dict[str, list[list[float]]]]" = weakref.WeakKeyDictionary()


def _get_exemplar_embeddings(embeddings) -> dict[str, list[list[float]]]:
    if embeddings not in _exemplar_embedding_cache:
        _exemplar_embedding_cache[embeddings] = {
            "on_topic":  embeddings.embed_documents(_ON_TOPIC_EXEMPLARS),
            "off_topic": embeddings.embed_documents(_OFF_TOPIC_EXEMPLARS),
        }
    return _exemplar_embedding_cache[embeddings]


def _max_cosine_similarity(query_vector: list[float], exemplar_vectors: list[list[float]]) -> float:
    """Dot product against pre-normalized vectors IS cosine similarity —
    see rag/factory.py's encode_kwargs={"normalize_embeddings": True}.
    Raises on a vector-length mismatch rather than letting zip() silently
    truncate to a plausible-but-wrong score — a misconfigured/swapped
    embedding backend should surface loudly (caught by the caller's
    try/except and logged) rather than produce a silently bogus verdict."""
    for vec in exemplar_vectors:
        if len(vec) != len(query_vector):
            raise ValueError(
                f"embedding dimension mismatch: query has {len(query_vector)}, exemplar has {len(vec)}"
            )
    return max(sum(q * e for q, e in zip(query_vector, vec)) for vec in exemplar_vectors)


def _embedding_classify_relevance(message: str, embeddings: Any) -> str | None:
    """
    Embeds *message* and compares it against the on-topic/off-topic exemplar
    clusters above. Returns "on_topic"/"off_topic" when one cluster is
    confidently closer (by at least _RELEVANCE_MARGIN), or None when the
    message is ambiguous or the embedding call itself fails — callers must
    fall through to _llm_classify_relevance in either case rather than
    trusting a low-confidence or missing signal.
    """
    try:
        query_vector = embeddings.embed_query(message)
        exemplars = _get_exemplar_embeddings(embeddings)
        on_score  = _max_cosine_similarity(query_vector, exemplars["on_topic"])
        off_score = _max_cosine_similarity(query_vector, exemplars["off_topic"])
    except Exception as exc:
        log.warning("   → embedding relevance classification failed, deferring to LLM: %s", exc)
        return None
    if on_score - off_score >= _RELEVANCE_MARGIN:
        return "on_topic"
    if off_score - on_score >= _RELEVANCE_MARGIN:
        return "off_topic"
    return None


def node_check_relevance(state: WildlensState, llm: BaseChatModel, embeddings: Any = None) -> dict:
    """
    Gate for text turns only (see route_entry/route_after_relevance in
    graphs.py) — classifies user_message into "on_topic" / "small_talk" /
    "off_topic" before any RAG retrieval or persona generation is attempted,
    so a nonsense or off-topic message doesn't pay for either.

    Layered cheapest-first so the LLM is only invoked for genuinely
    ambiguous messages. Species mention and wildlife-keyword checks run
    BEFORE the small-talk check — not after — because a message like "Hi
    Kate, what do lions eat?" or "Thanks! What about elephants?" contains a
    small-talk phrase AND a real question; checking small talk first would
    skip retrieval for a message that clearly needs it.
      1. species-mention match (free) — also resolves which species this
         turn's retrieval should target, overriding identification_result
         for cross-animal follow-ups (see node_retrieve_information)
      2. wildlife-keyword match (free)
      3. small-talk phrase match (free) — _is_small_talk requires the
         ENTIRE message (modulo filler words/punctuation) to be a known
         phrase, not merely contain one, so "hi, where is my mom?" or "hey,
         does it bite?" fall through to step 4 rather than being
         short-circuited to a warm reply that silently skips retrieval.
      4. embedding similarity vs. curated on-topic/off-topic exemplar
         clusters (cheap, local, no network call — see
         _embedding_classify_relevance) — dynamically handles phrasing a
         fixed keyword list can't enumerate, resolving common follow-ups
         ("does it bite?", "can it swim?") without an LLM call at all. Only
         runs when embeddings is provided (None gracefully skips to step 5,
         e.g. when the retriever backing this graph has no embedding model).
      5. LLM classification (cheap, rare — see _llm_classify_relevance) for
         whatever step 4 left ambiguous (or skipped), given session_species
         as context for pronoun/contextual follow-ups.
    """
    log.info("▶ NODE  check_relevance")
    message = state.get("user_message", "")

    # Most-recent-first, canonicalized — used to break ties when a message
    # mentions an ambiguous alias shared by more than one curated species
    # (e.g. "gazelle" -> Thomson's/Grant's) — see find_mentioned_species.
    session_species: list[str] = []
    for h in reversed(state.get("identification_history", [])):
        canon = canonical_common_name(h.get("species", ""))
        if canon and canon not in session_species:
            session_species.append(canon)

    mentioned = find_mentioned_species(message, session_species)
    if mentioned:
        log.info("   → on_topic (mentions %s)", mentioned)
        return {"message_relevance": {"status": "on_topic", "mentioned_species": mentioned, "classification_failed": False}}

    if _is_wildlife_related(message):
        log.info("   → on_topic (wildlife keyword match)")
        return {"message_relevance": {"status": "on_topic", "mentioned_species": None, "classification_failed": False}}

    if _is_small_talk(message):
        log.info("   → small_talk (phrase match)")
        return {"message_relevance": {"status": "small_talk", "mentioned_species": None, "classification_failed": False}}

    if embeddings is not None:
        embedded_status = _embedding_classify_relevance(message, embeddings)
        if embedded_status is not None:
            log.info("   → %s (embedding similarity)", embedded_status)
            return {"message_relevance": {"status": embedded_status, "mentioned_species": None, "classification_failed": False}}

    status, failed = _llm_classify_relevance(message, llm, session_species)
    log.info("   → %s (LLM fallback%s)", status, ", classification failed" if failed else "")
    return {"message_relevance": {"status": status, "mentioned_species": None, "classification_failed": failed}}
