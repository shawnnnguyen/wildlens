"""Prompt templates and exemplars for the check_relevance node."""
from __future__ import annotations

_RELEVANCE_PROMPT = (
    "You are a strict binary classifier for a wildlife safari guide chatbot. "
    "Decide whether the following visitor message is about wildlife, animals, "
    "nature, or the safari tour itself, or whether it is completely unrelated "
    "(e.g. technical support, general trivia, unrelated requests).\n\n"
    "{context_line}"
    "Reply with exactly one word: ON_TOPIC or OFF_TOPIC.\n\n"
    "Message: {message}"
)

# Inserted into _RELEVANCE_PROMPT only when session_species is non-empty —
# without this, a contextual pronoun follow-up ("can it swim?") has no
# keyword/species-alias match and reaches the LLM with zero session context,
# risking a false OFF_TOPIC refusal of a legitimate follow-up question.
_RELEVANCE_CONTEXT_LINE = (
    "This tourist has recently been discussing: {species}. If the message is "
    "a pronoun or short contextual follow-up about one of those animals (e.g. "
    "\"can it swim?\", \"how big is it?\", \"what about that one\"), treat it "
    "as ON_TOPIC.\n\n"
)

# ── Embedding-based semantic relevance classifier exemplars ───────────────────
# A dynamic middle tier between the free keyword heuristics and the LLM
# fallback: instead of hand-maintaining an ever-growing keyword list to catch
# every possible phrasing of a real wildlife question (or every possible
# off-topic one), embed the message and compare it against small curated
# exemplar clusters using the same local, zero-cost embedding model already
# loaded for RAG (see rag/factory.py — HuggingFace all-MiniLM-L6-v2, runs on
# CPU, no network call, normalize_embeddings=True). This generalizes to
# phrasing the exemplars don't literally contain (e.g. "will it hurt me"
# scores close to the on-topic cluster despite sharing no words with "does it
# bite") — something a keyword list structurally cannot do.
_ON_TOPIC_EXEMPLARS = [
    "does it bite",
    "can it swim",
    "what does it eat",
    "is it dangerous",
    "how fast can it run",
    "where does it sleep",
    "how big is it",
    "does it attack humans",
    "can they climb trees",
    "is it poisonous or venomous",
    "how long do they live",
    "do they hunt in packs",
    "what is its habitat",
    "are they endangered",
    "how do they communicate",
    "what sound does it make",
    "is it nocturnal",
    "how many babies do they have",
]

_OFF_TOPIC_EXEMPLARS = [
    "what's the weather like today",
    "where is my mom",
    "what time is it",
    "where's the bathroom",
    "who is the president",
    "can you help me with my homework",
    "what's the wifi password",
    "how do I get to the hotel",
    "what's for dinner",
    "can you recommend a restaurant",
    "tell me a joke",
    "what's the capital of France",
]
# Deliberately NOT included above: tour-logistics questions ("how much does
# this tour cost", "can I get a refund") — _RELEVANCE_PROMPT explicitly
# defines on-topic as "wildlife, animals, nature, OR THE SAFARI TOUR ITSELF",
# so curating those as off-topic exemplars would contradict the LLM tier's
# own definition and (since this tier runs BEFORE the LLM and can return a
# unilateral off_topic verdict) bypass its fail-open safety net entirely.
