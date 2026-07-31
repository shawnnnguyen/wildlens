"""NODE — RAG retrieval → hybrid BM25 + semantic + web search for verified guidebook facts."""
from __future__ import annotations

import hashlib
import logging
import re

from langchain_core.retrievers import BaseRetriever

from wildlens.data.species_lookup import canonical_common_name
from wildlens.nodes._shared import wrap_untrusted
from wildlens.rag import _EnsembleRetriever
from wildlens.state import WildlensState

log = logging.getLogger("safari_guide.nodes")


def node_retrieve_information(
    state: WildlensState,
    retriever: BaseRetriever,
) -> dict:
    """
    Hybrid BM25 + semantic + web retrieval for verified guidebook facts.
    Query blends species name + user_message for context-aware retrieval
    on text follow-up turns. On a fresh photo identification (no follow_up
    yet), the query is biased toward diet/circadian-rhythm topics instead of
    just the species name, since that's the biographical info the persona
    node is expected to lead with and the curated corpus has no dedicated
    field for it — see node_generate_guide_persona.

    Species resolution prefers message_relevance.mentioned_species (set by
    node_check_relevance when the current text message names a specific
    animal) over identification_result — otherwise a mid-session pivot
    ("we're discussing a lion, tourist asks about elephants") would keep
    retrieving/filtering on the previously-identified animal instead of the
    one actually being asked about.
    """
    log.info("▶ NODE  retrieve_information")
    mentioned_species = state.get("message_relevance", {}).get("mentioned_species")
    if mentioned_species:
        common_name = mentioned_species
    else:
        species     = state.get("identification_result", {}).get("species", "")
        # Canonicalize against species_list.json first so casing/whitespace drift in
        # Gemini's freeform output doesn't silently drop the species filter (falls
        # back to the naive split for genuinely unlisted/novel species — see #10).
        common_name = (canonical_common_name(species) if species else None) or species.split("(")[0].strip()
    follow_up   = state.get("user_message", "")
    if follow_up:
        query = f"{common_name} {follow_up}".strip()
    else:
        query = f"{common_name} diet feeding behavior circadian rhythm daily activity pattern".strip()
    query = query or "safari wildlife"

    if isinstance(retriever, _EnsembleRetriever):
        docs = retriever.retrieve(query, species=common_name or None)
        if common_name:
            _enqueue_enrichment(retriever, common_name, query, docs)
    else:
        docs = retriever.invoke(query)
    facts = "\n\n---\n\n".join(_format_fact(d) for d in docs)
    log.info("   → %d docs retrieved | query: '%s'", len(docs), query)
    return {"retrieved_facts": facts}


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify_section(text: str, max_len: int = 60) -> str:
    """Turn a retrieval query into a short, stable section label for enrichment writes."""
    slug = _SLUG_RE.sub("_", text.lower()).strip("_")
    return slug[:max_len] or "web_enrichment"


def _enqueue_enrichment(retriever: _EnsembleRetriever, species: str, query: str, docs: list) -> None:
    """
    Persist any web-sourced fact used to answer this turn's query back into
    the corpus (see _EnsembleRetriever.enrich_async). By construction, Tavily
    only fires for a given retrieval when the local corpus was already thin
    (see ranking.py's _local_corpus_is_thin gating), so every web doc here is
    filling a real gap worth keeping for next time — not a redundant re-save
    of something the guidebook already covered.

    Each doc gets its own section slug (query slug + a stable hash of its
    URL/content) rather than sharing one section per query — otherwise every
    web doc from the same query would upsert into the same (species, section,
    source) row and each write would silently overwrite the previous one,
    keeping only the last of several retrieved facts. sha1 (not the builtin
    hash()) is used so the same URL maps to the same section across process
    restarts — PYTHONHASHSEED randomizes hash() per-process, which would
    otherwise turn a repeat scrape into a new row instead of an idempotent
    overwrite.
    """
    base_section = _slugify_section(query)
    for doc in docs:
        if doc.metadata.get("source") == "web":
            identity = doc.metadata.get("url") or doc.page_content
            digest   = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:10]
            retriever.enrich_async(
                species=species,
                section=f"{base_section}__{digest}",
                content=doc.page_content,
                source_url=doc.metadata.get("url", ""),
                title=doc.metadata.get("title", ""),
            )


def _format_fact(doc) -> str:
    """
    Label each fact by provenance so the persona LLM can tell curated
    guidebook data (vetted at ingest time) apart from live/cached Tavily web
    results, and prefer the former on conflict — see node_generate_guide_persona.

    'web_enriched' (a past Tavily result written back by enrich_async, now
    resurfacing via the BM25 rebuild or the web_cache Pinecone namespace)
    must be labeled Web here too, not Guidebook — it never went through the
    vetting the ingest pipeline gives curated content, and mislabeling it
    would make the persona prompt's "prefer Guidebook on safety conflicts"
    instruction trust unverified scraped text.
    """
    source = doc.metadata.get("source")
    if source in ("web", "web_enriched"):
        # label (a page title or URL scraped by Tavily) is exactly as
        # attacker-controlled as page_content — wrap both together rather
        # than leaving label to be interpolated into trusted prompt territory.
        label = doc.metadata.get("title") or doc.metadata.get("url") or doc.metadata.get("species") or "cached web result"
        labeled_content = f"{label}\n{doc.page_content}"
        return f"[Source: Web]\n{wrap_untrusted(labeled_content)}"
    species = doc.metadata.get("species") or "Guidebook"
    return f"[Source: Guidebook — {species}]\n{doc.page_content}"
